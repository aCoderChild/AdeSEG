"""Pinned Lean-SAM2 scoring + Efficient-SAM2 sparse retrieval on native SAM2.

Author methods execute unchanged. Project adaptation: match each token to soft
foreground/context roles, using both sets of fixed prompt anchors. No training,
frame selection, confidence decision, destructive bank pruning or GT after prompt.
"""

import ast
from functools import lru_cache
from types import FunctionType, MethodType, SimpleNamespace

import torch
import torch.nn.functional as F

from modeling.complementary_memory.integration import NativeComplementaryRead, _tensor_sha256
from modeling.region_attention import author_attention_forward
from modeling.region_tokens import _verified_source, EFFICIENT_SAM2_REVISION
from modeling.sparse_memory import sparse_forward
from scripts.bank_integrity import fingerprint

LEAN_REVISION = "8d1cf2618519d291905d22fe859b8302476d7f50"
MODES = ("identity", "usage", "foreground", "role_blind", "regional")


@lru_cache(maxsize=1)
def author_selectors():
    names = {
        "_flatten_similarity_tokens", "_compute_similarity_maxcos",
        "_compute_anchor_similarity_maxcos", "_select_diverse_anchor_local",
        "_prompt_mask_to_mem_token_idx", "_prepare_usage_anchor_scores",
        "_canonicalize_usage_anchor_fusion_mode", "_fuse_usage_anchor",
    }
    path, source = _verified_source("Lean-SAM2", LEAN_REVISION,
                                    "sam2/modeling/sam2_base.py")
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "SAM2Base")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError("Pinned Lean-SAM2 selector methods changed")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    adapter = SimpleNamespace(anchor_lambda=2.0, usage_anchor_fusion_mode="mul")
    for name in names:
        setattr(adapter, name, MethodType(namespace[name], adapter))
    path, source = _verified_source("Efficient-SAM2", EFFICIENT_SAM2_REVISION,
                                    "sam2/modeling/sam2_base.py")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "get_topk_mask")
    env = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return adapter, env["get_topk_mask"]


def prompt_anchors(features, probabilities, usage):
    """Reuse Lean's mask-to-cell and diverse-anchor selection for both roles."""
    author, _ = author_selectors()
    anchors = []
    for support in (probabilities, 1 - probabilities):
        ids = author._prompt_mask_to_mem_token_idx(support.reshape(1, 1, 64, 64), 64, 64)
        if not ids.numel():
            anchors.append(features[:0])
            continue
        count = min(ids.numel(), 64, max(8, round(ids.numel() * .05)))
        local = author._select_diverse_anchor_local(features[ids], usage[ids], count)
        anchors.append(features[ids[local]].detach().clone())
    if not anchors[0].numel():
        raise RuntimeError("Prompt has no foreground anchor on the memory grid")
    return tuple(anchors)


def select_tokens(features, probabilities, usage, anchors, mode, keep_fraction=.25):
    if mode not in MODES or not 0 < keep_fraction <= 1:
        raise ValueError("Invalid selector or fixed retention fraction")
    if features.ndim != 2 or usage.shape != features.shape[:1] or probabilities.shape != usage.shape:
        raise ValueError("Token features, probabilities and importance must align")
    if (not torch.isfinite(features).all() or not torch.isfinite(usage).all()
            or not torch.isfinite(probabilities).all() or (usage < 0).any()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Invalid selector features/importance/occupancy")
    author, topk = author_selectors()
    count = features.shape[0]
    if mode == "identity" or keep_fraction == 1:
        return torch.ones(count, device=features.device, dtype=torch.bool)
    scores = usage
    if mode in ("foreground", "role_blind", "regional"):
        foreground, background = anchors
        sf = author._compute_anchor_similarity_maxcos(features, foreground)
        if sf is None:
            raise RuntimeError("Missing foreground anchors")
        affinity = sf
        if mode in ("role_blind", "regional") and background.numel():
            sb = author._compute_anchor_similarity_maxcos(features, background)
            # New integration rule: preserve agreement with BOTH predicted roles.
            # It is a soft mixture, not a boundary class or keep/drop label.
            affinity = (torch.maximum(sf, sb) if mode == "role_blind" else
                        probabilities * sf + (1 - probabilities) * sb)
        scores = author._fuse_usage_anchor(usage, affinity, anchor_lambda=2., fusion_mode="mul")
    selected = topk(scores, max(1, int(count * keep_fraction)))
    if selected.sum().item() != max(1, int(count * keep_fraction)):
        raise RuntimeError("Author selector violated the fixed token budget")
    return selected


class NativeRegionalTokenSelection(NativeComplementaryRead):
    """Read-time selection; native bank, pointers and frame schedule retained."""

    def __init__(self, model, mode, *, inference_state=None, keep_fraction=.25):
        super().__init__(model, None, inference_state=inference_state, trace_only=True)
        if mode not in MODES or not 0 < keep_fraction <= 1:
            raise ValueError("Invalid selector settings")
        self.selector_mode, self.keep_fraction = mode, keep_fraction
        self.masks, self.anchors, self.patched = {}, {}, []
        self.author_attention = author_attention_forward()
        identity = lambda x: x
        self.adapter = SimpleNamespace(q_proj=identity, k_proj=identity, v_proj=identity,
            out_proj=identity, _separate_heads=lambda x, n: x, _recombine_heads=identity,
            num_heads=1, training=False, dropout_p=0.)

    def _prepare(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        if data["is_init_cond_frame"]:
            return super()._prepare(*args, **kwargs)
        self.selection_layers = []
        bank = data["output_dict"]
        # Only active native memories: copying/hash of the entire historical bank
        # would grow quadratically. These references include mask and pointer data.
        from modeling.native_memory_layout import selected_memory_frames
        spans = selected_memory_frames(self.model, data["frame_idx"], bank, data["num_frames"])
        active = [bank["cond_frame_outputs"].get(s["frame"],
                    bank["non_cond_frame_outputs"].get(s["frame"])) for s in spans]
        before = fingerprint(active)["sha256"]
        result = super()._prepare(*args, **kwargs)
        if fingerprint(active)["sha256"] != before:
            raise RuntimeError("Token selection mutated a stored native memory")
        if [r["layer"] for r in self.selection_layers] != [0, 1, 2, 3]:
            raise RuntimeError("Selection did not visit all four memory layers")
        self.calls[-1].update(selection_layers=self.selection_layers, bank_read_unchanged=True)
        return result

    def _before_selection(self, layer, module, args, kwargs):
        if self.context is None or args or module.training:
            raise RuntimeError("Unexpected memory cross-attention invocation")
        c = self.context
        spans, owner = c["spans"], c["object_id"]
        pointers = int(kwargs.get("num_k_exclude_rope", 0))
        spatial = sum(s["stop"] - s["start"] for s in spans)
        if kwargs["v"].shape != (1, spatial + pointers, 64):
            raise RuntimeError("Unsupported spatial/object-pointer memory layout")
        if spans[-1]["frame"] != c["frame"] - 1 or spans[0]["t_pos"] != 0:
            raise RuntimeError("Native prompt/latest layout changed")
        masks, counts, ids = [], [], []
        for i, span in enumerate(spans):
            if i in (0, len(spans) - 1) or self.selector_mode == "identity" or self.keep_fraction == 1:
                keep = torch.ones(4096, dtype=torch.bool, device=kwargs["v"].device)
            else:
                key = (owner, layer, span["frame"])
                if key not in self.masks:
                    raise RuntimeError("Older memory lacks a causal first-read selection")
                keep = self.masks[key]
            counts.append(int(keep.sum()))
            ids.append(_tensor_sha256(torch.where(keep)[0]))
            if 0 < i < len(spans) - 1:
                masks.append(keep)
        module._sparse_mask = torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)
        self.current_layer = layer
        self.current_pointers = pointers
        self.selection_layers.append({"layer": layer, "frame_ids": [s["frame"] for s in spans],
            "counts": counts, "selected_indices_sha256": ids,
            "spatial_keys_before": spatial, "spatial_keys_after": sum(counts),
            "pointer_keys": pointers, "mode": self.selector_mode})

    def _capture_sdpa(self, q, k, v, **kwargs):
        # This call alone provides the model output. Author recomputation only
        # supplies first-recollection importance for subsequent reads.
        output = F.scaled_dot_product_attention(q, k, v, **kwargs)
        c, layer = self.context, self.current_layer
        if q.shape[0] != 1 or q.shape[-2] != 4096:
            raise RuntimeError("Only one persistent instance with a 64x64 query grid is verified")
        spatial = k.shape[-2] - self.current_pointers
        row = self.selection_layers[-1]
        if spatial != row["spatial_keys_after"]:
            raise RuntimeError("Actual gathered keys differ from budget receipt")
        latest = c["spans"][-1]["frame"]
        cache_key = (c["object_id"], layer, latest)
        if cache_key in self.masks:
            raise RuntimeError("First-recollection selector cache reused for the wrong frame")
        mass = torch.zeros(4096, dtype=torch.float64, device=q.device)
        latest_k, latest_v = k[..., spatial-4096:spatial, :], v[..., spatial-4096:spatial, :]
        for start in range(0, 4096, 128):
            _, (_, _, weights) = self.author_attention(self.adapter, q[..., start:start+128, :], latest_k, latest_v)
            mass += weights.sum((0, 1, 2)).double()
        mass = (mass / (q.shape[0] * q.shape[1] * 4096)).float()
        if not torch.isfinite(mass).all() or abs(float(mass.sum()) - 1) > 1e-4:
            raise RuntimeError("Latest-frame first-read attention does not sum to one")
        features = c["outputs"][-1]["maskmem_features"][0].float().flatten(1).T
        probabilities = c["probabilities"][0, -1].flatten()
        anchor_key = (c["object_id"], layer)
        if anchor_key not in self.anchors:
            if len(c["spans"]) != 1:
                raise RuntimeError("Anchor initialization missed the native prompt read")
            self.anchors[anchor_key] = prompt_anchors(features, probabilities, mass)
        self.masks[cache_key] = select_tokens(features, probabilities, mass,
            self.anchors[anchor_key], self.selector_mode, self.keep_fraction)
        row.update(latest_usage_sum=float(mass.sum()),
            latest_selection_sha256=_tensor_sha256(self.masks[cache_key]),
            anchor_counts=[len(x) for x in self.anchors[anchor_key]])
        return output

    def __enter__(self):
        super().__enter__()
        try:
            for i, block in enumerate(self.model.memory_attention.layers):
                module = block.cross_attn_image
                if "forward" in module.__dict__:
                    raise RuntimeError("Cross-attention already patched")
                function = sparse_forward()
                namespace = dict(function.__globals__)
                namespace["F"] = SimpleNamespace(scaled_dot_product_attention=self._capture_sdpa)
                module.forward = MethodType(FunctionType(function.__code__, namespace,
                                                        function.__name__, function.__defaults__), module)
                self.patched.append(module)
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, a, k, i=i: self._before_selection(i, m, a, k), with_kwargs=True))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for module in self.patched:
            del module.forward
            for name in ("_sparse_mask", "_sparse_actual_keys"):
                if hasattr(module, name):
                    delattr(module, name)
        self.patched.clear()
        return super().__exit__(*exc)

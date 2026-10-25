"""Region-sparse history around pinned SAMURAI/SAM2Long author code.

Author eligibility loops, branch choice, pointer assembly and score modulation
are retained. Only the spatial-frame loop and gather/RoPE are adapted. This
combination policy is project glue, not an unchanged published model.
"""

import ast
import copy
import inspect
from types import MethodType

import torch

from modeling.region_definitions import mixed_cell_regions
from modeling.sparse_memory import author_sparse_parts, retained_cells
from scripts.bank_integrity import fingerprint
from scripts.run_author_baseline import REPOSITORIES
from modeling.region_tokens import _verified_source


def author_nodes(method):
    repo, revision = REPOSITORIES[method]
    relative = "sam2/sam2/modeling/sam2_base.py" if method == "samurai" else "sam2/modeling/sam2_base.py"
    path, source = _verified_source(repo.split("/")[-1], revision, relative)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "SAM2Base")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_memory_conditioned_features")
    branch = next(n for n in node.body if isinstance(n, ast.If) and ast.unparse(n.test) == "not is_init_cond_frame")
    if method == "samurai":
        spatial = next(n for n in branch.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.samurai_mode")
        selection = spatial.body
    else:
        spatial = branch
        selection = branch.body
    loop = next(n for n in selection if isinstance(n, ast.For) and ast.unparse(n.target) == "t_pos")
    if method == "samurai":
        eligibility = selection[:selection.index(loop)]
    else:
        eligibility = [next(n for n in selection if isinstance(n, ast.If) and "mem_pick_index == 0" in ast.unparse(n.test))]
    return path, node, spatial, loop, eligibility


def eligibility_function(method, namespace):
    path, _, _, _, nodes = author_nodes(method)
    nodes = copy.deepcopy(nodes)
    replaced = 0
    for n in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if isinstance(n, ast.Compare) and ast.unparse(n.left) == "len(valid_indices)":
            n.comparators = [ast.Name(id="limit", ctx=ast.Load())]
            replaced += 1
    if replaced != 1:
        raise RuntimeError("Author eligibility break structure changed")
    function = ast.parse("def eligible(self, frame_idx, output_dict, num_frames, mem_pick_index=0, start_frame_idx=0, iou_thre=0.3, limit=31):\n pass").body[0]
    function.body = nodes + ast.parse("return valid_indices").body
    env = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), env)
    return env[function.name]


def adapted_prepare(method, namespace):
    path, node, spatial, loop, _ = author_nodes(method)
    replacement = ast.parse(
        "for t_pos, region_idx in self._region_history_plan(locals()):\n"
        " out = output_dict['non_cond_frame_outputs'].get(region_idx, None)\n"
        " if out is None:\n"
        "  out = unselected_cond_outputs.get(region_idx, None)\n"
        " t_pos_and_prevs.append((t_pos, out))\n"
        + (" prev_idxs.append(region_idx)\n" if method == "sam2long" else "")
    ).body[0]
    spatial.body[spatial.body.index(loop)] = replacement
    env = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), env)
    return env[node.name]


def adapted_cross_attention(module):
    """Same gather adaptation as E2, but preserve this author's own SDPA code."""
    original = module.forward.__func__
    node = ast.parse(inspect.getsource(original).strip()).body[0]
    calls = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "apply_rotary_enc":
            n.func.id = "author_rotary"
            n.keywords += [ast.keyword(arg=k, value=ast.parse(v, mode="eval").body) for k, v in {
                "N_k": "original_spatial", "prune_mask": "self._sparse_mask",
                "MTP_id_st": "middle_start", "MTP_id_ed": "middle_end",
            }.items()]
            calls += 1
    if calls != 1:
        raise RuntimeError("Author RoPE call changed")
    node.body = ast.parse(
        "original_spatial = k.shape[1] - num_k_exclude_rope\n"
        "middle_start = 4096\n"
        "middle_end = max(middle_start, original_spatial - 4096)\n"
        "pointer_k, pointer_v = k[:, original_spatial:], v[:, original_spatial:]\n"
        "k, v = author_gather(k, v, self._sparse_mask, middle_start, middle_end)\n"
        "if num_k_exclude_rope and not (torch.equal(k[:, -num_k_exclude_rope:], pointer_k) and torch.equal(v[:, -num_k_exclude_rope:], pointer_v)):\n"
        " raise RuntimeError('Pointer tail changed')\n"
        "self._sparse_actual_keys = k.shape[1]\n"
    ).body + node.body
    gather, rotary = author_sparse_parts()
    env = {**original.__globals__, "author_gather": gather, "author_rotary": rotary}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), inspect.getfile(original), "exec"), env)
    return env[node.name]


def branch_memory(output, frame, selection):
    features, masks = output["maskmem_features"], output["pred_masks"]
    branch = None
    if features.ndim == 5:
        if not isinstance(selection, dict) or frame not in selection:
            raise RuntimeError("Missing author-selected memory branch")
        branch = selection[frame]
        features, masks = features[..., branch], masks[..., branch]
    if features.shape != (1, 64, 64, 64) or masks.shape != (1, 1, 256, 256):
        raise RuntimeError("Unsupported memory layout")
    return features, masks, branch


class AuthorSparseMemory:
    def __init__(self, model, method, prompt, identity=False):
        if method not in ("samurai", "sam2long") or model.num_maskmem != 7:
            raise ValueError("Expected pinned seven-slot author model")
        if method == "samurai" and not model.samurai_mode:
            raise ValueError("SAMURAI must remain enabled")
        self.model, self.method, self.prompt, self.identity = model, method, prompt, identity
        self.prepare = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.prepare)
        self.eligible = eligibility_function(method, self.prepare.__func__.__globals__)
        self.calls, self.handles, self.originals = [], [], []
        self.label_cache = {}

    def plan(self, context):
        current = context["frame_idx"]
        selection = context.get("mem_pick_index", 0)
        limit = (self.model.max_obj_ptrs_in_encoder if self.method == "samurai" else min(context["num_frames"], self.model.max_obj_ptrs_in_encoder))-1
        arguments = {k: context[k] for k in ("frame_idx", "output_dict", "num_frames", "mem_pick_index", "start_frame_idx", "iou_thre") if k in context}
        native = self.eligible(self.model, **arguments, limit=limit)
        if native != context["valid_indices"]:
            raise RuntimeError("Reused author eligibility differs from native selection")
        candidates = native[-6:] if self.identity else self.eligible(self.model, **arguments, limit=31)[-31:]
        outputs = context["output_dict"]
        # The first SAMURAI read forces index0 into valid_indices, but its
        # selected conditioning prompt is not a nonconditioning memory again.
        available = set(outputs["non_cond_frame_outputs"]) | set(context["unselected_cond_outputs"])
        candidates = [f for f in candidates if f in available]

        def unit(frame):
            stored = outputs["cond_frame_outputs"].get(frame, outputs["non_cond_frame_outputs"].get(frame))
            features, masks, branch = branch_memory(stored, frame, selection)
            key = (frame, branch)
            if key not in self.label_cache:
                self.label_cache[key] = retained_cells(mixed_cell_regions(masks[0, 0])["labels"])
            return {"frame": frame, "branch": branch, "features": features, "keep": self.label_cache[key]}

        if not self.identity and candidates:
            latest, older, used = candidates[-1], [], 0
            for frame in reversed(candidates[:-1]):
                count = int(unit(frame)["keep"].sum())
                if used+count > 20480:
                    break
                used += count
                older.append(frame)
            candidates = list(reversed(older))+[latest]
        self.units = [unit(self.prompt)]+[unit(f) for f in candidates]
        if any(u["frame"] >= current for u in self.units) or len(self.units) > 32:
            raise RuntimeError("Future memory or frame-cap violation")
        # Native slots follow eligible-frame rank, not raw chronological age.
        return [(max(1, 7-len(candidates)+i), frame) for i, frame in enumerate(candidates)]

    def prepare_read(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        context = bound.arguments
        if context["track_in_reverse"]:
            raise RuntimeError("Only forward propagation is verified")
        if context["is_init_cond_frame"]:
            return self.adapted(*args, **kwargs)
        before = fingerprint(context["output_dict"])["sha256"]
        self.layers = []
        result = self.adapted(*args, **kwargs)
        if fingerprint(context["output_dict"])["sha256"] != before:
            raise RuntimeError("Memory reading mutated stored author hypotheses")
        if [r["layer"] for r in self.layers] != list(range(4)):
            raise RuntimeError("Missing author branch attention layer")
        self.calls.append({"frame": context["frame_idx"], "selection": copy.deepcopy(context.get("mem_pick_index", 0)),
                           "bank_sha256": before, "layers": self.layers})
        return result

    def before(self, layer, module, args, kwargs):
        if args or module.training:
            raise RuntimeError("Expected eval keyword cross attention")
        values = kwargs["v"]
        pointers = int(kwargs.get("num_k_exclude_rope", 0))
        if values.shape[0] != 1 or values.shape[1]-pointers != len(self.units)*4096:
            raise RuntimeError("Assembled author frame layout differs")
        counts, masks = [], []
        for i, unit in enumerate(self.units):
            expected = unit["features"].flatten(2).transpose(1, 2).float()
            if not torch.equal(values[:, i*4096:(i+1)*4096].float().cpu(), expected.cpu()):
                raise RuntimeError("Actual author memory/branch mismatch")
            keep = torch.ones(4096, dtype=torch.bool) if self.identity or i in (0, len(self.units)-1) else unit["keep"]
            counts.append(int(keep.sum()))
            if 0 < i < len(self.units)-1:
                masks.append(keep)
        module._sparse_mask = (torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)).to(values.device)
        if sum(counts) > 28672:
            raise RuntimeError("Spatial budget exceeded")
        self.layers.append({"layer": layer, "frames": [u["frame"] for u in self.units],
                            "branches": [u["branch"] for u in self.units], "counts": counts,
                            "pointer_keys": pointers, "selected_spatial_keys": sum(counts),
                            "query_tokens": kwargs["q"].shape[1]})

    def after(self, layer, module, args, kwargs, output):
        row = self.layers[-1]
        if row["layer"] != layer or module._sparse_actual_keys != row["selected_spatial_keys"]+row["pointer_keys"]:
            raise RuntimeError("Actual gather length differs from receipt")

    def __enter__(self):
        self.adapted = MethodType(adapted_prepare(self.method, self.prepare.__func__.__globals__), self.model)
        self.model._region_history_plan = self.plan
        self.model._prepare_memory_conditioned_features = self.prepare_read
        for i, block in enumerate(self.model.memory_attention.layers):
            module = block.cross_attn_image
            if "forward" in module.__dict__:
                raise RuntimeError("Attention already patched")
            module.forward = MethodType(adapted_cross_attention(module), module)
            self.originals.append(module)
            self.handles.append(module.register_forward_pre_hook(lambda m, a, k, i=i: self.before(i, m, a, k), with_kwargs=True))
            self.handles.append(module.register_forward_hook(lambda m, a, k, o, i=i: self.after(i, m, a, k, o), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for module in self.originals:
            del module.forward
            for key in ("_sparse_mask", "_sparse_actual_keys"):
                if hasattr(module, key):
                    delattr(module, key)
        del self.model._prepare_memory_conditioned_features
        del self.model._region_history_plan

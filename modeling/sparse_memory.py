"""Efficient-SAM2 gather/RoPE integration around native SAM2 SDPA.

Author gather statements and apply_rotary_enc_prune execute unchanged. The
native forward is AST-adapted only to gather before projection and call that
RoPE helper. Temporal assembly/region policies are explicitly project glue.
No encoder, decoder, parameter or memory-writing path is replaced.
"""

import ast
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import math
from types import MethodType

import numpy as np
import torch

from modeling.region_definitions import mixed_cell_regions
from modeling.region_tokens import (
    EFFICIENT_SAM2_REVISION, SAM2_REVISION, _verified_source,
)
from scripts.region_sparse_statistics import author_dilate
from scripts.bank_integrity import fingerprint


@lru_cache(maxsize=1)
def native_positions():
    path, _ = _verified_source("sam2", SAM2_REVISION, "sam2/modeling/position_encoding.py")
    spec = spec_from_file_location("adeseg_native_sparse_positions", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def author_sparse_parts():
    path, text = _verified_source(
        "Efficient-SAM2", EFFICIENT_SAM2_REVISION, "build_model/window_bypass.py"
    )
    tree = ast.parse(text)
    namespace = {"torch": torch, "reshape_for_broadcast": native_positions().reshape_for_broadcast}
    rotary = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_rotary_enc_prune")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WB_RoPEAttention_ca")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    branch = next(n for n in forward.body if isinstance(n, ast.If))
    wanted = {"k_sel", "v_sel", "k", "v"}
    statements = [n for n in branch.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted
                  and (n.targets[0].id.endswith("_sel") or
                       (isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "torch.cat"))]
    if len(statements) != 4:
        raise RuntimeError("Pinned author sparse gather structure changed")
    # Execute exactly the author's four gather/concatenate assignments.
    gather = ast.parse("def gather(k,v,mem_mask,MTP_id_st,MTP_id_ed):\n return k,v").body[0]
    gather.body = statements + gather.body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rotary, gather], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["gather"], namespace[rotary.name]


@lru_cache(maxsize=1)
def sparse_forward():
    """Preserve native projection/SDPA/output source, adapt selection and RoPE."""
    path, text = _verified_source("sam2", SAM2_REVISION, "sam2/modeling/sam/transformer.py")
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "RoPEAttention")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "apply_rotary_enc":
            n.func.id = "author_rotary"
            n.keywords += [ast.keyword(arg=k, value=ast.parse(v, mode="eval").body) for k, v in {
                "N_k": "original_spatial", "prune_mask": "self._sparse_mask",
                "MTP_id_st": "middle_start", "MTP_id_ed": "middle_end",
            }.items()]
    prelude = ast.parse(
        "original_spatial = k.shape[1] - num_k_exclude_rope\n"
        "middle_start = 4096\n"
        "middle_end = max(middle_start, original_spatial - 4096)\n"
        "pointer_k = k[:, original_spatial:]\n"
        "pointer_v = v[:, original_spatial:]\n"
        "k, v = author_gather(k, v, self._sparse_mask, middle_start, middle_end)\n"
        "if num_k_exclude_rope and not (torch.equal(k[:, -num_k_exclude_rope:], pointer_k) and torch.equal(v[:, -num_k_exclude_rope:], pointer_v)):\n"
        " raise RuntimeError('Sparse gather changed object pointers')\n"
        "self._sparse_actual_keys = k.shape[1]\n"
    ).body
    node.body = prelude + node.body
    gather, rotary = author_sparse_parts()
    namespace = {"torch": torch, "Tensor": torch.Tensor, "math": math,
                 "F": torch.nn.functional, "author_gather": gather, "author_rotary": rotary}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["forward"]


def retained_cells(labels):
    occupied = torch.as_tensor(labels) != 0
    return author_dilate()(occupied, 3).flatten()


def temporal_slot(frame, current, prompt):
    if frame >= current:
        raise ValueError("Current/future memory is forbidden")
    return 6 if frame == prompt else min(current-frame-1, 5)


class SparseMemoryRead:
    """One-object forward-only native-memory gate; optional long-read assembly."""

    def __init__(self, model, prompt, mode, max_frames=32):
        if mode not in ("all_native", "all_rebuilt", "region_native", "region_long"):
            raise ValueError("Unknown E2 mode")
        if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
            raise ValueError("E2 requires the native seven-slot stride-one configuration")
        self.model, self.prompt, self.mode, self.max_frames = model, prompt, mode, max_frames
        self.frame, self.bank, self.handles, self.originals = prompt, {}, [], []
        self.layers, self.units, self.selected = [], [], None

    def _assemble(self, module, args, kwargs):
        self.read_bank_hash = fingerprint(self.bank)["sha256"]
        if args or self.mode not in ("all_rebuilt", "region_long"):
            if args:
                raise RuntimeError("Unexpected positional memory-attention arguments")
            return
        if self.frame <= self.prompt:
            raise RuntimeError("Unexpected conditioning memory-attention read")
        pointer_count = int(kwargs.get("num_obj_ptr_tokens", 0))
        memory, pos = kwargs["memory"], kwargs["memory_pos"]
        latest = self.frame-1
        older = list(range(max(self.prompt+1, self.frame-(self.max_frames-1)), latest))
        if self.mode == "all_rebuilt":
            older = list(range(max(self.prompt+1, self.frame-6), latest))
        else:
            used, kept = 0, []
            for frame in reversed(older):
                count = int(self.bank[frame]["keep"].sum())
                if used+count > 20480:
                    break
                used += count
                kept.append(frame)
            older = list(reversed(kept))
        frames = [self.prompt] + older + ([latest] if latest != self.prompt else [])
        values, positions = [], []
        for frame in frames:
            item = self.bank[frame]
            values.append(item["features"].to(memory.device).flatten(2).permute(2, 0, 1))
            spatial_pos = item["pos"].to(pos.device).flatten(2).permute(2, 0, 1)
            positions.append(spatial_pos+self.model.maskmem_tpos_enc[temporal_slot(frame, self.frame, self.prompt)])
        if pointer_count:
            values.append(memory[-pointer_count:])
            positions.append(pos[-pointer_count:])
        new_memory, new_pos = torch.cat(values), torch.cat(positions)
        if self.mode == "all_rebuilt" and not (
            torch.equal(new_memory, memory) and torch.equal(new_pos, pos)
        ):
            raise RuntimeError("Rebuilt seven-slot inputs differ from native memory/positions")
        self.assembled_frames = frames
        return args, {**kwargs, "memory": new_memory, "memory_pos": new_pos}

    def _after(self, module, args, kwargs, output):
        if fingerprint(self.bank)["sha256"] != self.read_bank_hash:
            raise RuntimeError("Attention mutated captured native memory buffers")
        for trace, block in zip(self.layers, self.model.memory_attention.layers):
            if block.cross_attn_image._sparse_actual_keys != trace["selected_spatial_keys"]+trace["pointer_keys"]:
                raise RuntimeError("Actual gathered key length differs from budget receipt")

    def _before(self, layer, module, args, kwargs):
        if args or module.training:
            raise RuntimeError("E2 supports eval-mode keyword cross attention only")
        memory = kwargs["v"]
        pointers = int(kwargs.get("num_k_exclude_rope", 0))
        spatial = memory.shape[1]-pointers
        if memory.shape[0] != 1 or spatial % 4096 or spatial < 4096:
            raise RuntimeError("Unexpected B+ spatial/pointer layout")
        frames = (self.assembled_frames if self.mode in ("all_rebuilt", "region_long") else
                  [self.prompt]+list(range(max(self.prompt+1, self.frame-6), self.frame)))
        if len(frames)*4096 != spatial:
            raise RuntimeError("Selected native frames do not match actual read length")
        masks = []
        for index, frame in enumerate(frames):
            expected = self.bank[frame]["features"].flatten(2).transpose(1, 2).float()
            if not torch.equal(memory[:, index*4096:(index+1)*4096].float().cpu(), expected.cpu()):
                raise RuntimeError("Actual memory values do not match captured frame")
            if 0 < index < len(frames)-1:
                masks.append(self.bank[frame]["keep"] if self.mode.startswith("region") else torch.ones(4096, dtype=torch.bool))
        mask = torch.cat(masks).to(memory.device) if masks else torch.empty(0, dtype=torch.bool, device=memory.device)
        module._sparse_mask = mask
        retained = spatial-len(mask)+int(mask.sum())
        if retained > 28672 or pointers < 0:
            raise RuntimeError("Spatial read budget exceeded")
        self.layers.append({"layer": layer, "frames": frames, "raw_spatial_keys": spatial,
                            "selected_spatial_keys": retained, "pointer_keys": pointers,
                            "temporal_slots": [temporal_slot(f, self.frame, self.prompt) for f in frames]})
        if layer and self.layers[-1]["frames"] != self.layers[0]["frames"]:
            raise RuntimeError("Layer-to-layer memory frame selection changed")

    def capture(self, state, index):
        if index != self.frame:
            raise RuntimeError("Nonchronological sparse propagation")
        if index > self.prompt and [r["layer"] for r in self.layers] != list(range(4)):
            raise RuntimeError("Not all four memory-attention layers were observed")
        outputs = state["output_dict_per_obj"][0]
        stored = outputs["cond_frame_outputs"].get(index, outputs["non_cond_frame_outputs"].get(index))
        if stored is None or stored["maskmem_features"] is None:
            raise RuntimeError("Missing actual native memory write")
        labels = mixed_cell_regions(stored["pred_masks"][0, 0])["labels"]
        self.bank[index] = {"features": stored["maskmem_features"].detach(),
                            "pos": stored["maskmem_pos_enc"][-1].detach(),
                            "keep": retained_cells(labels)}
        record = {"frame": index, "mode": self.mode, "layers": self.layers,
                  "region_counts": np.bincount(labels.flatten(), minlength=3).tolist()}
        self.layers = []
        self.frame += 1
        return record

    def __enter__(self):
        forward = sparse_forward()
        self.handles.append(self.model.memory_attention.register_forward_pre_hook(self._assemble, with_kwargs=True))
        self.handles.append(self.model.memory_attention.register_forward_hook(self._after, with_kwargs=True))
        for layer, block in enumerate(self.model.memory_attention.layers):
            module = block.cross_attn_image
            if hasattr(module, "_sparse_mask") or "forward" in module.__dict__:
                raise RuntimeError("Cross attention already modified")
            self.originals.append(module)
            module.forward = MethodType(forward, module)
            self.handles.append(module.register_forward_pre_hook(
                lambda module, args, kwargs, layer=layer: self._before(layer, module, args, kwargs), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for module in self.originals:
            del module.forward
            if hasattr(module, "_sparse_mask"):
                del module._sparse_mask
            if hasattr(module, "_sparse_actual_keys"):
                del module._sparse_actual_keys

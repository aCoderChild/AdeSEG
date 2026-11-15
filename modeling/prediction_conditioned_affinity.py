"""Use SAM2's first-pass mask to bias a second, dense memory read."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from modeling.soft_role_affinity import mask_logits_to_soft_roles
from modeling.region_mass_attention import native_identity


class PredictionRoleScales(nn.Module):
    """Learn bounded per-layer penalties for mismatched soft region roles."""

    def __init__(self, layers=4, max_strength=2.0):
        super().__init__()
        if layers != 4 or max_strength <= 0:
            raise ValueError("Expected four SAM2 memory-attention layers")
        self.raw_strength = nn.Parameter(torch.zeros(layers, 3))
        self.max_strength = float(max_strength)

    def strengths(self):
        return self.max_strength * self.raw_strength.tanh()


def _stored_role_map(pred_masks, device):
    masks = torch.as_tensor(pred_masks)
    if masks.ndim != 4 or masks.shape[1:] != (1, 256, 256):
        raise ValueError(f"Expected one stored 256x256 mask per batch item, got {tuple(masks.shape)}")
    if masks.shape[0] == 0:
        raise ValueError("Stored mask batch must not be empty")
    return torch.stack([
        mask_logits_to_soft_roles(mask[0]).reshape(4096, 3) for mask in masks
    ]).to(device)


def memory_role_sequence(model, frame_idx, output_dict, num_frames,
                         track_in_reverse=False, mode="regional"):
    """Mirror SAM2's memory-frame selection and return roles in key order."""
    from sam2.modeling.sam2_utils import select_closest_cond_frames

    cond_outputs = output_dict["cond_frame_outputs"]
    selected, unselected = select_closest_cond_frames(
        frame_idx, cond_outputs, model.max_cond_frames_in_attn
    )
    t_pos_and_prevs = [(0, int(index), out) for index, out in selected.items()]
    stride = 1 if model.training else model.memory_temporal_stride_for_eval

    for t_pos in range(1, model.num_maskmem):
        t_rel = model.num_maskmem - t_pos
        if t_rel == 1:
            prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
        elif track_in_reverse:
            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
            prev_frame_idx += (t_rel - 2) * stride
        else:
            prev_frame_idx = ((frame_idx - 2) // stride) * stride
            prev_frame_idx -= (t_rel - 2) * stride
        out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
        if out is None:
            out = unselected.get(prev_frame_idx)
        t_pos_and_prevs.append((t_pos, int(prev_frame_idx), out))

    role_maps, frame_spans = [], []
    cursor = 0
    for t_pos, frame_index, output in t_pos_and_prevs:
        if output is None:
            continue
        role_map = _stored_role_map(output["pred_masks"], device=output["maskmem_features"].device)
        if role_map.shape[0] != output["maskmem_features"].shape[0]:
            raise RuntimeError("Stored mask and memory object batches differ")
        tokens = output["maskmem_features"].shape[-2] * output["maskmem_features"].shape[-1]
        if role_map.shape[1] != tokens:
            raise RuntimeError(f"Mask roles and stored memory grid differ at t_pos={t_pos}")
        if mode == "global":
            role_map = role_map.mean(dim=1, keepdim=True).expand(-1, tokens, -1)
        elif mode != "regional":
            raise ValueError(f"Unknown memory-role mode: {mode}")
        role_maps.append(role_map)
        frame_spans.append({
            "frame": frame_index,
            "t_pos": int(t_pos),
            "start": cursor,
            "stop": cursor + tokens,
        })
        cursor += tokens

    if not role_maps:
        raise RuntimeError("SAM2 selected no spatial memory frames for the second read")
    return torch.cat(role_maps, dim=1), frame_spans


class PredictionConditionedRead:
    def __init__(self, model, scales, query_roles, memory_roles, mode, chunk_size=512):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global memory roles")
        if query_roles.ndim != 3 or query_roles.shape != (1, 4096, 3):
            raise ValueError("Expected current query roles [1, 4096, 3]")
        if memory_roles.ndim != 3 or memory_roles.shape[0] != 1 or memory_roles.shape[-1] != 3:
            raise ValueError("Expected memory roles [1, spatial keys, 3]")
        self.model = model
        self.scales = scales
        self.query_roles = query_roles
        self.memory_roles = memory_roles
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.calls = []
        self.active = None
        self.inside = False

    def _before(self, index):
        def hook(module, args, kwargs):
            if self.active is not None:
                raise RuntimeError("Nested SAM2 memory-attention call")
            self.active = (index, int(kwargs.get("num_k_exclude_rope", 0)))
        return hook

    def _after(self, module, args, kwargs, output):
        self.active = None

    def _biased(self, query, key, value, layer, pointer_tokens, args, kwargs):
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.0) != 0.0:
            raise RuntimeError("Unsupported SAM2 memory SDPA options")
        batch, heads, query_count, _ = query.shape
        spatial = key.shape[-2] - pointer_tokens
        if batch != 1 or spatial != self.memory_roles.shape[1]:
            raise RuntimeError(
                f"Memory role/key mismatch: batch={batch}, roles={self.memory_roles.shape[1]}, "
                f"spatial_keys={spatial}, pointers={pointer_tokens}"
            )
        if query_count != self.query_roles.shape[1]:
            raise RuntimeError("Query grid does not match the first-pass mask grid")

        strengths = self.scales.strengths()[layer].to(device=query.device, dtype=torch.float32)
        q_roles = self.query_roles.to(device=query.device, dtype=torch.float32)
        m_roles = self.memory_roles.to(device=query.device, dtype=torch.float32)
        chunks = []
        for start in range(0, query_count, self.chunk_size):
            stop = min(start + self.chunk_size, query_count)
            weighted_queries = q_roles[:, start:stop] * strengths[None, None]
            bias = -torch.einsum("bqr,bkr->bqk", weighted_queries, 1.0 - m_roles)
            if pointer_tokens:
                bias = torch.cat(
                    (bias, bias.new_zeros((batch, stop - start, pointer_tokens))), dim=-1
                )
            chunks.append(self.sdpa(
                query[:, :, start:stop], key, value,
                attn_mask=bias[:, None].to(query.dtype), dropout_p=0.0,
            ))
        return torch.cat(chunks, dim=2)

    def _attention(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.sdpa(query, key, value, *args, **kwargs)
        layer, pointer_tokens = self.active
        self.inside = True
        try:
            zero = not bool(torch.count_nonzero(self.scales.raw_strength[layer].detach()).item())
            if zero:
                biased = self._biased(query, key, value, layer, pointer_tokens, args, kwargs)
                output = native_identity(query, key, value, self.sdpa, biased)
            else:
                output = self._biased(query, key, value, layer, pointer_tokens, args, kwargs)
        finally:
            self.inside = False
        spatial = key.shape[-2] - pointer_tokens
        self.calls.append({
            "layer": layer,
            "query_tokens": query.shape[-2],
            "spatial_keys": spatial,
            "pointer_keys": pointer_tokens,
            "pointer_bias_max_abs": 0.0,
            "mode": self.mode,
            "mean_abs_role_strength": float(self.scales.strengths()[layer].detach().abs().mean()),
        })
        return output

    def __enter__(self):
        if len(self.model.memory_attention.layers) != 4:
            raise RuntimeError("Expected four SAM2 memory-attention layers")
        self.handles = []
        for index, layer in enumerate(self.model.memory_attention.layers):
            self.handles.append(layer.cross_attn_image.register_forward_pre_hook(
                self._before(index), with_kwargs=True
            ))
            self.handles.append(layer.cross_attn_image.register_forward_hook(
                self._after, with_kwargs=True, always_call=True
            ))
        self.sdpa = F.scaled_dot_product_attention
        self.override = patch.object(F, "scaled_dot_product_attention", self._attention)
        self.override.start()
        return self

    def __exit__(self, *exc):
        self.override.stop()
        for handle in self.handles:
            handle.remove()
        self.active = None


class PredictionConditionedTrackStep:
    """Run native SAM2 once, then reuse its decoder and memory writer after a biased read."""

    def __init__(self, model, scales, mode="regional", chunk_size=512):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global mode")
        self.model = model
        self.scales = scales
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.reads = []
        self.memory_write_frames = []

    def __enter__(self):
        model = self.model
        self.original = model.track_step

        def track_step(
            frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
            output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True,
            prev_sam_mask_logits=None,
        ):
            if (is_init_cond_frame or model.num_maskmem == 0
                    or (mask_inputs is not None and model.use_mask_input_as_output_without_sam)):
                return self.original(
                    frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                    output_dict, num_frames, track_in_reverse, run_mem_encoder,
                    prev_sam_mask_logits,
                )

            current_out, first_outputs, high_res_features, _ = model._track_step(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                output_dict, num_frames, track_in_reverse, prev_sam_mask_logits,
            )
            query_roles = mask_logits_to_soft_roles(first_outputs[3][0, 0]).reshape(1, 4096, 3)
            memory_roles, spans = memory_role_sequence(
                model, frame_idx, output_dict, num_frames, track_in_reverse, self.mode
            )
            read = PredictionConditionedRead(
                model, self.scales, query_roles, memory_roles, self.mode, self.chunk_size
            )
            with read:
                pix_feat = model._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=current_vision_feats[-1:],
                    current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                    feat_sizes=feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse,
                )
            if [call["layer"] for call in read.calls] != [0, 1, 2, 3]:
                raise RuntimeError("Second memory read did not cover all four layers")
            if any(call["spatial_keys"] != memory_roles.shape[1] for call in read.calls):
                raise RuntimeError("Spatial key count changed during second read")
            self.reads.append({
                "frame": int(frame_idx),
                "memory_frames": spans,
                "query_role_mean": query_roles.mean(dim=1)[0].detach().cpu().tolist(),
                "memory_role_mean": memory_roles.mean(dim=1)[0].detach().cpu().tolist(),
                "layers": read.calls,
            })

            decode_mask_inputs = prev_sam_mask_logits if prev_sam_mask_logits is not None else mask_inputs
            sam_outputs = model._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=decode_mask_inputs,
                high_res_features=high_res_features,
                multimask_output=model._use_multimask(is_init_cond_frame, point_inputs),
            )
            (
                _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits,
            ) = sam_outputs
            current_out["pred_masks"] = low_res_masks
            current_out["pred_masks_high_res"] = high_res_masks
            current_out["obj_ptr"] = obj_ptr
            if not model.training:
                current_out["object_score_logits"] = object_score_logits
            model._encode_memory_in_output(
                current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
                high_res_masks, object_score_logits, current_out,
            )
            if run_mem_encoder:
                self.memory_write_frames.append(int(frame_idx))
            return current_out

        model.track_step = track_step
        return self

    def __exit__(self, *exc):
        self.model.track_step = self.original

"""Autoregressive, truncated training utilities for the complementary reader."""

import ast
from functools import lru_cache

import torch
import torch.nn.functional as F

from modeling.region_tokens import SAM2_REVISION, _verified_source
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.regional_refinement import refinement_loss


def clone_tensors(value):
    """Make ordinary autograd-capable copies of nested inference-state tensors."""
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    return value


@lru_cache(maxsize=1)
def _author_focal_loss():
    path, source = _verified_source("sam2", SAM2_REVISION, "training/loss_fns.py")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "sigmoid_focal_loss"
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sigmoid_focal_loss"]


def object_presence_loss(object_score, target):
    """Reuse the pinned SAM2 sigmoid focal loss for object presence."""
    presence = target.flatten(1).gt(0).any(dim=1).to(object_score.dtype).unsqueeze(1)
    if object_score.shape != presence.shape:
        raise ValueError("Expected one object score per target mask")
    return _author_focal_loss()(object_score, presence, len(presence), alpha=-1, gamma=0)


def detach_tensors(value):
    """Detach a nested memory bank at a truncated-BPTT boundary."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_tensors(item) for item in value)
    return value


def capture_current_features(model, inference_state, frame):
    """Copy frozen SAM2 image features into ordinary tensors for autograd use."""
    with torch.inference_mode():
        _, _, vision, positions, sizes = model._get_image_feature(
            inference_state, frame, 1
        )
    with torch.inference_mode(False):
        return clone_tensors({
            "vision": vision,
            "positions": positions,
            "sizes": sizes,
        })


def reader_clip_loss(model, samples, initial_bank, auxiliary=None):
    """Run a consecutive clip using its own predicted masks and memory writes.

    ``samples`` contain only image features, frame IDs and GT loss targets.
    GT is never passed to ``track_step`` or used to populate the returned bank.
    The caller owns ``NativeComplementaryRead`` and truncates gradients by
    detaching the returned bank after each optimizer update.
    """
    if not samples or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected one or more consecutive post-prompt frames")

    bank = clone_tensors(initial_bank)
    losses, writes = [], []
    for sample in samples:
        if sample.get("mask_inputs") is not None or sample.get("point_inputs") is not None:
            raise ValueError("Post-prompt training frames cannot contain GT prompts")
        with decoder_training_mask(model.sam_mask_decoder) as captured:
            output = model.track_step(
                frame_idx=sample["frame"],
                is_init_cond_frame=False,
                current_vision_feats=sample["vision"],
                current_vision_pos_embeds=sample["positions"],
                feat_sizes=sample["sizes"],
                point_inputs=None,
                mask_inputs=None,
                output_dict=bank,
                num_frames=sample["num_frames"],
                track_in_reverse=False,
                run_mem_encoder=True,
            )

        logits = select_decoder_mask(captured)
        target = F.interpolate(
            sample["target"], size=logits.shape[-2:], mode="nearest"
        )
        loss = present_mask_loss(logits, target, refinement_loss)
        loss = loss + object_presence_loss(output["object_score_logits"], target)
        # Auxiliary supervision is evaluated only AFTER the native forward and
        # never supplies masks or labels to track_step or its memory writer.
        if auxiliary is not None:
            loss = loss + auxiliary(sample)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at frame {sample['frame']}")
        losses.append(loss)

        required = (
            "maskmem_pos_enc",
            "pred_masks",
            "obj_ptr",
            "object_score_logits",
            "maskmem_features",
        )
        if any(output.get(key) is None for key in required):
            raise RuntimeError("Native memory writer omitted a required output")
        stored = {key: output[key] for key in required}
        # Match SAM2VideoPredictor's persistent bank storage dtype.
        stored["maskmem_features"] = stored["maskmem_features"].to(torch.bfloat16)
        bank["non_cond_frame_outputs"][sample["frame"]] = stored
        writes.append({
            "frame": sample["frame"],
            "source": "this_arm_predicted_output",
            "target_used_for_memory": False,
            "memory_dtype": str(stored["maskmem_features"].dtype),
        })

    return torch.stack(losses).mean(), bank, writes

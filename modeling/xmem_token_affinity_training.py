"""Autoregressive SAM2 clip loss for the XMem-style memory attention adapter.

This follows the project's tested SAM2 training path: supervision is captured
from the decoder, while each later read receives memory written from the
adapter's own preceding prediction. The ground-truth target is loss-only.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.regional_refinement import refinement_loss


@contextmanager
def xmem_native_read_audit(model, mode="generic"):
    """Trace SAM2's actual key/value/position assembly without replacing it.

    The existing trace-only observer requires a frozen model when it registers
    hooks. Temporarily freeze only the newly added XMem parameters for hook
    registration, restore their flags before yielding, and keep the model in
    eval mode throughout. The observer returns SAM2's exact memory-attention
    output while checking stored-mask alignment, selected-frame order, native
    spatial/temporal positions, and pointer counts.
    """
    adapter_parameters = [
        parameter for name, parameter in model.named_parameters()
        if ".xmem_" in name
    ]
    if not adapter_parameters:
        raise RuntimeError("No XMem parameters found for the native read audit")
    if model.training:
        raise RuntimeError("The complete SAM2 model must remain in eval mode")
    previous = [parameter.requires_grad for parameter in adapter_parameters]
    observer = NativeComplementaryRead(
        model, reader=None, mode=mode, trace_only=True
    )
    entered = False
    try:
        for parameter in adapter_parameters:
            parameter.requires_grad_(False)
        observer.__enter__()
        entered = True
    finally:
        for parameter, requires_grad in zip(adapter_parameters, previous, strict=True):
            parameter.requires_grad_(requires_grad)
    try:
        yield observer
    finally:
        if entered:
            observer.__exit__(None, None, None)


def xmem_clip_loss(model, samples, initial_bank, *, step_observer=None):
    """Run one ordered post-prompt clip and return loss, updated bank, receipts.

    Samples must contain contiguous frame indices and image features captured
    through SAM2's inference cache. Their targets never enter ``track_step`` or
    the memory writer. The returned bank contains only this model's predictions.
    """
    if not samples or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive post-prompt frames")

    bank = clone_tensors(initial_bank)
    losses, writes = [], []
    for sample in samples:
        if sample.get("mask_inputs") is not None or sample.get("point_inputs") is not None:
            raise ValueError("Post-prompt samples cannot contain ground-truth prompts")
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
        if step_observer is not None:
            step_observer(sample["frame"], output)

        logits = select_decoder_mask(captured)
        target = F.interpolate(sample["target"], size=logits.shape[-2:], mode="nearest")
        loss = present_mask_loss(logits, target, refinement_loss)
        loss = loss + object_presence_loss(output["object_score_logits"], target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite XMem training loss at frame {sample['frame']}")
        losses.append(loss)

        required = (
            "maskmem_pos_enc",
            "pred_masks",
            "obj_ptr",
            "object_score_logits",
            "maskmem_features",
        )
        if any(output.get(key) is None for key in required):
            raise RuntimeError("SAM2 memory writer omitted a required output")
        stored = {key: output[key] for key in required}
        stored["maskmem_features"] = stored["maskmem_features"].to(torch.bfloat16)
        bank["non_cond_frame_outputs"][sample["frame"]] = stored
        writes.append({
            "frame": sample["frame"],
            "source": "this_arm_predicted_output",
            "target_used_for_memory": False,
            "memory_dtype": str(stored["maskmem_features"].dtype),
        })

    return torch.stack(losses).mean(), bank, writes

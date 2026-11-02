"""Train SAM2 memory projections with native and CFBI-region objectives."""

import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.regional_refinement import refinement_loss


def enable_memory_qk_training(model):
    """Unfreeze only q/k projections in the four image-memory attention layers."""
    model.requires_grad_(False)
    parameters = []
    names = []
    for layer_index, layer in enumerate(model.memory_attention.layers):
        attention = layer.cross_attn_image
        for projection_name in ("q_proj", "k_proj"):
            projection = getattr(attention, projection_name)
            for parameter_name, parameter in projection.named_parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
                names.append(
                    f"memory_attention.layers.{layer_index}.cross_attn_image."
                    f"{projection_name}.{parameter_name}"
                )
    if len(model.memory_attention.layers) != 4 or not parameters:
        raise RuntimeError("Expected q/k projections in four memory-attention layers")
    return names, parameters


def native_qk_clip_loss(model, samples, *, check_native=False):
    """Train q/k through the native autoregressive SAM2 prediction path."""
    if len(samples) < 2 or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive frames")

    bank = clone_tensors(samples[0]["output_dict"])
    losses = []
    for index, sample in enumerate(samples):
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
                run_mem_encoder=index + 1 < len(samples),
            )
        logits = select_decoder_mask(captured)
        target = F.interpolate(
            sample["target"], size=logits.shape[-2:], mode="nearest"
        )
        losses.append(
            present_mask_loss(logits, target, refinement_loss)
            + object_presence_loss(output["object_score_logits"], target)
        )
        if check_native:
            if not torch.equal(output["pred_masks"].detach(), sample["native_logits"]):
                raise RuntimeError(f"Native q/k control differs at {sample['frame']}")
            if not torch.equal(
                output["object_score_logits"].detach(), sample["native_object_score"]
            ):
                raise RuntimeError(f"Native object score differs at {sample['frame']}")
        if index + 1 < len(samples):
            stored = {
                key: output[key]
                for key in (
                    "maskmem_pos_enc",
                    "pred_masks",
                    "obj_ptr",
                    "object_score_logits",
                )
            }
            stored["maskmem_features"] = output["maskmem_features"].to(torch.bfloat16)
            if check_native:
                reference = samples[index + 1]["output_dict"][
                    "non_cond_frame_outputs"
                ][sample["frame"]]
                for key in (
                    "maskmem_features",
                    "pred_masks",
                    "obj_ptr",
                    "object_score_logits",
                ):
                    if not torch.equal(stored[key].detach(), reference[key]):
                        raise RuntimeError(f"Native stored {key} differs at {sample['frame']}")
            bank["non_cond_frame_outputs"][sample["frame"]] = stored
    return torch.stack(losses).mean()

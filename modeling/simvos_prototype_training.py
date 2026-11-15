"""Train SimVOS-informed memory prototypes with SAM2 propagation losses."""

import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.native_simvos_prototype_read import NativeSimVOSPrototypeRead
from modeling.regional_refinement import refinement_loss


def prototype_clip_loss(model, heads, samples, sequence, mode, *, check_native=False):
    if mode not in ("regional", "global"):
        raise ValueError("Expected regional or global prototype control")
    if len(samples) < 2 or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive predicted-memory frames")

    bank = clone_tensors(samples[0]["output_dict"])
    losses, deltas = [], []
    observer = NativeSimVOSPrototypeRead(model, heads, mode, sequence)
    with observer:
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
            target = F.interpolate(sample["target"], size=logits.shape[-2:], mode="nearest")
            losses.append(
                present_mask_loss(logits, target, refinement_loss)
                + object_presence_loss(output["object_score_logits"], target)
            )
            if check_native:
                if not torch.equal(output["pred_masks"].detach(), sample["native_logits"]):
                    raise RuntimeError(f"Zero-gain mask mismatch at frame {sample['frame']}")
                if not torch.equal(
                    output["object_score_logits"].detach(), sample["native_object_score"]
                ):
                    raise RuntimeError(f"Zero-gain presence mismatch at frame {sample['frame']}")
            deltas.append(float(
                (output["pred_masks"].detach().float() - sample["native_logits"].float())
                .abs().max().cpu()
            ))
            if index + 1 < len(samples):
                stored = {
                    key: output[key]
                    for key in (
                        "maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits"
                    )
                }
                stored["maskmem_features"] = output["maskmem_features"].to(torch.bfloat16)
                if check_native:
                    reference = samples[index + 1]["output_dict"]["non_cond_frame_outputs"][sample["frame"]]
                    for key in ("maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"):
                        if not torch.equal(stored[key].detach(), reference[key]):
                            raise RuntimeError(f"Zero-gain stored {key} mismatch at {sample['frame']}")
                bank["non_cond_frame_outputs"][sample["frame"]] = stored

    expected = [layer for _ in samples for layer in range(4)]
    if [call["layer"] for call in observer.calls] != expected:
        raise RuntimeError("Expected one prototype read in each of four layers per frame")
    if any(call["spatial_keys"] <= 0 or call["pointer_keys"] < 0 for call in observer.calls):
        raise RuntimeError("Invalid native spatial/pointer accounting")
    if any(not torch.isfinite(torch.tensor(call["prototype_read_rms"])) for call in observer.calls):
        raise RuntimeError("Nonfinite prototype-read statistic")
    return torch.stack(losses).mean(), observer.calls, deltas

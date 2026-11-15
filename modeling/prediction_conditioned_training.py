"""Train prediction-conditioned regional attention on autoregressive SAM2 clips."""

import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import (
    decoder_training_mask, present_mask_loss, select_decoder_mask,
)
from modeling.prediction_conditioned_affinity import PredictionConditionedTrackStep
from modeling.regional_refinement import refinement_loss


def clip_loss(model, scales, samples, mode, *, check_native=False):
    if mode not in ("regional", "global"):
        raise ValueError("Expected regional or global mode")
    if len(samples) < 2 or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive training frames")

    bank = clone_tensors(samples[0]["output_dict"])
    losses, records = [], []
    with PredictionConditionedTrackStep(model, scales, mode) as tracker:
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
            if len(captured) != 2:
                raise RuntimeError("Expected first-pass and second-pass decoder outputs")
            training_logits = select_decoder_mask(captured[-1:])
            target = F.interpolate(
                sample["target"], size=training_logits.shape[-2:], mode="nearest"
            )
            seg_loss = present_mask_loss(training_logits, target, refinement_loss)
            presence_loss = object_presence_loss(output["object_score_logits"], target)
            loss = seg_loss + presence_loss
            losses.append(loss)

            if check_native:
                if not torch.equal(output["pred_masks"].detach(), sample["native_logits"]):
                    raise RuntimeError(f"Zero-strength two-pass mask differs at {sample['frame']}")
                if "native_object_score" in sample and not torch.equal(
                    output["object_score_logits"].detach(), sample["native_object_score"]
                ):
                    raise RuntimeError(f"Zero-strength object score differs at {sample['frame']}")
                if "native_obj_ptr" in sample and not torch.equal(
                    output["obj_ptr"].detach(), sample["native_obj_ptr"]
                ):
                    raise RuntimeError(f"Zero-strength object pointer differs at {sample['frame']}")
                if index + 1 < len(samples):
                    native = samples[index + 1]["output_dict"]["non_cond_frame_outputs"][sample["frame"]]
                    for key in (
                        "maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"
                    ):
                        actual_tensor = output[key].detach()
                        if key == "maskmem_features":
                            actual_tensor = actual_tensor.to(torch.bfloat16)
                        if not torch.equal(actual_tensor, native[key]):
                            actual = actual_tensor.float()
                            expected = native[key].detach().to(actual.device).float()
                            maximum = float((actual - expected).abs().max())
                            raise RuntimeError(
                                f"Zero-strength stored {key} differs at {sample['frame']} "
                                f"(max_abs={maximum})"
                            )

            reads = tracker.reads[-1]
            records.append({
                "frame": int(sample["frame"]),
                "segmentation_loss": float(seg_loss.detach()),
                "presence_loss": float(presence_loss.detach()),
                "mask_supervision_source": "second_pass_decoder_before_presence_suppression",
                "target_present": bool(target.any()),
                "predicted_present": bool((output["object_score_logits"] > 0).any()),
                "memory_frames": reads["memory_frames"],
                "layers": reads["layers"],
            })
            if index + 1 < len(samples):
                stored = {
                    key: output[key]
                    for key in (
                        "maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits"
                    )
                }
                stored["maskmem_features"] = output["maskmem_features"].to(torch.bfloat16)
                bank["non_cond_frame_outputs"][sample["frame"]] = stored

    expected_writes = [sample["frame"] for sample in samples[:-1]]
    if tracker.memory_write_frames != expected_writes:
        raise RuntimeError("The second-pass output was not written exactly once per eligible frame")
    expected_layers = [layer for _ in samples for layer in range(4)]
    observed = [call["layer"] for row in records for call in row["layers"]]
    if observed != expected_layers:
        raise RuntimeError("Expected four second-pass attention layers per frame")
    return torch.stack(losses).mean(), records

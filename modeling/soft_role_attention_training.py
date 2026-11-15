"""Train soft region roles on autoregressive SAM2 clips."""

import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.native_region_readout import NativeRegionalReadout
from modeling.regional_refinement import refinement_loss
from modeling.soft_role_affinity import mask_logits_to_soft_roles, target_to_soft_roles


def balanced_soft_role_loss(role_logits, role_target):
    if role_logits.ndim != 4 or role_logits.shape[-1] != 3:
        raise ValueError("Expected query-role logits [batch, heads, tokens, 3]")
    if role_target.ndim != 4 or role_target.shape[0] != role_logits.shape[0]:
        raise ValueError("Expected soft role targets [batch, 1, tokens, 3]")
    if role_target.shape[1] == 1 and role_logits.shape[1] != 1:
        role_target = role_target.expand(-1, role_logits.shape[1], -1, -1)
    if role_target.shape != role_logits.shape:
        raise ValueError("Query logits and soft targets must align")
    frequency = role_target.mean(dim=(0, 1, 2))
    weight = torch.where(
        frequency > 0, frequency.reciprocal(), torch.zeros_like(frequency)
    )
    normalizer = (frequency * weight).sum().clamp_min(1e-12)
    loss = -(
        role_target * weight[None, None, None, :] * role_logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean() / normalizer
    return loss, frequency, weight


def clip_loss(model, heads, samples, mode, sequence, *, check_native=False,
              role_loss_weight=0.1):
    if mode not in ("soft_role_attention", "soft_role_global"):
        raise ValueError("Unknown soft role attention mode")
    if len(samples) < 2 or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive sampled frames")

    bank = clone_tensors(samples[0]["output_dict"])
    observer = NativeRegionalReadout(
        model, heads, mode, sequence, region_encoder=mask_logits_to_soft_roles,
    )
    losses, records = [], []
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
            segmentation_loss = present_mask_loss(logits, target, refinement_loss)
            segmentation_loss = segmentation_loss + object_presence_loss(
                output["object_score_logits"], target
            )
            role_target = target_to_soft_roles(sample["target"]).to(logits.device)
            role_target = role_target.reshape(1, 1, 4096, 3)
            per_layer_role_loss = []
            role_frequency = None
            class_weight = None
            for layer_index, head in enumerate(heads.layers):
                role_logits = head.last_query_logits
                if role_logits is None or role_logits.shape[0] != 1 or role_logits.shape[2:] != (4096, 3):
                    raise RuntimeError(f"Missing 64x64 query roles at layer {layer_index}")
                layer_loss, role_frequency, class_weight = balanced_soft_role_loss(
                    role_logits, role_target
                )
                per_layer_role_loss.append(layer_loss)
            role_loss = torch.stack(per_layer_role_loss).mean()
            total_loss = segmentation_loss + role_loss_weight * role_loss
            if check_native and not torch.equal(
                output["pred_masks"].detach(), sample["native_logits"]
            ):
                raise RuntimeError(f"Zero-bias native replay changed at frame {sample['frame']}")

            losses.append(total_loss)
            records.append({
                "frame": sample["frame"],
                "segmentation_loss": float(segmentation_loss.detach()),
                "role_loss": float(role_loss.detach()),
                "target_role_frequency": role_frequency.detach().cpu().tolist(),
                "role_class_weight": class_weight.detach().cpu().tolist(),
                "mean_query_role": torch.stack([
                    head.last_query_roles.detach().mean(dim=(0, 1, 2))
                    for head in heads.layers
                ]).mean(dim=0).cpu().tolist(),
            })

            if index + 1 < len(samples):
                stored = {
                    key: output[key]
                    for key in (
                        "maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits"
                    )
                }
                stored["maskmem_features"] = output["maskmem_features"].to(torch.bfloat16)
                if check_native:
                    native = samples[index + 1]["output_dict"]["non_cond_frame_outputs"][sample["frame"]]
                    for key in (
                        "maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"
                    ):
                        if not torch.equal(stored[key].detach(), native[key]):
                            raise RuntimeError(f"Zero-bias stored {key} changed")
                bank["non_cond_frame_outputs"][sample["frame"]] = stored

    if [row["layer"] for row in observer.calls] != [0, 1, 2, 3] * len(samples):
        raise RuntimeError("Expected four memory-attention layers per sampled frame")
    return torch.stack(losses).mean(), records, observer.calls

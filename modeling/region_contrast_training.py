"""Autoregressive SAM2 training with RegionContrast on predicted-role centers."""

import torch
import torch.nn.functional as F
from torch import nn

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import (
    decoder_training_mask,
    present_mask_loss,
    select_decoder_mask,
)
from modeling.cfbi_region_gate import NativeCFBIGate
from modeling.region_contrast import RegionContrastBank
from modeling.region_definitions import mixed_cell_regions
from modeling.regional_refinement import refinement_loss


class _CaptureRegionLabels(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None

    def region_context(self, vectors, labels, mode):
        self.labels = labels.detach().clone()
        return vectors.new_zeros((1, 192))

    def forward(self, attention, context):
        return attention


def region_contrast_clip_loss(
    model, banks, samples, sequence, *, contrast_weight=0.1, check_native=False
):
    if len(samples) < 2 or any(
        right["frame"] != left["frame"] + 1
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("Expected consecutive predicted-memory frames")
    if len(banks) != 4:
        raise ValueError("Expected one RegionContrast bank per memory layer")

    bank = clone_tensors(samples[0]["output_dict"])
    capture = _CaptureRegionLabels()
    observer = NativeCFBIGate(model, capture, "regional", sequence)
    activations = {}
    handles = []
    losses = []
    contrast_records = []
    for layer_index, layer in enumerate(model.memory_attention.layers):
        attention = layer.cross_attn_image

        def save(name):
            def hook(module, args, output):
                activations[name] = output
            return hook

        handles.extend([
            attention.q_proj.register_forward_hook(save((layer_index, "q"))),
            attention.k_proj.register_forward_hook(save((layer_index, "k"))),
        ])

    try:
        with observer:
            for index, sample in enumerate(samples):
                activations.clear()
                capture.labels = None
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
                segmentation_loss = present_mask_loss(logits, target, refinement_loss)
                segmentation_loss = segmentation_loss + object_presence_loss(
                    output["object_score_logits"], target
                )
                if check_native:
                    if not torch.equal(output["pred_masks"].detach(), sample["native_logits"]):
                        raise RuntimeError("Initial RegionContrast SAM2 replay changed masks")
                    if not torch.equal(
                        output["object_score_logits"].detach(), sample["native_object_score"]
                    ):
                        raise RuntimeError("Initial RegionContrast replay changed object scores")

                if set(activations) != {
                    (layer_index, kind)
                    for layer_index in range(4)
                    for kind in ("q", "k")
                } or capture.labels is None:
                    raise RuntimeError("Did not capture four-layer q/k and native region labels")
                query_regions = mixed_cell_regions(output["pred_masks"][0, 0])[
                    "labels"
                ]
                query_regions = torch.as_tensor(
                    query_regions, device=output["pred_masks"].device
                )
                pointer_count = observer.trace[-1]["pointer_keys"]
                contrast_losses = []
                per_layer = []
                for layer_index, region_bank in enumerate(banks):
                    q = activations[(layer_index, "q")]
                    k = activations[(layer_index, "k")]
                    layer_loss, layer_info = region_bank(
                        q,
                        k.transpose(0, 1),
                        capture.labels,
                        query_regions,
                        pointer_count=pointer_count,
                    )
                    contrast_losses.append(layer_loss)
                    per_layer.append(layer_info)
                contrast_loss = torch.stack(contrast_losses).mean()
                losses.append(segmentation_loss + contrast_weight * contrast_loss)
                contrast_records.append({
                    "frame": sample["frame"],
                    "contrast_loss": float(contrast_loss.detach()),
                    "layers": per_layer,
                })

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
                    stored["maskmem_features"] = output["maskmem_features"].to(
                        torch.bfloat16
                    )
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
                                raise RuntimeError(
                                    f"Initial stored {key} differs from native"
                                )
                    bank["non_cond_frame_outputs"][sample["frame"]] = stored
    finally:
        for handle in handles:
            handle.remove()

    if [record["frame"] for record in observer.trace] != [s["frame"] for s in samples]:
        raise RuntimeError("Memory-label observer missed or reordered a clip read")
    if any(record["layers"] != [0, 1, 2, 3] for record in observer.trace):
        raise RuntimeError("A training read did not traverse all four attention layers")
    if len(contrast_records) != len(samples):
        raise RuntimeError("RegionContrast did not score all four layers per frame")
    return torch.stack(losses).mean(), contrast_records, observer.trace

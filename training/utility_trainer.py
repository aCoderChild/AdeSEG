"""Frozen-SAM2 training path for the matched-budget attention baseline."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from external.MedSAM2.training.loss_fns import dice_loss, sigmoid_focal_loss
from external.MedSAM2.training.model.sam2 import SAM2Train
from external.MedSAM2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from adenoid.io import load_label_mask
from modeling.medsam2 import build_video_predictor
from modeling.utility_memory import UtilityController, UtilityHead, attach_utility_controller
from scripts.infer import predict_label, stage_frames


PROMPT_OVERRIDES = {
    "prob_to_use_pt_input_for_train": 0.0,
    "prob_to_use_pt_input_for_eval": 0.0,
    "num_init_cond_frames_for_train": 1,
    "num_init_cond_frames_for_eval": 1,
    "rand_init_cond_frames_for_train": False,
    "rand_init_cond_frames_for_eval": False,
}


class UtilitySAM2Train(SAM2Train):
    def __init__(self, fill_hole_area=0, **kwargs):
        super().__init__(**kwargs)


def build_attention_baseline(model_cfg, checkpoint, device):
    model = build_video_predictor(
        model_cfg, checkpoint, device,
        predictor_target="training.utility_trainer.UtilitySAM2Train",
        predictor_overrides=PROMPT_OVERRIDES,
    )
    controller = UtilityController(
        UtilityHead(query_dim=model.hidden_dim, memory_dim=model.mem_dim)
    )
    attach_utility_controller(model, controller)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in controller.parameters():
        parameter.requires_grad = True
    model.eval()  # fixed inference distribution; the controller has no dropout
    controller.train()
    return model, controller


def run_attention_clip(model, batch):
    with torch.no_grad():
        backbone_out = model.forward_image(batch.flat_img_batch)
    backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
    return model.forward_tracking(backbone_out, batch)


def segmentation_loss(outputs, masks):
    """SAM2's focal+Dice loss on every propagated frame, including empty masks."""
    terms = []
    for output, target in zip(outputs[1:], masks[1:].float()):
        logits = output["pred_masks_high_res"][0, 0].float()[None]
        labels = target.flatten()[None]
        inputs = logits.flatten(1)
        terms.append(20.0 * sigmoid_focal_loss(inputs, labels, 1) + dice_loss(inputs, labels, 1))
    return torch.stack(terms).mean()


def video_batch(images, masks) -> BatchedVideoDatapoint:
    """One video with one object: images [T, 3, H, W], masks [T, H, W]."""
    frames = torch.arange(images.size(0), dtype=torch.int, device=images.device)
    return BatchedVideoDatapoint(
        img_batch=images.unsqueeze(1),
        obj_to_frame_idx=torch.stack([frames, torch.zeros_like(frames)], dim=-1).unsqueeze(1),
        masks=masks.bool().unsqueeze(1),
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=torch.stack(
                [torch.zeros_like(frames), torch.ones_like(frames), frames], dim=-1
            ).unsqueeze(1).long(),
            frame_orig_size=torch.tensor(images.shape[-2:], device=images.device).repeat(images.size(0), 1, 1),
            batch_size=[images.size(0)],
        ),
        dict_key="pseudo_video",
        batch_size=[images.size(0)],
    )


def sequence_scores(predictor, videos, label_ids):
    """Average whole-sequence validation Dice and absent-frame false-positive rate."""
    present_dice, absent_fp = [], []
    for frames in videos.values():
        with tempfile.TemporaryDirectory(prefix="utility_val_") as temporary:
            staged = Path(temporary)
            stage_frames(frames, staged)
            for label_id in label_ids:
                logits, prompt, _, _ = predict_label(
                    predictor, frames, label_id, label_ids, staged, False, "mask"
                )
                if prompt is None:
                    continue
                dice, empty_fp = [], []
                for index in range(prompt + 1, len(frames)):
                    predicted = logits[index] > 0
                    target = load_label_mask(frames[index].mask_path, label_ids) == label_id
                    if target.any():
                        dice.append(2 * (predicted & target).sum() / (predicted.sum() + target.sum()))
                    else:
                        empty_fp.append(predicted.any())
                if dice:
                    present_dice.append(np.mean(dice))
                if empty_fp:
                    absent_fp.append(np.mean(empty_fp))
    return {
        "present_dice": float(np.mean(present_dice)),
        "absent_fp_rate": float(np.mean(absent_fp)) if absent_fp else None,
    }

"""Kalman-memory training on MedSAM2's own training model, loss, and batch format."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from MedSAM2.training.loss_fns import CORE_LOSS_KEY, MultiStepMultiMasksAndIous
from MedSAM2.training.model.sam2 import SAM2Train
from MedSAM2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from modeling.kalman_memory import KalmanMemoryMixin, KalmanMemoryUpdate
from modeling.medsam2 import build_video_predictor

# Match KalmanMemoryVideoPredictor: one box prompt on frame 0, no correction clicks, eval mode.
PROMPT_OVERRIDES = {
    "prob_to_use_pt_input_for_eval": 1.0,
    "prob_to_use_box_input_for_eval": 1.0,
    "num_frames_to_correct_for_eval": 1,
    "num_init_cond_frames_for_eval": 1,
}


class KalmanSAM2Train(KalmanMemoryMixin, SAM2Train):
    """Kalman-memory training with only the frame-0 MedSAM2 prompt.
    """

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        backbone_out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        backbone_out["frames_to_add_correction_pt"] = []
        return backbone_out


def build_training_model(model_cfg, checkpoint, device, **update_kwargs):
    model = build_video_predictor(
        model_cfg, checkpoint, device,
        predictor_target="training.kalman_trainer.KalmanSAM2Train",
        predictor_overrides=PROMPT_OVERRIDES,
    )
    model.eval()
    model.straight_through_object_gate = True
    model.memory_update = KalmanMemoryUpdate(model.mem_dim, model.hidden_dim, **update_kwargs).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.memory_update.parameters():
        parameter.requires_grad = True
    return model


def video_batch(images, masks) -> BatchedVideoDatapoint:
    """One video with one object: ``images`` [T, 3, H, W], ``masks`` [T, H, W]."""
    num_frames = images.size(0)
    frames = torch.arange(num_frames, dtype=torch.int, device=images.device)
    return BatchedVideoDatapoint(
        img_batch=images.unsqueeze(1),
        obj_to_frame_idx=torch.stack([frames, torch.zeros_like(frames)], dim=-1).unsqueeze(1),
        masks=masks.bool().unsqueeze(1),
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=torch.stack(
                [torch.zeros_like(frames), torch.ones_like(frames), frames], dim=-1
            ).unsqueeze(1).long(),
            frame_orig_size=torch.tensor(images.shape[-2:], device=images.device).repeat(num_frames, 1, 1),
            batch_size=[num_frames],
        ),
        dict_key="pseudo_video",
        batch_size=[num_frames],
    )


def run_clip(model, batch, run_teacher):
    """Kalman (student) outputs, and native-bank (teacher) outputs from the same prompt."""
    with torch.no_grad():
        backbone_out = model.forward_image(batch.flat_img_batch)
    backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
    student = model.forward_tracking(backbone_out, batch)
    teacher = None
    if run_teacher:
        model.kalman_enabled = False
        try:
            with torch.no_grad():
                teacher = model.forward_tracking(backbone_out, batch)
        finally:
            model.kalman_enabled = True
    return student, teacher

# TODO: careful check
class KalmanLoss(torch.nn.Module):
    """MedSAM2 fine-tuning loss on frames t >= 1, with the object-score term split by class.

    Mask terms reuse ``MultiStepMultiMasksAndIous`` with MedSAM2's focal/Dice weights
    (``sam2.1_hiera_tiny_finetune512.yaml``), applied only where the object is present
    and the presence gate is open (a closed gate is a presence error, left to the BCE).
    The IoU term is off by default: the IoU head is frozen and is not an output.
    They supervise the output mask (the candidate MedSAM2's frozen IoU head selects)
    rather than the best-matching of the candidate masks. Its object-score term is replaced by BCE averaged separately over present frames
    (weight 1) and empty frames (``absence_weight``), so the ratio does not depend on
    how many frames are empty.
    """

    def __init__(self, absence_weight=1.0, distill_weight=0.0, focal=20.0, dice=1.0, iou=0.0):
        super().__init__()
        self.absence_weight = absence_weight
        self.distill_weight = distill_weight
        self.masks = MultiStepMultiMasksAndIous(
            weight_dict={"loss_mask": focal, "loss_dice": dice, "loss_iou": iou, "loss_class": 0.0},
            supervise_all_iou=True,
            iou_use_l1_loss=True,
            pred_obj_scores=True,
        )

    def forward(self, student, teacher, masks):
        student, targets = student[1:], masks[1:].unsqueeze(1).float()
        present = targets.flatten(1).any(dim=1)
        zero = student[0]["multistep_object_score_logits"][-1].sum() * 0.0
        scores = torch.cat([frame["multistep_object_score_logits"][-1] for frame in student]).flatten()
        # A polyp frame whose object score closed the gate has mask logits of -1024; its error is the
        # presence decision (supervised below), so it gets no mask loss.
        keep = [i for i in range(len(student)) if not (present[i] and scores[i] <= 0)]
        terms = {key: zero for key in ("loss_mask", "loss_dice", "loss_iou", CORE_LOSS_KEY)}
        if keep:
            terms = self.masks([output_mask_only(student[i]) for i in keep], targets[keep])
        segmentation = terms[CORE_LOSS_KEY] / len(student)
        presence = F.binary_cross_entropy_with_logits(scores[present], torch.ones_like(scores[present])) if present.any() else zero
        absence = F.binary_cross_entropy_with_logits(scores[~present], torch.zeros_like(scores[~present])) if (~present).any() else zero
        distill = zero
        if teacher is not None and self.distill_weight > 0 and present.any():
            index = torch.nonzero(present).flatten().tolist()
            student_pixels = torch.stack([student[i]["pix_feat_with_mem"] for i in index])
            teacher_pixels = torch.stack([teacher[1:][i]["pix_feat_with_mem"] for i in index])
            distill = F.mse_loss(student_pixels, teacher_pixels) / teacher_pixels.var().clamp_min(1e-6)
        total = segmentation + presence + self.absence_weight * absence + self.distill_weight * distill
        parts = {
            "segmentation": segmentation,
            "focal": terms["loss_mask"] / len(student),
            "dice": terms["loss_dice"] / len(student),
            "iou": terms["loss_iou"] / len(student),
            "presence": presence,
            "absence": absence,
            "distill": distill,
        }
        return total, {key: float(value.detach()) for key, value in parts.items()}


def output_mask_only(frame):
    return {
        "multistep_pred_multimasks_high_res": [frame["pred_masks_high_res"]],
        "multistep_pred_ious": [frame["multistep_pred_ious"][-1].max(dim=-1, keepdim=True).values],
        "multistep_object_score_logits": frame["multistep_object_score_logits"],
    }

"""Kalman-memory training on MedSAM2's own training model, loss, and batch format."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from MedSAM2.sam2.modeling.sam2_utils import sample_box_points
from MedSAM2.training.loss_fns import CORE_LOSS_KEY, MultiStepMultiMasksAndIous
from MedSAM2.training.model.sam2 import SAM2Train
from MedSAM2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from modeling.ema_memory import ConstantGainMemory
from modeling.kalman_memory import KalmanMemoryMixin, KalmanMemoryUpdate
from modeling.rde_memory import RDEMemoryUpdate
from modeling.medsam2 import build_video_predictor

# The EMA baseline's gains (docs/README.md, --fixed_gain): the training teacher.
EMA_GAINS = (0.835, 0.266)
# Forward value of a closed gate's mask logits in the mask loss (MedSAM2 writes -1024 there).
CLOSED_GATE_LOGIT = -10.0

# Match KalmanMemoryVideoPredictor: one box prompt on frame 0, no correction clicks, eval mode.
PROMPT_OVERRIDES = {
    "prob_to_use_pt_input_for_eval": 1.0,
    "prob_to_use_box_input_for_eval": 1.0,
    "num_frames_to_correct_for_eval": 1,
    "num_init_cond_frames_for_eval": 1,
}


class KalmanSAM2Train(KalmanMemoryMixin, SAM2Train):
    """Kalman-memory training with only the frame-0 MedSAM2 prompt: a tight
    ground-truth box, as in scripts/infer.py (SAM2Train would add box noise)."""

    def __init__(self, fill_hole_area=0, **kwargs):
        # MedSAM2's video builder always passes fill_hole_area, which SAM2Train does not accept.
        super().__init__(**kwargs)

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        backbone_out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        backbone_out["frames_to_add_correction_pt"] = []
        for t in backbone_out["point_inputs_per_frame"]:
            points, labels = sample_box_points(backbone_out["gt_masks_per_frame"][t], noise=0.0)
            backbone_out["point_inputs_per_frame"][t] = {"point_coords": points, "point_labels": labels}
        return backbone_out


def build_training_model(model_cfg, checkpoint, device, update_type="kalman", **update_kwargs):
    model = build_video_predictor(
        model_cfg, checkpoint, device,
        predictor_target="training.kalman_trainer.KalmanSAM2Train",
        predictor_overrides=PROMPT_OVERRIDES,
    )
    model.eval()
    model.straight_through_object_gate = True
    update_class = RDEMemoryUpdate if update_type == "rde" else KalmanMemoryUpdate
    model.memory_update = update_class(model.mem_dim, model.hidden_dim, **update_kwargs).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.memory_update.parameters():
        parameter.requires_grad = True
    if isinstance(model.memory_update, KalmanMemoryUpdate):
        # A learned offset on every memory read, not an update rule; it grew steadily while
        # held-out Dice fell. Zero (as initialised) keeps the EMA's read path.
        model.memory_update.uncertainty_embedding.requires_grad = False
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


def run_clip(model, batch, teacher_update=None):
    """Learned-update (student) outputs, and, given ``teacher_update`` (the EMA), the
    teacher's outputs on the same clip and prompt."""
    with torch.no_grad():
        backbone_out = model.forward_image(batch.flat_img_batch)
    backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
    student = model.forward_tracking(backbone_out, batch)
    teacher = None
    if teacher_update is not None:
        learned, model.memory_update = model.memory_update, teacher_update
        try:
            with torch.no_grad():
                teacher = model.forward_tracking(backbone_out, batch)
        finally:
            model.memory_update = learned
    return student, teacher


def ema_teacher(model, gains=EMA_GAINS):
    return ConstantGainMemory(model.mem_dim, model.hidden_dim, *gains)

class KalmanLoss(torch.nn.Module):
    """MedSAM2 fine-tuning loss on frames t >= 1, with the object-score term split by class.

    Focal and Dice terms reuse ``MultiStepMultiMasksAndIous`` with MedSAM2's weights
    (``sam2.1_hiera_tiny_finetune512.yaml``) on the output mask (the candidate the frozen
    IoU head selects) of every propagated frame. A present frame whose gate closed is
    charged too, as a confident empty mask (logits ``CLOSED_GATE_LOGIT``, with the
    gradient passed straight through to the ungated mask), so losing the object is never
    cheaper than segmenting it badly. They are summed and divided by all propagated
    frames. The IoU term is off by
    default because the IoU head is frozen. The object-score BCE is a per-frame mean over
    all propagated frames (weight 1 on present frames, ``absence_weight`` on empty ones),
    so each empty frame counts as much as a present one. A per-class mean let the few
    empty frames outweigh the present ones by ~40x and taught the memory to stop
    following the object.

    With a teacher (the EMA), ``distill_weight`` pulls the memory-conditioned features
    toward it on present frames, and ``no_worse_weight`` penalises only where the student
    is worse than the teacher on a present frame: a lower object score, or a higher soft
    Dice loss. Neither term rewards departing from the teacher where the student is better.
    """

    def __init__(self, absence_weight=1.0, distill_weight=1.0, no_worse_weight=1.0,
                 focal=20.0, dice=1.0, iou=0.0):
        super().__init__()
        self.absence_weight = absence_weight
        self.distill_weight = distill_weight
        self.no_worse_weight = no_worse_weight
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
        outputs = [output_mask_only(frame, closed=bool(score <= 0)) for frame, score in zip(student, scores)]
        terms = self.masks(outputs, targets)
        segmentation = terms[CORE_LOSS_KEY] / len(student)
        bce = F.binary_cross_entropy_with_logits(scores, present.to(scores.dtype), reduction="none")
        presence = bce[present].sum() / len(student)
        absence = bce[~present].sum() / len(student)
        distill = no_worse = zero
        if teacher is not None and present.any():
            teacher = teacher[1:]
            index = torch.nonzero(present).flatten().tolist()
            student_pixels = torch.stack([student[i]["pix_feat_with_mem"] for i in index])
            teacher_pixels = torch.stack([teacher[i]["pix_feat_with_mem"] for i in index])
            distill = F.mse_loss(student_pixels, teacher_pixels) / teacher_pixels.var().clamp_min(1e-6)
            teacher_scores = torch.cat([teacher[i]["multistep_object_score_logits"][-1] for i in index]).flatten()
            student_dice = soft_dice_loss(torch.cat([outputs[i]["multistep_pred_multimasks_high_res"][0] for i in index]), targets[index])
            teacher_dice = soft_dice_loss(torch.cat([teacher[i]["pred_masks_high_res"] for i in index]), targets[index])
            no_worse = (F.relu(teacher_scores - scores[present]).sum()
                        + F.relu(student_dice - teacher_dice).sum()) / len(student)
        total = (segmentation + presence + self.absence_weight * absence
                 + self.distill_weight * distill + self.no_worse_weight * no_worse)
        parts = {
            "segmentation": segmentation,
            "focal": terms["loss_mask"] / len(student),
            "dice": terms["loss_dice"] / len(student),
            "iou": terms["loss_iou"] / len(student),
            "presence": presence,
            "absence": absence,
            "distill": distill,
            "no_worse": no_worse,
        }
        return total, {key: float(value.detach()) for key, value in parts.items()}


def soft_dice_loss(logits, targets):
    """Per-frame soft Dice loss, ``logits`` and ``targets`` [N, 1, H, W]."""
    probability, targets = logits.sigmoid().flatten(1), targets.flatten(1)
    return 1 - (2 * (probability * targets).sum(1) + 1) / (probability.sum(1) + targets.sum(1) + 1)


def output_mask_only(frame, closed=False):
    masks = frame["pred_masks_high_res"]
    if closed:  # forward value CLOSED_GATE_LOGIT, gradient straight through to the ungated mask
        masks = masks + (torch.full_like(masks, CLOSED_GATE_LOGIT) - masks).detach()
    return {
        "multistep_pred_multimasks_high_res": [masks],
        "multistep_pred_ious": [frame["multistep_pred_ious"][-1].max(dim=-1, keepdim=True).values],
        "multistep_object_score_logits": frame["multistep_object_score_logits"],
    }

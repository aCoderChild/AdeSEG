"""Memory-update training (the modified RKN with detector observations, or an
RDE-VOS-derived compression control) on MedSAM2's own training model and batch format."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from external.MedSAM2.sam2.modeling.sam2_utils import sample_box_points
from external.MedSAM2.training.loss_fns import dice_loss, sigmoid_focal_loss
from external.MedSAM2.training.model.sam2 import SAM2Train
from external.MedSAM2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from modeling.kalman_memory import RKNMemoryUpdate
from modeling.recurrent_memory import RecurrentMemory
from modeling.rde_memory import RDEMemoryUpdate
from modeling.medsam2 import build_video_predictor
# Official training code (importable once modeling.kalman_memory / rde_memory put it on the path).
from Algo.LossFunctions import GaussianLikelihoodLoss  # noqa: E402
from model.losses import BootstrappedCE  # noqa: E402
from model.model import KLDivLoss  # noqa: E402
from model.network import RDE_VOS  # noqa: E402

# Forward value of a closed gate's mask logits in the mask loss (MedSAM2 writes -1024 there).
CLOSED_GATE_LOGIT = -10.0

# Match RecurrentMemoryVideoPredictor: one box prompt on frame 0, no correction clicks, eval mode.
PROMPT_OVERRIDES = {
    "prob_to_use_pt_input_for_eval": 1.0,
    "prob_to_use_box_input_for_eval": 1.0,
    "num_frames_to_correct_for_eval": 1,
    "num_init_cond_frames_for_eval": 1,
}


class RecurrentMemorySAM2Train(RecurrentMemory, SAM2Train):
    """Memory-update training with only the frame-0 MedSAM2 prompt: a tight
    ground-truth box, as in scripts/infer.py (SAM2Train would add box noise)."""

    def __init__(self, fill_hole_area=0, **kwargs):
        # MedSAM2's video builder always passes fill_hole_area, which SAM2Train does not accept.
        super().__init__(**kwargs)
        self.collect_memory_targets = False
        self.memory_targets = None
        self._carried_target = None

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        backbone_out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        backbone_out["frames_to_add_correction_pt"] = []
        for t in backbone_out["point_inputs_per_frame"]:
            points, labels = sample_box_points(backbone_out["gt_masks_per_frame"][t], noise=0.0)
            backbone_out["point_inputs_per_frame"][t] = {"point_coords": points, "point_labels": labels}
        self.memory_targets = backbone_out["gt_masks_per_frame"] if self.collect_memory_targets else None
        self._carried_target = None
        return backbone_out

    def _encode_memory_in_output(self, current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
                                 high_res_masks, object_score_logits, current_out):
        """Also store the appearance state's true value: the memory MedSAM2 writes from the
        ground-truth mask (as logits 20m - 10, its mask-prompt encoding) of the most recent frame
        where the object is present (the prompt frame included), carried through absences."""
        pending = self._memory_step
        super()._encode_memory_in_output(current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
                                         high_res_masks, object_score_logits, current_out)
        if self.memory_targets is None or pending is None:
            return
        mask = self.memory_targets[pending["frame_idx"]].float()
        if mask.any():
            with torch.no_grad():
                score = torch.full((mask.size(0), 1), 10.0, device=mask.device)
                target, _ = self._encode_new_memory(current_vision_feats, feat_sizes, mask * 20.0 - 10.0, score, False)
            self._carried_target = target.to(torch.bfloat16).float()  # MedSAM2 stores memories in bf16
        step = current_out.get("memory")
        if step is not None:
            step["target"] = self._carried_target


def build_training_model(model_cfg, checkpoint, device, update_type="rkn", **update_kwargs):
    model = build_video_predictor(
        model_cfg, checkpoint, device,
        predictor_target="training.memory_trainer.RecurrentMemorySAM2Train",
        predictor_overrides=PROMPT_OVERRIDES,
    )
    model.eval()
    model.straight_through_object_gate = True
    update_class = RDEMemoryUpdate if update_type == "rde" else RKNMemoryUpdate
    model.collect_memory_targets = update_class is RKNMemoryUpdate
    model.memory_update = update_class(model.mem_dim, model.hidden_dim, **update_kwargs).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = getattr(model.memory_update, "trainable_networks", lambda: (model.memory_update,))()
    for network in trainable:
        for parameter in network.parameters():
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


def run_clip(model, batch, native_teacher=False, detections=None):
    """Learned-update (student) outputs, and with ``native_teacher`` MedSAM2's uncompressed
    7-frame bank on the same clip and prompt (the RDE-VOS distillation target). ``detections``
    ({clip frame index: {"box", "confidence"}}) feed an update with ``uses_detector``."""
    model.detections = detections
    with torch.no_grad():
        backbone_out = model.forward_image(batch.flat_img_batch)
    backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
    student = model.forward_tracking(backbone_out, batch)
    teacher = None
    if native_teacher:
        model.recurrent_memory_enabled = False
        try:
            with torch.no_grad():
                teacher = model.forward_tracking(backbone_out, batch)
        finally:
            model.recurrent_memory_enabled = True
    return student, teacher


def _iou(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """IoU of two boolean masks; 1 when both are empty."""
    union = (predicted | truth).sum()
    return torch.where(union > 0, (predicted & truth).sum() / union.clamp_min(1), torch.ones_like(union, dtype=torch.float))


class RKNLoss(torch.nn.Module):
    """Modified-RKN objective, on every propagated frame of the clip:

    - MedSAM2's mask loss (focal 20, Dice 1) where the object is present; a frame's mask depends on
      every earlier write, so this trains each write through the frames after it (a closed gate
      counts as an empty mask, with the gradient passed straight through to the ungated mask);
    - presence: BCE of the corrected object score toward the object's presence, each frame weighted
      by what a wrong gate costs the selection metric (polyp Dice - fp_weight x empty-frame FP
      rate): on a polyp frame the Dice of the ungated mask (closing the gate loses it) over the
      clip's polyp frames, on an empty frame fp_weight over the clip's empty frames;
    - reliability: BCE of rho toward the tracker mask's IoU with the ground truth (a soft label;
      the mask before any re-detection);
    - the official ``GaussianLikelihoodLoss`` of the appearance state under N(x, P) against the
      carried ground-truth memory (the last frame with the object).
    """

    def __init__(self, presence_weight=1.0, reliability_weight=0.5, nll_weight=0.02, fp_weight=0.5):
        super().__init__()
        self.weights = {"mask": 1.0, "presence": presence_weight, "reliability": reliability_weight, "nll": nll_weight}
        self.fp_weight = fp_weight
        self.nll = GaussianLikelihoodLoss()

    def forward(self, student, teacher, masks, step=0):
        frames, targets = student[1:], masks[1:].float()
        scores = torch.cat([frame["multistep_object_score_logits"][-1] for frame in frames]).flatten()
        present = targets.flatten(1).any(1)
        mask_terms, presence_weights, tracker_iou = [], [], []
        for frame, score, target, is_present in zip(frames, scores, targets, present):
            truth = target > 0
            tracker_iou.append(_iou(frame["tracker_masks_high_res"][0, 0] > 0, truth))
            if is_present:
                logits = mask_logits(frame, closed=bool(score <= 0)).float()
                inputs, labels = logits.flatten(1), target.flatten()[None]
                mask_terms.append(20.0 * sigmoid_focal_loss(inputs, labels, 1) + dice_loss(inputs, labels, 1))
                ungated = F.interpolate(frame["ungated_mask"].float(), size=target.shape, mode="bilinear")[0, 0] > 0
                dice = 2 * (ungated & truth).sum() / (ungated.sum() + truth.sum()).clamp_min(1)
                presence_weights.append(dice.float() / present.sum())
            else:
                presence_weights.append(torch.tensor(self.fp_weight, device=scores.device) / (~present).sum())
        zero = scores.sum() * 0.0
        presence = F.binary_cross_entropy_with_logits(scores.float(), present.float(), reduction="none")
        parts = {
            "mask": torch.stack(mask_terms).mean() if mask_terms else zero,
            "presence": (presence * torch.stack(presence_weights).detach()).sum(),
            "reliability": F.binary_cross_entropy_with_logits(
                torch.cat([frame["reliability_logit"] for frame in frames]).flatten(), torch.stack(tracker_iou).float()),
        }
        memories = [frame["memory"] for frame in frames]
        series = lambda key: torch.stack([memory[key].reshape(-1, 1) for memory in memories], dim=-1).cpu()
        covariance = torch.stack([memory["covariance"] for memory in memories], dim=-1).cpu()
        parts["nll"] = self.nll(series("mean"), series("target"), covariance).reshape(()).to(scores.device)
        total = sum(self.weights[key] * value for key, value in parts.items())
        stats = {key: float(value.detach()) for key, value in parts.items()}
        stats["tracker_iou"] = float(torch.stack(tracker_iou).float().mean())
        stats["redetected"] = sum(bool(memory.get("redetected")) for memory in memories) / len(memories)
        stats["association_present"] = _mean_where([m["association"] for m in memories], present)
        stats["association_absent"] = _mean_where([m["association"] for m in memories], ~present)
        return total, stats


def _mean_where(values, where):
    values = torch.stack([v.float().mean() for v in values])
    return float(values[where].mean()) if where.any() else None


class RDELoss(torch.nn.Module):
    """RDE-VOS loss primitives adapted to the MedSAM2 integration."""

    def __init__(self, distill_weight=10.0, temperature=1.0, top_p=0.15, start_warm=67, end_warm=233):
        super().__init__()
        self.distill_weight = distill_weight
        self.bootstrapped_ce = BootstrappedCE({"start_warm": start_warm, "end_warm": end_warm, "top_p": top_p})
        self.kl = KLDivLoss(temperature=temperature)

    def forward(self, student, teacher, masks, step=0):
        student, targets = student[1:], masks[1:].long()
        scores = torch.cat([frame["multistep_object_score_logits"][-1] for frame in student]).flatten()
        segmentation, kept = [], []
        for frame, score, target in zip(student, scores, targets):
            logits = mask_logits(frame, closed=bool(score <= 0))
            two_class = RDE_VOS.aggregate(None, torch.sigmoid(logits.float()))  # [1, 2, H, W]: background, object
            loss, kept_fraction = self.bootstrapped_ce(two_class, target.unsqueeze(0), step)
            segmentation.append(loss)
            kept.append(kept_fraction)
        segmentation = torch.stack(segmentation).mean()
        distill = segmentation * 0.0
        if teacher is not None and self.distill_weight > 0:
            distill = torch.stack([
                self.kl(frame["pix_feat_with_mem"].float(), reference["pix_feat_with_mem"].float())
                for frame, reference in zip(student, teacher[1:])
            ]).mean()
        total = segmentation + self.distill_weight * distill
        parts = {"segmentation": segmentation, "distill": distill, "bootstrap_p": torch.tensor(float(kept[-1]))}
        return total, {key: float(value.detach()) for key, value in parts.items()}


def mask_logits(frame, closed=False):
    """MedSAM2's output mask logits; a closed gate's are CLOSED_GATE_LOGIT forward, with the
    gradient passed straight through to the ungated mask."""
    masks = frame["pred_masks_high_res"]
    if closed:
        masks = masks + (torch.full_like(masks, CLOSED_GATE_LOGIT) - masks).detach()
    return masks

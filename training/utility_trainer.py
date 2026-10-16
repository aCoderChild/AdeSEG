"""Frozen-SAM2 training path for the matched-budget attention baseline."""

import torch
from torch.nn import functional as F

from external.MedSAM2.training.loss_fns import dice_loss, sigmoid_focal_loss
from external.MedSAM2.training.model.sam2 import SAM2Train
from modeling.medsam2 import build_video_predictor
from modeling.utility_memory import UtilityController, UtilityHead, attach_utility_controller


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

"""Four separately versioned mechanism arms on certified warm reads.

This module defines train/checkpoint primitives only. A label-guarded launcher,
held-out prediction, score/audit and MPS probe are required before a study.
"""
from pathlib import Path

import numpy as np
import torch

from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from modeling.multiscale_mechanism_controls_v1 import MultiscaleMechanismControlV1
from modeling.rgb_contrast_control_v1 import RGBContrastControlV1
from scripts.channel_value_fixed_read_v1 import CONFIG
from scripts.memory_token_io import digest
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.warm_crossfit_shared_fold_v1 import capture_rng, restore_rng, seed_arm

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('early_only', 'deep_only', 'spatial_shuffle', 'rgb_contrast')
BASELINE_SOURCE = ROOT / 'modeling/multiscale_channel_value_refiner_v1.py'
MECHANISM_SOURCE = ROOT / 'modeling/multiscale_mechanism_controls_v1.py'
RGB_SOURCE = ROOT / 'modeling/rgb_contrast_control_v1.py'
RGB_LOADER_SOURCE = ROOT / 'scripts/load_aligned_rgb_control_v1.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def model_for_control(arm):
    require(arm in ARMS, 'Unknown mechanism arm')
    return RGBContrastControlV1(**CONFIG) if arm == 'rgb_contrast' else MultiscaleMechanismControlV1(arm, **CONFIG)


def make_arms(expected_initial_sha, device):
    models, optimizers, states = {}, {}, {}
    for arm in ARMS:
        seed_arm(device)
        model = model_for_control(arm).train().to(device)
        require(initial_hash(model) == expected_initial_sha, 'Control initialization differs from baseline')
        models[arm] = model
        optimizers[arm] = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
        states[arm] = capture_rng(device)
    require(len({id(p) for model in models.values() for p in model.parameters()}) ==
            sum(1 for model in models.values() for _ in model.parameters()), 'Control parameters are shared')
    return models, optimizers, states


def forward(model, arm, values, rgb):
    return model(*values, rgb) if arm == 'rgb_contrast' else model(*values)


def step(model, arm, optimizer, rng_state, values, rgb, target, shape, device):
    restore_rng(rng_state, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = forward(model, arm, values, rgb)
    loss = cmsa_boundary_loss(to_video(prediction, shape), target)
    require(torch.isfinite(prediction).all() and torch.isfinite(loss), 'Nonfinite control prediction or loss')
    loss.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
            'Nonfinite control gradient')
    norm2 = sum((p.grad.float().square().sum() for p in model.parameters() if p.grad is not None),
                torch.zeros((), device=device))
    norm = float(torch.sqrt(norm2).detach().cpu())
    require(np.isfinite(norm), 'Nonfinite control gradient norm')
    optimizer.step()
    require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite control parameter')
    return float(loss.detach().cpu()), norm, capture_rng(device)


def save_payload(model, *, arm, fold, warm_plan, warm_plan_sha, study_plan_sha, steps, sources):
    require(arm in ARMS and type(fold) is int and 1 <= fold <= 5 and type(steps) is int and steps > 0,
            'Invalid control checkpoint identity')
    result = {'format': 'rgb_contrast_control_v1' if arm == 'rgb_contrast' else 'multiscale_mechanism_controls_v1',
        'research_only': True, 'training_steps': steps,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'config': CONFIG,
        'control_source_sha256': digest(RGB_SOURCE if arm == 'rgb_contrast' else MECHANISM_SOURCE),
        'baseline_source_sha256': digest(BASELINE_SOURCE), 'backbone_sha256': warm_plan['backbone_sha256'],
        'feature_pipeline_sha256': warm_plan['feature_pipeline_sha256'], 'train_manifest_sha256': warm_plan_sha,
        'crossfit_fold': fold, 'crossfit_arm': arm, 'preflight_sha256': warm_plan_sha,
        'aggregate_sha256': warm_plan['aggregate_sha256'], 'control_plan_sha256': study_plan_sha,
        'source_sha256_map': sources}
    if arm != 'rgb_contrast':
        result['control'] = arm
    return result


def trained_model(payload, *, arm, fold, warm_plan, warm_plan_sha, study_plan_sha, sources):
    require(payload.get('crossfit_fold') == fold and payload.get('crossfit_arm') == arm and
            payload.get('preflight_sha256') == warm_plan_sha and
            payload.get('aggregate_sha256') == warm_plan['aggregate_sha256'] and
            payload.get('control_plan_sha256') == study_plan_sha and payload.get('source_sha256_map') == sources,
            'Control checkpoint fold or provenance differs')
    model = model_for_control(arm)
    model.load_trained_state(payload, backbone_sha256=warm_plan['backbone_sha256'],
        feature_pipeline_sha256=warm_plan['feature_pipeline_sha256'], train_manifest_sha256=warm_plan_sha)
    return model

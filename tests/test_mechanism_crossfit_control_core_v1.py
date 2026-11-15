import copy

import pytest
import torch

from scripts.channel_value_fixed_read_v1 import model_for
from scripts.mechanism_crossfit_control_core_v1 import ARMS, make_arms, save_payload, step, trained_model
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.warm_crossfit_shared_fold_v1 import seed_arm


def test_controls_share_baseline_initial_weights_and_one_update_is_finite():
    seed_arm('cpu'); expected = initial_hash(model_for('foreground_background_asinh'))
    models, optimizers, states = make_arms(expected, 'cpu')
    values = (torch.randn(1, 32, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 1, 8, 8))
    rgb, target = torch.rand(1, 3, 16, 16), torch.ones(1, 1, 16, 16)
    for arm in ARMS:
        loss, norm, states[arm] = step(models[arm], arm, optimizers[arm], states[arm],
                                       values, rgb, target, (16, 16), 'cpu')
        assert torch.isfinite(torch.tensor(loss)) and norm > 0
        assert models[arm].head[-1].bias.abs().max() > 0


@pytest.mark.parametrize('arm', ARMS)
def test_control_checkpoint_rejects_wrong_fold_or_source(arm):
    seed_arm('cpu'); expected = initial_hash(model_for('foreground_background_asinh'))
    models, _, _ = make_arms(expected, 'cpu')
    warm = {'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64, 'aggregate_sha256': 'c'*64}
    args = dict(arm=arm, fold=1, warm_plan=warm, warm_plan_sha='d'*64, study_plan_sha='e'*64,
                sources={'example': 'f'*64})
    payload = save_payload(models[arm], steps=1, **args)
    restored = trained_model(payload, **args)
    assert initial_hash(restored) == initial_hash(models[arm])
    with pytest.raises(ValueError, match='fold or provenance'):
        trained_model(payload, **dict(args, fold=2))
    altered = copy.deepcopy(payload); altered['control_source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='checkpoint or source'):
        trained_model(altered, **args)

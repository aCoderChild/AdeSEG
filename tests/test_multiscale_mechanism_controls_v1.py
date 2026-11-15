import copy
from pathlib import Path

import pytest
import torch

from modeling.foreground_conditioned_refiner import _source_sha256
import modeling.multiscale_channel_value_refiner_v1 as baseline
import modeling.multiscale_mechanism_controls_v1 as controls


CONFIG = dict(early_channels=2, deep_channels=3, hidden=4, attention_size=4)


def inputs():
    generator = torch.Generator().manual_seed(15)
    return (torch.randn(1, 2, 5, 6, generator=generator),
            torch.randn(1, 3, 3, 4, generator=generator),
            torch.randn(1, 1, 8, 8, generator=generator))


def model(control):
    result = controls.MultiscaleMechanismControlV1(control, **CONFIG)
    torch.nn.init.constant_(result.head[-1].weight, 0.1)
    return result


def test_same_allocated_parameters_and_initial_noop():
    reference = baseline.MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh', **CONFIG)
    e, d, logits = inputs()
    for name in controls.MultiscaleMechanismControlV1.CONTROLS:
        candidate = controls.MultiscaleMechanismControlV1(name, **CONFIG)
        assert [(k, tuple(v.shape)) for k, v in candidate.state_dict().items()] == [
            (k, tuple(v.shape)) for k, v in reference.state_dict().items()]
        assert torch.equal(candidate(e, d, logits), logits)


@pytest.mark.parametrize('name,zero_index', [('early_only', 1), ('deep_only', 0)])
def test_one_branch_equals_baseline_with_that_input_zeroed(name, zero_index):
    candidate = model(name)
    reference = baseline.MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh', **CONFIG)
    reference.load_state_dict(candidate.state_dict())
    args = list(inputs())
    expected = args.copy(); expected[zero_index] = torch.zeros_like(expected[zero_index])
    assert torch.equal(candidate(*args), reference(*expected))
    first = candidate(*args)
    args[zero_index] = torch.randn_like(args[zero_index])
    assert torch.equal(first, candidate(*args))
    active = 1 - zero_index
    args[active] = args[active] + 5
    assert not torch.equal(first, candidate(*args))


def test_spatial_shuffle_is_deterministic_bijective_and_does_not_change_rng():
    x = torch.arange(24.).reshape(1, 1, 4, 6)
    before = torch.get_rng_state().clone()
    shuffled = controls.spatial_shuffle(x, seed=2107)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(shuffled, controls.spatial_shuffle(x, seed=2107))
    assert not torch.equal(shuffled, x)
    assert torch.equal(shuffled.flatten().sort().values, x.flatten().sort().values)
    candidate = model('spatial_shuffle')
    reference = baseline.MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh', **CONFIG)
    reference.load_state_dict(candidate.state_dict())
    e, d, logits = inputs()
    assert torch.equal(candidate(e, d, logits), reference(
        controls.spatial_shuffle(e, seed=2107), controls.spatial_shuffle(d, seed=2108), logits))


@pytest.mark.parametrize('name', controls.MultiscaleMechanismControlV1.CONTROLS)
def test_control_trains_and_does_not_hide_nonfinite_unused_inputs(name):
    candidate = controls.MultiscaleMechanismControlV1(name, **CONFIG)
    e, d, logits = inputs()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(candidate(e, d, logits),
                                                                  torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(candidate.head[-1].bias.grad).all()
    assert candidate.head[-1].bias.grad.abs().max() > 0
    optimizer.step()
    assert candidate.head[-1].bias.abs().max() > 0
    for index in (0, 1):
        args = [e.clone(), d.clone(), logits.clone()]
        args[index].flatten()[0] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            candidate(*args)


def test_control_checkpoint_binds_both_sources_and_cannot_cross_arms():
    net = model('early_only')
    hashes = dict(backbone_sha256='a' * 64, feature_pipeline_sha256='b' * 64,
                  train_manifest_sha256='c' * 64)
    payload = dict(format='multiscale_mechanism_controls_v1', research_only=True,
        control='early_only', training_steps=1, state_dict=copy.deepcopy(net.state_dict()),
        config=CONFIG, control_source_sha256=_source_sha256(Path(controls.__file__)),
        baseline_source_sha256=_source_sha256(Path(baseline.__file__)), **hashes)
    net.load_trained_state(payload, **hashes)
    net.eval()
    with torch.no_grad():
        assert torch.isfinite(net(*inputs())).all()
    wrong = dict(payload, control='deep_only')
    with pytest.raises(ValueError, match='Control checkpoint'):
        net.load_trained_state(wrong, **hashes)
    with torch.no_grad(), pytest.raises(RuntimeError, match='Untrained'):
        net(*inputs())
    with pytest.raises(ValueError, match='Control checkpoint'):
        controls.MultiscaleMechanismControlV1('deep_only', **CONFIG).load_trained_state(payload, **hashes)

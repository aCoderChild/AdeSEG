import copy

import pytest

from scripts.preflight_refiner_hardcase_v3 import validate_samples
from scripts.train_region_conditioned_hardcase_v1 import ARMS, make_model


def samples():
    result = []
    for seq in ('seq2', 'seq3'):
        result.extend({'sequence': seq, 'frame': i, 'split': 'train'} for i in range(5))
    result.extend({'sequence': 'seq4', 'frame': i, 'split': 'train'}
                  for i in (10, 17, 18, 19, 21, 24, 28, 38, 47))
    for seq in ('seq9', 'seq10'):
        result.extend({'sequence': seq, 'frame': i, 'split': 'validation'} for i in range(5))
    return result


def test_hardcase_split_excludes_duplicates_and_holds_new_cases_in_training():
    rows = samples()
    validate_samples(rows)
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match='distinct'):
        validate_samples(duplicate)
    changed = copy.deepcopy(rows)
    next(x for x in changed if x['sequence'] == 'seq4' and x['frame'] == 24)['frame'] = 25
    with pytest.raises(ValueError, match='seq4'):
        validate_samples(changed)


def test_gate_controls_have_same_capacity_and_reject_unknown_arm():
    assert len({sum(p.numel() for p in make_model(arm).parameters()) for arm in ARMS}) == 1
    with pytest.raises(ValueError, match='Unknown'):
        make_model('other')

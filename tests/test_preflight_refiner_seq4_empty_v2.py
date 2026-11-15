import copy

import pytest

from scripts.preflight_refiner_seq4_empty_v2 import validate_samples


def _samples():
    result = []
    for seq in ('seq2', 'seq3', 'seq4'):
        result.extend({'sequence': seq, 'frame': i, 'split': 'train'} for i in range(5))
    result.extend({'sequence': 'seq4', 'frame': i, 'split': 'train'} for i in (17, 18))
    for seq in ('seq9', 'seq10'):
        result.extend({'sequence': seq, 'frame': i, 'split': 'validation'} for i in range(5))
    return result


def test_extension_has_exact_split_and_no_duplicate_overlap():
    samples = _samples()
    validate_samples(samples)
    duplicate = copy.deepcopy(samples)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match='unique'):
        validate_samples(duplicate)
    wrong_split = copy.deepcopy(samples)
    wrong_split[15]['split'] = 'validation'
    with pytest.raises(ValueError, match='17 train'):
        validate_samples(wrong_split)
    missing = copy.deepcopy(samples)
    missing[16]['frame'] = 19
    with pytest.raises(ValueError, match='Missing label-selected'):
        validate_samples(missing)

import numpy as np
import pytest
import torch

from scripts.generate_utility_labels import _attention_by_unit, soft_dice


def test_soft_dice_is_probability_based_and_keeps_hard_tie_distinct():
    target = np.array([[1, 0]], dtype=bool)
    confident = torch.tensor([[[[8.0, -8.0]]]])
    uncertain = torch.tensor([[[[0.1, -0.1]]]])
    assert soft_dice(confident, target) > soft_dice(uncertain, target)


def test_attention_saliency_uses_only_foreground_queries_and_keeps_layers():
    trace = {
        "units": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        "foreground_query_mask": torch.tensor([[True, False]]),
        "attention": [
            torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]]),
            torch.tensor([[[[0.6, 0.4], [0.2, 0.8]]]]),
        ],
    }
    aggregate, layers = _attention_by_unit(trace)
    np.testing.assert_allclose(layers, [[0.8, 0.2], [0.6, 0.4]], atol=1e-7)
    assert aggregate == pytest.approx([0.7, 0.3])

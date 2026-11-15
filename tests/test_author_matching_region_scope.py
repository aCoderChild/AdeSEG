import pytest
import torch
import torch.nn.functional as F
from modeling.soft_role_affinity import mask_logits_to_soft_roles
from scripts.report_author_matching_region_scope import window_scope


def test_mixed_regions_can_restrict_matching_without_changing_the_token_budget():
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[:, :2] = 1
    result = window_scope(F.one_hot(labels, 3).float())
    assert result['windows'] == 256
    assert result['mixed_role_windows'] == 16
    assert result['mixed_windows_with_unchanged_budget'] == 16
    assert sum(result['merges_per_window']) == 2048
    assert sum(result['role_token_counts']) == 4096


def test_native_negative_mask_yields_no_local_region_constraint():
    logits = -torch.linspace(.001, 20., 256*256).reshape(256, 256)
    result = window_scope(mask_logits_to_soft_roles(logits))
    assert result['mixed_role_windows'] == 0
    assert result['role_token_counts'] == [4096, 0, 0]
    with pytest.raises(ValueError):
        window_scope(torch.ones(64, 64, 3))


def test_quarter_budget_counts_per_role_rounding():
    labels = torch.zeros(4, 4, dtype=torch.long)
    labels[0, :3] = 1  # Counts 13/3/0: quarter budget is 3, not 4 or 7.
    roles = F.one_hot(labels, 3).float()
    quarter = window_scope(roles, merge_divisor=4)
    assert quarter['merges_per_window'] == [3]
    assert quarter['mixed_windows_with_unchanged_budget'] == 0
    assert window_scope(roles)['merges_per_window'] == [7]
    labels[0, 3] = 1  # Counts 12/4/0: all four quarter-budget merges remain feasible.
    quarter = window_scope(F.one_hot(labels, 3).float(), merge_divisor=4)
    assert quarter['merges_per_window'] == [4]
    assert quarter['mixed_windows_with_unchanged_budget'] == 1


@pytest.mark.parametrize('divisor', [True, 1, 0, 2.5])
def test_invalid_merge_divisor_rejected(divisor):
    with pytest.raises(ValueError, match='Merge divisor'):
        window_scope(F.one_hot(torch.zeros(4, 4, dtype=torch.long), 3).float(),
                     merge_divisor=divisor)

import torch

from modeling.cfbi_region_gate import shuffle_regions
from scripts.run_paired_readout_pilot import stability_schedule


def test_schedules_are_balanced_reproducible_and_distinct():
    a = stability_schedule(20261101)
    b = stability_schedule(20261102)
    assert a == stability_schedule(20261101)
    assert a != b
    assert all(sorted(a[i:i + 3]) == [0, 1, 2] for i in range(0, 24, 3))
    assert all(sorted(b[i:i + 3]) == [0, 1, 2] for i in range(0, 24, 3))


def test_shuffled_labels_keep_counts_and_vary_by_seed():
    labels = torch.arange(120) % 3
    a = shuffle_regions(labels, 'seq2', 1, 20261101)
    b = shuffle_regions(labels, 'seq2', 1, 20261102)
    assert torch.equal(a, shuffle_regions(labels, 'seq2', 1, 20261101))
    assert not torch.equal(a, b)
    assert torch.equal(torch.bincount(a), torch.bincount(labels))
    assert torch.equal(torch.bincount(b), torch.bincount(labels))

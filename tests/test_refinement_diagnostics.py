import torch

from scripts.diagnose_regional_refinement import donor_mapping, paired_modes
from modeling.regional_refinement import RegionalRefinement


def test_donors_are_cross_sequence_and_pattern_matched():
    rows = [{'sequence': n, 'frame': i} for i, n in enumerate(['seq16', 'seq16', 'seq17', 'seq18'])]
    patterns = [(True,True,True), (True,False,True), (True,True,True), (True,True,True)]
    mapping = donor_mapping(rows, patterns)
    assert mapping[1] is None
    assert mapping == donor_mapping(rows, patterns)
    for i, j in enumerate(mapping):
        if j is not None:
            assert rows[i]['sequence'] != rows[j]['sequence']
            assert patterns[i] == patterns[j]


def test_paired_modes_restores_all_buffers_and_weights():
    torch.manual_seed(3)
    head = RegionalRefinement()
    with torch.no_grad():
        head.head.conv_out4.conv.weight.fill_(.01)
    state = {k: v.clone() for k, v in head.state_dict().items()}
    inputs = (torch.randn(4,256,8,8), torch.randn(4,64,3,1),
              torch.ones(4,3,dtype=torch.bool), torch.randn(4,1,32,32), torch.ones(4))
    a, b = paired_modes(head, state, inputs)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not head.training
    assert all(torch.equal(v, state[k]) for k,v in head.state_dict().items())
    again, _ = paired_modes(head, state, inputs)
    assert torch.equal(a, again)

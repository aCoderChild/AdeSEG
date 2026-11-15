from copy import deepcopy

import pytest

from scripts.verify_token_window_refiner_routing_mps import verify_routes


def make_runs():
    rows = [{'frame': 9, 'mask_sha256': 'm0', 'logits_sha256': 'l0'},
            {'frame': 10, 'mask_sha256': 'm1', 'logits_sha256': 'l1',
             'pred_masks_sha256': 'p1', 'maskmem_features_sha256': 's1',
             'obj_ptr_sha256': 'o1', 'object_score_logits_sha256': 'v1'},
            {'frame': 11, 'mask_sha256': 'm2', 'logits_sha256': 'l2',
             'pred_masks_sha256': 'p2', 'maskmem_features_sha256': 's2',
             'obj_ptr_sha256': 'o2', 'object_score_logits_sha256': 'v2'}]
    base = {'rows': rows, 'token_window_events': [1], 'selection_events': [2], 'positions': [3]}
    runs = {'baseline': deepcopy(base)}
    for arm in 'ABCD':
        runs['identity_' + arm] = deepcopy(base)
    for arm in 'BCD':
        forced = deepcopy(base)
        if arm in 'BD':
            forced['rows'][1]['logits_sha256'] = 'forced'
        if arm in 'CD':
            forced['rows'][1]['maskmem_features_sha256'] = 'forced'
        runs['forced_' + arm] = forced
    return runs


def test_routes_accept_expected_output_and_memory_separation():
    verify_routes(make_runs())


@pytest.mark.parametrize('run,row,key', [
    ('identity_C', 2, 'maskmem_features_sha256'),
    ('forced_B', 2, 'pred_masks_sha256'),
    ('forced_C', 1, 'obj_ptr_sha256'),
    ('forced_D', 1, 'object_score_logits_sha256'),
])
def test_routes_fail_on_compact_state_contamination(run, row, key):
    runs = make_runs()
    runs[run]['rows'][row][key] = 'wrong'
    with pytest.raises(ValueError):
        verify_routes(runs)

import pytest

from scripts.verify_key_adapter_capacity_identity import require_owned_capacity


def calls(ids=(1, 2)):
    return [{'object_id': owner, 'frame': frame, 'layer': layer,
             'spatial_keys': min(frame, 7) * 4096, 'spatial_frames': min(frame, 7),
             'pointer_keys': 4, 'position_aligned_before_adapter': True,
             'values_unchanged': True, 'pointer_keys_unchanged': True,
             'key_delta_l1': 0.0, 'key_delta_nonzero': 0}
            for frame in range(1, 9) for owner in ids for layer in range(4)]


def test_full_capacity_owned_reads_pass():
    require_owned_capacity(calls(), (1, 2))


def test_wrong_read_owner_fails():
    value = calls()
    value[0]['object_id'] = 42
    with pytest.raises(RuntimeError):
        require_owned_capacity(value, (1, 2))


def test_missing_full_capacity_is_not_a_pass():
    value = calls()
    for row in value:
        row['spatial_keys'] = min(row['spatial_keys'], 24576)
    with pytest.raises(RuntimeError):
        require_owned_capacity(value, (1, 2))


def test_repeated_frame_does_not_replace_missing_frame():
    value = calls()
    value[-1]['frame'] = 7
    with pytest.raises(RuntimeError):
        require_owned_capacity(value, (1, 2))


@pytest.mark.parametrize('field,value', [('pointer_keys', 0), ('values_unchanged', False), ('spatial_keys', 32768)])
def test_native_invariant_violation_fails(field, value):
    rows = calls()
    rows[-1][field] = value
    with pytest.raises(RuntimeError):
        require_owned_capacity(rows, (1, 2))

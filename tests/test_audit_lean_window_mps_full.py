import pytest

from scripts.audit_lean_window_mps_full import verify_identities


def test_recorded_identity_must_match_for_each_replay():
    expected = {'runner_sha256': 'a', 'mps_rope_sha256': 'b', 'torch': 'v'}
    valid = {'passed': True, 'device': 'mps', 'fallback': '0', **expected}
    verify_identities([valid, dict(valid)], expected)
    for key, replacement in [('runner_sha256', 'different'), ('mps_rope_sha256', 'different'),
                             ('torch', 'different'), ('device', 'cpu'), ('fallback', '1'),
                             ('passed', False)]:
        changed = {**valid, key: replacement}
        with pytest.raises(ValueError):
            verify_identities([valid, changed], expected)

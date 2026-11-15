import pytest

from scripts import run_boundary_video_factorial_v1 as runner


def test_twelve_matched_full_runs_precede_single_scoring_phase(tmp_path):
    phases, mapping = runner.commands(tmp_path, tmp_path / 'gate.json')
    assert [x[0] for x in phases] == [f'{seq}_{arm}' for seq in runner.SEQUENCES for arm in runner.ARMS] + ['score']
    for name, cmd in phases[:-1]:
        seq, arm = name.split('_')
        assert cmd[cmd.index('--sequence') + 1] == seq
        assert cmd[cmd.index('--arm') + 1] == arm
        assert cmd[cmd.index('--output') + 1] == mapping[seq][arm]
        assert '--limit' not in cmd
        assert cmd[cmd.index('--engineering-gate') + 1] == str(tmp_path / 'gate.json')
        if arm == 'A':
            assert '--matched-a' not in cmd
        else:
            assert cmd[cmd.index('--matched-a') + 1] == mapping[seq]['A']
    assert phases[-1][1][phases[-1][1].index('--run-map') + 1] == str(tmp_path / 'run_map.json')


def test_free_space_guard_refuses_before_phase(monkeypatch, tmp_path):
    class Usage:
        free = runner.GUARD - 1
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: Usage())
    with pytest.raises(RuntimeError, match='below'):
        runner.free_bytes(tmp_path)


def test_child_environment_matches_short_gate(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', 'different')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '1')
    env = runner.phase_environment()
    assert env['PYTHONHASHSEED'] == '0'
    assert env['PYTORCH_MPS_FAST_MATH'] == '0'
    assert env['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'

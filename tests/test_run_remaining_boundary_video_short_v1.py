import os
from pathlib import Path
import pytest

from scripts import run_remaining_boundary_video_short_v1 as runner


def test_short_phases_have_two_baselines_and_four_routing_arms(tmp_path):
    phases,mapping=runner.commands(tmp_path,Path('/pre'),Path('/old-gate'))
    assert [name for name,_ in phases] == [f'{s}_{a}' for s in runner.SEQUENCES for a in runner.ARMS]+['gate','independent_audit']
    for seq in runner.SEQUENCES:
        for arm in runner.ARMS:
            cmd=dict(phases)[f'{seq}_{arm}']
            assert cmd[cmd.index('--sequence')+1] == seq
            assert cmd[cmd.index('--arm')+1] == ('A' if arm=='A_repeat' else arm)
            assert cmd[cmd.index('--limit')+1] == '4'
            assert ('--matched-a' in cmd) is (arm in ('B','C','D'))
            if arm in ('B','C','D'):assert cmd[cmd.index('--matched-a')+1] == mapping[seq]['A']
            assert '--extension-gate' not in cmd and '--extension-audit' not in cmd


def test_short_children_force_fixed_environment(monkeypatch):
    for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTHONHASHSEED'):
        monkeypatch.setenv(key,'1')
    env=runner.phase_environment()
    assert all(env[key]=='0' for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTHONHASHSEED'))


def test_short_inputs_must_be_under_experiment_root(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(runner,'BASE',root)
    with pytest.raises(ValueError,match='experiment root'):
        runner.check_inputs(root/'pre',tmp_path/'outside',root/'audit')

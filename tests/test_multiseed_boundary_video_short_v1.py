import json
import sys
from pathlib import Path

import pytest

from scripts import run_multiseed_boundary_refiner_v1 as runner
from scripts import run_multiseed_boundary_video_short_v1 as launcher
from scripts import verify_multiseed_boundary_video_short_gate_v1 as gate


def test_short_plan_is_two_distinct_seed_factorials_then_independent_audits(tmp_path):
    phases,mapping=launcher.commands(tmp_path)
    assert len(phases)==24
    assert set(mapping)=={'20261125','20261135'}
    for seed in launcher.SEEDS:
        names=[n for n,s,_ in phases if s==seed]
        assert names == [f'{seed}_{seq}_{arm}' for seq in launcher.SEQUENCES for arm in launcher.ARMS]+[f'{seed}_gate',f'{seed}_independent_audit']
        for seq in launcher.SEQUENCES:
            for arm in launcher.ARMS:
                cmd=dict((n,c) for n,_,c in phases)[f'{seed}_{seq}_{arm}']
                assert cmd[cmd.index('--seed')+1]==str(seed)
                assert cmd[cmd.index('--limit')+1]=='4'
                assert ('--matched-a' in cmd) is (arm in ('B','C','D'))


def test_gate_rejects_wrong_seed_and_incomplete_mapping(tmp_path):
    path=tmp_path/'gate.json';path.write_text(json.dumps({'passed':True,'seed':20261125}))
    with pytest.raises(ValueError,match='incomplete'):
        gate.require_gate(path,20261135)
    with pytest.raises(ValueError,match='Unexpected study seed'):
        gate.verify_mapping({},1)
    with pytest.raises(ValueError,match='Two full'):
        gate.verify_mapping({'seq4':{}},20261125)


def test_full_gate_rejects_changed_detector_or_seq8_preflight(tmp_path,monkeypatch):
    from scripts.memory_token_io import digest
    detector=tmp_path/'detector';detector.write_bytes(b'detector')
    pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text('{}');(pre/'completion.json').write_text('{}')
    bound={'detector_sha256':digest(detector),'seq8_preflight_sha256':digest(pre/'manifest.json'),
           'seq8_preflight_completion_sha256':digest(pre/'completion.json'),'runtime_identity':{}}
    monkeypatch.setattr(gate,'require_gate',lambda *_:bound)
    common=dict(multiseed_root=tmp_path,seed=20261125,checkpoint=tmp_path/'ckpt',detector=detector,
                seq8_preflight=pre,split_preflight=pre,baseline_reference=pre,seq4_capture=pre)
    detector.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Detector or seq8 preflight'):
        gate.verify_full_gate(tmp_path/'gate',**common)
    detector.write_bytes(b'detector');(pre/'completion.json').write_text('tampered')
    with pytest.raises(ValueError,match='Detector or seq8 preflight'):
        gate.verify_full_gate(tmp_path/'gate',**common)


def test_full_inference_requires_gate_and_independent_audit_before_inputs(tmp_path,monkeypatch):
    args=['run']
    for key in ('multiseed-root','split-preflight','checkpoint','detector','baseline-reference','seq4-capture','output','seq8-preflight'):
        args += ['--'+key,str(tmp_path/key)]
    args += ['--seed','20261125','--sequence','seq8','--arm','A']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='gate and independent audit'):runner.main()
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('sequence,limit',[('seq8',4),('seq11',4),('seq4',1),('seq23',5)])
def test_only_declared_four_step_short_clips_are_permitted(tmp_path,monkeypatch,sequence,limit):
    args=['run']
    for key in ('multiseed-root','split-preflight','checkpoint','detector','baseline-reference','seq4-capture','output','seq8-preflight'):
        args += ['--'+key,str(tmp_path/key)]
    args += ['--seed','20261125','--sequence',sequence,'--arm','A','--limit',str(limit)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='Only seq4/seq23'):runner.main()


def test_short_launcher_binds_seq8_prompt_preflight():
    inputs=launcher.input_paths(); sources=launcher.source_paths()
    assert inputs['seq8_preflight'].name=='manifest.json'
    assert inputs['seq8_preflight_completion'].name=='completion.json'
    assert sources['seq8_preflight_source'].name=='preflight_extended_boundary_video_v1.py'

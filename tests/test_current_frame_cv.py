import json
from pathlib import Path
import pytest
from scripts.run_current_frame_cv import validate_folds, sampled_targets, cleanup_new_tensors


def test_folds_partition_without_leakage():
    folds=json.loads(Path('protocols/current_frame_cv_v1.json').read_text())['validation_folds']
    names=validate_folds(folds)
    for val in folds:
        train=set(names)-set(val)
        assert len(train)==14 and not train.intersection(val)
    with pytest.raises(ValueError):validate_folds([folds[0],folds[0],folds[2]])


def test_sampling_before_outcomes_and_safe_cleanup(tmp_path):
    assert sampled_targets([2,3,4])==[2,3,4]
    assert len(sampled_targets(list(range(100))))==16
    assert sampled_targets(list(range(100)))[-1]==99
    folder=tmp_path/'new';folder.mkdir()
    old=tmp_path/'old.pt';old.write_text('original')
    with pytest.raises(RuntimeError):cleanup_new_tensors([{'path':str(old)}],folder)
    assert old.read_text()=='original'


def test_resume_requires_untampered_scores_checkpoint_and_complete_frames(tmp_path):
    from scripts.run_current_frame_cv import verified_sequence_result
    from scripts.memory_token_io import digest
    from scripts.run_regional_refinement import write
    checkpoint=tmp_path/'head.pt';checkpoint.write_bytes(b'checkpoint')
    result=tmp_path/'seq2_results.json'
    write(result,{'frames':{'seq2':[{'frame':1},{'frame':2}]},'sequences':{'seq2':{'present_dice':.5}}})
    write(tmp_path/'seq2_complete.json',{'checkpoint_sha256':digest(checkpoint),'results_sha256':digest(result)})
    assert verified_sequence_result(tmp_path,'seq2',[1,2])['sequences']['seq2']['present_dice']==.5
    with pytest.raises(RuntimeError,match='coverage'):verified_sequence_result(tmp_path,'seq2',[1,2,3])
    original=result.read_bytes();result.write_bytes(original+b' ')
    with pytest.raises(RuntimeError,match='integrity'):verified_sequence_result(tmp_path,'seq2',[1,2])
    result.write_bytes(original);checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(RuntimeError,match='integrity'):verified_sequence_result(tmp_path,'seq2',[1,2])


def test_compatibility_accepts_only_exact_reviewed_hashes(tmp_path,monkeypatch):
    from scripts import run_current_frame_cv as runner
    from scripts.memory_token_io import digest
    source=tmp_path/'source.py';source.write_text('repaired')
    amendment=tmp_path/'amendment.json'
    amendment.write_text(json.dumps({'repairs':{'source.py':{'before':'old','after':digest(source)}}}))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'COMPATIBILITY','amendment.json')
    runner.check_compatible_sources({'source.py':'old'})
    with pytest.raises(RuntimeError):runner.check_compatible_sources({'source.py':'different historical revision'})
    source.write_text('unreviewed')
    with pytest.raises(RuntimeError):runner.check_compatible_sources({'source.py':'old'})

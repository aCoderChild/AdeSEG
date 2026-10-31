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

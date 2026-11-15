"""CPU-only split and label-free capture-index preflight tests."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import preflight_exploratory_crossfit_capture_v2 as pre


def test_user_reported_patient_groups_cover_all_23_without_splitting_known_stems():
    owner=pre.verify_folds()
    assert len(owner)==23
    assert {owner[s] for s in ('seq6','seq10','seq11','seq20','seq23')}=={0}
    assert owner['seq1']==owner['seq2'] and owner['seq3']==owner['seq4']
    assert owner['seq7']==owner['seq8']


def test_uniform_index_grid_excludes_initialization_and_handles_short_video():
    assert pre.uniform_indices(13,25)==list(range(14,25))
    rows=pre.uniform_indices(0,100)
    assert len(rows)==12 and len(set(rows))==12 and rows[0]==1 and rows[-1]==99
    with pytest.raises(ValueError,match='post-prompt'):pre.uniform_indices(4,5)


def test_cross_fold_duplicate_is_rejected_and_same_fold_is_recorded():
    owner=pre.verify_folds();seen={}
    assert pre.register_image(seen,'key','seq1',0,owner,'decoded') is None
    assert pre.register_image(seen,'key','seq2',3,owner,'decoded')['sequences']==['seq1','seq2']
    with pytest.raises(ValueError,match='Cross-fold decoded'):
        pre.register_image(seen,'key','seq8',1,owner,'decoded')


def test_source_frame_index_mismatch_and_legacy_metadata_are_fail_closed():
    frame=[{'source_frame_index':42,'image_sha256':'image'}]
    modern=[{'frame':0,'sha256':'image','source_frame_index':42}]
    assert 'recorded in baseline' in pre.validate_baseline_rows('seq8',modern,frame)
    modern[0]['source_frame_index']=43
    with pytest.raises(ValueError,match='source frame index'):
        pre.validate_baseline_rows('seq8',modern,frame)
    legacy=[{'frame':0,'sha256':'image'}]
    assert 'not recorded' in pre.validate_baseline_rows('seq11',legacy,frame,[42])
    with pytest.raises(ValueError,match='Source index'):
        pre.validate_baseline_rows('seq11',legacy,frame,[43])
    with pytest.raises(ValueError,match='separately bound'):
        pre.validate_baseline_rows('seq11',legacy,frame)


def test_evidence_projection_does_not_deserialize_unrelated_outcomes(tmp_path):
    file=tmp_path/'evidence.json'
    file.write_text('{\n  "passed": true,\n  "outcomes": deliberately-not-json\n}')
    assert pre.project_field(file,'passed') is True
    with pytest.raises(json.JSONDecodeError):json.loads(file.read_text())


def test_corrected_loader_returns_actual_real_manifest_digest_without_opening_labels():
    if not pre.EXISTING.exists():pytest.skip('Drive-backed capture manifest is not mounted')
    code='''from pathlib import Path
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts import preflight_exploratory_crossfit_capture_v2 as target
from scripts.source_group_refiner_shared_v2 import load_prepared as corrected_loader
from scripts.memory_token_io import digest
g=install_mask_open_guard(Path("data/polypgen_sequence.jsonl"))
assert target.load_prepared is corrected_loader
_,returned=target.load_prepared(target.EXISTING)
assert returned==digest(target.EXISTING/"manifest.json")
assert g["open_attempts"]==0
'''
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)

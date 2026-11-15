"""Patient-disjoint three-fold contract checks, without target data or MPS."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts import crossfit_training_contract_3fold_v1 as contract


def test_three_folds_cover_every_patient_once_and_keep_known_pairs_together():
    expected = {f'seq{i}' for i in range(1, 24)}
    test_sets = []
    for fold in range(1, 4):
        train, validation, test = contract.fold_split(fold)
        assert not validation and not (train & test) and train | test == expected
        assert len(test - set(contract.NOINIT)) == 7
        test_sets.append(test)
    assert Counter(s for test in test_sets for s in test) == Counter(expected)
    for a, b in (('seq1', 'seq2'), ('seq3', 'seq4'), ('seq6', 'seq10'), ('seq7', 'seq8')):
        assert any({a, b} <= test for test in test_sets)
    for fold in (0, 4, True, '1'):
        with pytest.raises(ValueError):
            contract.fold_split(fold)


def test_schedule_is_deterministic_balanced_and_excludes_test_and_noinit():
    rows = [{'sequence': f'seq{i}', 'frame': frame}
            for i in range(1, 24) if f'seq{i}' not in contract.NOINIT for frame in range(3)]
    for fold in range(1, 4):
        train, _, test = contract.fold_split(fold)
        order = contract.patient_balanced_schedule(rows, fold, seed=41, epochs=2, updates_per_patient=5)
        assert order == contract.patient_balanced_schedule(rows, fold, seed=41, epochs=2, updates_per_patient=5)
        for epoch in order:
            assert Counter(seq for seq, _ in epoch) == Counter({s: 5 for s in train - set(contract.NOINIT)})
            assert not ({s for s, _ in epoch} & (test | set(contract.NOINIT)))
    with pytest.raises(ValueError, match='duplicate'):
        contract.patient_balanced_schedule(rows + [rows[0]], 1, seed=41, epochs=1)


def test_label_hash_projection_rejects_held_out_patients_before_access():
    for fold in range(1, 4):
        _, _, test = contract.fold_split(fold)
        held_out = next(s for s in test if s not in contract.NOINIT)
        with pytest.raises(PermissionError, match='training sequences'):
            contract.training_label_hashes(fold, [{'sequence': held_out}], {})


def test_fold_guard_allows_training_only_in_a_fresh_process(tmp_path):
    rows = []
    for number in range(1, 24):
        path = tmp_path / f'seq{number}.png'
        path.write_bytes(b'guard-only')
        rows.append({'video_id': f'seq{number}', 'mask': str(path)})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    script = '''
import sys
from scripts.crossfit_training_contract_3fold_v1 import install_fold_training_guard
guard = install_fold_training_guard(sys.argv[1], 1)
with open(sys.argv[2], 'rb') as f: assert f.read() == b'guard-only'
for path in sys.argv[3:]:
    try: open(path, 'rb')
    except PermissionError: continue
    raise AssertionError('test or no-init label opened')
assert guard['blocked_open_attempts'] == 3
'''
    # seq6 trains; seq2/11 are held-out; seq7 is no-init even on the training side.
    paths = [rows[i - 1]['mask'] for i in (6, 2, 11, 7)]
    result = subprocess.run([sys.executable, '-c', script, str(manifest), *paths],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

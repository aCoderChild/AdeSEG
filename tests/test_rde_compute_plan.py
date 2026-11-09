import json
import pytest
from scripts.benchmark_rde_value_repair import CONDITIONS, schedule, prediction_identity


def test_balanced_fresh_process_plan_has_no_test_or_validation_sequences():
    jobs = schedule()
    assert len(jobs) == 24
    assert {j['sequence'] for j in jobs} == {'seq2', 'seq3'}
    for sequence in ('seq2', 'seq3'):
        subset = [j for j in jobs if j['sequence'] == sequence]
        assert [j['phase'] for j in subset[:3]] == ['warmup'] * 3
        for index in range(3):
            assert {subset[3 + 3 * r + index]['condition'] for r in range(3)} == set(CONDITIONS)


def test_prediction_identity_rejects_empty_or_duplicate_frames(tmp_path):
    path = tmp_path / 'frames.jsonl'
    path.write_text('')
    with pytest.raises(RuntimeError, match='Missing or duplicate'):
        prediction_identity(path)
    row = {'frame': 1, 'prediction_sha256': 'logit', 'mask_sha256': 'mask'}
    path.write_text(json.dumps(row) + '\n')
    assert prediction_identity(path) == [(1, 'logit', 'mask')]
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(RuntimeError, match='Missing or duplicate'):
        prediction_identity(path)

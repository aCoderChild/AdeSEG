"""A train-only diagnostic cannot borrow an incomplete or mismatched gate."""
import json

import pytest

from scripts.diagnose_key_adapter_optimization import require_gate
from scripts.memory_token_io import digest


def write_gate(tmp_path, *, scope='engineering gate', passed=True):
    result = tmp_path / 'results.json'
    result.write_text(json.dumps({'passed': passed, 'scope': scope, 'identity': {'sources': 'frozen'}}))
    (tmp_path / 'completion.json').write_text(json.dumps({'passed': passed, 'results_sha256': digest(result)}))
    return result


def test_completed_gate_matches_sources(tmp_path):
    require_gate(write_gate(tmp_path), {'sources': 'frozen'})


@pytest.mark.parametrize('scope,passed', [('TRAIN-only optimization diagnostic', True), ('engineering gate', False)])
def test_training_result_or_failed_gate_cannot_release_diagnostic(tmp_path, scope, passed):
    with pytest.raises(RuntimeError):
        require_gate(write_gate(tmp_path, scope=scope, passed=passed), {'sources': 'frozen'})


def test_changed_sources_cannot_reuse_gate(tmp_path):
    with pytest.raises(RuntimeError):
        require_gate(write_gate(tmp_path), {'sources': 'changed'})


def test_modified_result_rejected_even_if_identity_matches(tmp_path):
    result = write_gate(tmp_path)
    result.write_text(result.read_text() + '\n')
    with pytest.raises(RuntimeError):
        require_gate(result, {'sources': 'frozen'})

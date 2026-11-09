"""Reference provenance must fail closed before fresh-process continuation."""
import json

import pytest

from scripts.memory_token_io import digest
from scripts.verify_key_adapter_process_resume import require_reference


def reference(tmp_path, **changes):
    result = {'phase': 'reference', 'identity': {'source': 'a'}, 'efficacy_authorized': False}
    result.update(changes)
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(result))
    (tmp_path / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(path)}))
    return path


def test_unchanged_engineering_reference_is_accepted(tmp_path):
    reference(tmp_path)
    assert require_reference(tmp_path, {'source': 'a'})['phase'] == 'reference'


def test_modified_reference_is_rejected(tmp_path):
    path = reference(tmp_path)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(RuntimeError):
        require_reference(tmp_path, {'source': 'a'})


@pytest.mark.parametrize('changes', [{'phase': 'resume'}, {'efficacy_authorized': True}, {'identity': {'source': 'b'}}])
def test_wrong_reference_scope_or_identity_is_rejected(tmp_path, changes):
    reference(tmp_path, **changes)
    with pytest.raises(RuntimeError):
        require_reference(tmp_path, {'source': 'a'})

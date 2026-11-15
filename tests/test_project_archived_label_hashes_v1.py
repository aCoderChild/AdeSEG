from pathlib import Path
import hashlib

import pytest

from scripts.project_archived_label_hashes_v1 import project_label_hashes


def test_projects_only_permitted_hashes_without_parsing_outcome_fields(tmp_path):
    file=tmp_path/'frames.json';a='a'*64;b='b'*64
    file.write_text('[\n {\n "label_sha256": "'+a+'",\n "outcome": INVALID\n },\n {\n "label_sha256": "'+b+'"\n }\n]')
    binding={'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    assert project_label_hashes(binding,2,[1])=={1:b}
    file.write_text(file.read_text()+' ')
    with pytest.raises(ValueError,match='metadata changed'):project_label_hashes(binding,2,[1])


def test_rejects_duplicate_or_out_of_range_ordinals(tmp_path):
    file=tmp_path/'frames.json';file.write_text('  "label_sha256": "'+'a'*64+'"\n')
    binding={'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='ordinals'):project_label_hashes(binding,1,[0,0])
    with pytest.raises(ValueError,match='ordinals'):project_label_hashes(binding,1,[1])

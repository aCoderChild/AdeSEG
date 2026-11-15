import hashlib
import json

import pytest

from scripts import verify_refiner_capture_static as verify
from scripts.verify_refiner_capture_static import check_canonical_pipeline, check_files


def test_source_change_is_rejected(tmp_path):
    source=tmp_path/'source.py'; source.write_text('original\n')
    expected={'source':hashlib.sha256(source.read_bytes()).hexdigest()}
    check_files(expected,{'source':source})
    source.write_text('modified\n')
    with pytest.raises(ValueError,match='source'):
        check_files(expected,{'source':source})


def test_canonical_pipeline_binds_values_not_mapping_order():
    pipeline={'feature':'early','runtime':{'a':1,'b':2}}
    checksum=hashlib.sha256(json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check_canonical_pipeline({'runtime':{'b':2,'a':1},'feature':'early'},checksum)
    with pytest.raises(ValueError,match='pipeline'):
        check_canonical_pipeline({'feature':'deep','runtime':{'a':1,'b':2}},checksum)


def test_detector_requires_expected_path_and_captured_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'ROOT', tmp_path)
    folder=tmp_path/'checkpoints';folder.mkdir();detector=folder/'polypgen_yolov8n.pt'
    detector.write_bytes(b'expected')
    checksum=hashlib.sha256(b'expected').hexdigest()
    assert verify.check_detector(detector,checksum)==detector.resolve()
    duplicate=tmp_path/'copy.pt';duplicate.write_bytes(b'expected')
    with pytest.raises(ValueError,match='detector'):
        verify.check_detector(duplicate,checksum)
    detector.write_bytes(b'changed')
    with pytest.raises(ValueError,match='detector'):
        verify.check_detector(detector,checksum)

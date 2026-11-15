import numpy as np
from PIL import Image
import pytest
from scripts.audit_unarchived_no_detection_scores_v1 import actual_mask


def test_jpeg_empty_and_present_are_decoded(tmp_path):
    x=tmp_path/'empty.jpg';Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(x,quality=100)
    assert not actual_mask(x).any()
    x=tmp_path/'present.jpg';img=np.zeros((32,32),dtype=np.uint8);img[8:24,8:24]=255
    Image.fromarray(img).save(x,quality=100);mask=actual_mask(x)
    assert mask[10,10] and not mask[0,0]


def test_nonjpeg_is_refused(tmp_path):
    x=tmp_path/'mask.png';Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(x)
    with pytest.raises(ValueError,match='unexpected label format'):actual_mask(x)

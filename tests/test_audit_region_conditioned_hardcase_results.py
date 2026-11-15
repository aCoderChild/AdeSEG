import numpy as np
import pytest
from PIL import Image

from scripts.audit_region_conditioned_hardcase_results import compute, decode_label, summarize


def test_independent_dice_and_empty_false_alarm():
    empty=np.zeros((3,4),dtype=bool);target=empty.copy();target[0,0]=True
    assert compute(empty,empty)=={'dice':1.0,'predicted_pixels':0}
    assert compute(target,empty)=={'dice':0.,'predicted_pixels':1}
    assert compute(target,target)=={'dice':1.,'predicted_pixels':1}
    with pytest.raises(ValueError):compute(empty,target[:2])
    rows=[{'split':'validation','present':True,'baseline':compute(target,target)},
          {'split':'validation','present':False,'baseline':compute(target,empty)},
          {'split':'train','present':True,'baseline':compute(empty,target)}]
    assert summarize(rows,'baseline')=={'mean_present_dice':1.,'present_frames':1,
        'empty_false_alarms':1,'empty_frames':1,'mean_empty_predicted_pixels':1.}


def test_label_decoder_uses_binary_jpeg_and_rejects_png(tmp_path):
    image=np.zeros((32,32,3),dtype=np.uint8);image[4:28,4:28]=255
    path=tmp_path/'label.jpg';Image.fromarray(image).save(path,quality=100)
    result=decode_label(path)
    assert result.shape==(32,32) and result[10,10] and not result[0,0]
    png=tmp_path/'label.png';Image.fromarray(image).save(png)
    with pytest.raises(ValueError,match='non-JPEG'):decode_label(png)

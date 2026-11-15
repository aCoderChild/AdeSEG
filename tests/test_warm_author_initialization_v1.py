import json

import numpy as np
import pytest
import torch

from modeling.warm_author_initialization_v1 import archived_warm_components, apply_warm_author_initialization
from scripts.memory_token_io import digest


def test_archived_prompt_is_hash_bound_and_view_relative(tmp_path):
    mask_file=tmp_path/'prompt.png';mask_file.write_bytes(b'placeholder')
    prompt=tmp_path/'prompt.json'
    prompt.write_text(json.dumps({'first_detection_index':0,'box_viewport_xyxy':[1,1,7,4],
                                  'points_viewport':[[2,2],[6,3]]}))
    provenance={'prompt_file':str(mask_file),'prompt_metadata_sha256':digest(prompt),'roi_xyxy':[446,0,454,5]}
    mask=np.zeros((5,8),bool);mask[2,2]=True
    values=archived_warm_components('seq23',0,mask,provenance)
    assert values['box'].tolist()==[1,1,7,4] and values['points'].tolist()==[[2,2],[6,3]]
    assert np.array_equal(values['mask'],mask)
    with pytest.raises(ValueError,match='viewport'):
        archived_warm_components('seq23',0,mask,provenance|{'roi_xyxy':[446,0,455,5]})
    prompt.write_text('{}')
    with pytest.raises(ValueError,match='metadata changed'):
        archived_warm_components('seq23',0,mask,provenance)


def test_rejects_coordinates_outside_view(tmp_path):
    mask_file=tmp_path/'prompt.png';mask_file.write_bytes(b'x');prompt=tmp_path/'prompt.json'
    prompt.write_text(json.dumps({'first_detection_index':3,'box':[0,0,100,4],'points':[[1,1],[2,2]]}))
    provenance={'prompt_file':str(mask_file),'prompt_metadata_sha256':digest(prompt),'roi_xyxy':[0,0,8,5]}
    with pytest.raises(ValueError,match='outside'):
        archived_warm_components('seq2',3,np.ones((5,8),bool),provenance)


def test_warm_order_and_archived_mask_are_unconditional():
    class Model:
        def __init__(self):self.calls=[]
        def add_new_points_or_box(self,state,**kw):
            self.calls.append(kw)
            return kw['frame_idx'],[1],torch.ones((1,1,5,8))
        def add_new_mask(self,state,**kw):
            self.calls.append(kw)
            return kw['frame_idx'],[1],torch.zeros((1,1,5,8))
    mask=np.zeros((5,8),bool);mask[2,2]=True;model=Model()
    result=apply_warm_author_initialization(model,{},3,{'box':np.array([1,1,7,4]),
        'points':np.array([[2,2],[6,3]]),'mask':mask})
    assert len(model.calls)==3 and 'box' in model.calls[0]
    assert model.calls[1]['clear_old_points'] is False and model.calls[1]['labels'].tolist()==[1,0]
    assert np.array_equal(model.calls[2]['mask'],mask) and not result['generated_equals_archived']

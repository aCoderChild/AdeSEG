"""FGPI video uses verified detector selection; no oracle masks for prompts."""
import hashlib
import json
import numpy as np
from PIL import Image
import torch
import pytest

from datasets.video import ManifestFrame
from scripts.infer_fgpi_video import load_detector_records, infer_detector_sequence


class FakeVideo:
    def init_state(self,**kwargs):
        return {"video_height":16,"video_width":16}
    def _get_image_feature(self,state,frame_idx,batch_size):
        f=torch.zeros(1,2,16,16)
        f[:,1]=4
        f[:,0,4:12,4:12]=4
        f[:,1,4:12,4:12]=0
        return None,{"backbone_fpn":[f]},None,None,None
    def add_new_points_or_box(self,state,**kwargs):
        self.received=kwargs
    def propagate_in_video(self,state,start_frame_idx,reverse):
        for i in (0,1,2):
            yield i,[1],torch.ones(1,1,16,16)


def make_data(tmp_path):
    frames=[]
    for idx in range(3):
        image=tmp_path/f"{idx}.jpg"
        mask=tmp_path/f"{idx}.png"
        Image.fromarray(np.ones((16,16,3),np.uint8)*42).save(image)
        Image.fromarray(np.ones((16,16),np.uint8)).save(mask)
        frames.append(ManifestFrame("seq",idx,image,mask))
    detection={"video_id":"seq","frame_index":0,"box_xyxy":[4,4,12,12],
               "confidence":0.9,"image_sha256":hashlib.sha256(frames[0].image_path.read_bytes()).hexdigest()}
    path=tmp_path/"detector.jsonl"
    path.write_text(json.dumps(detection)+"\n")
    return frames,detection,path


def test_detector_manifest_rejects_image_change_and_duplicate(tmp_path):
    frames, record, path=make_data(tmp_path)
    assert load_detector_records(path,{"seq":frames})["seq"][0]["confidence"]==.9
    path.write_text(json.dumps(record)+"\n"+json.dumps(record)+"\n")
    with pytest.raises(ValueError,match="duplicate"):
        load_detector_records(path,{"seq":frames})
    path.write_text(json.dumps(record)+"\n")
    frames[0].image_path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="SHA256"):
        load_detector_records(path,{"seq":frames})


def test_fgpi_and_box_only_have_same_prompt_position_and_bbox(tmp_path):
    frames,record,_=make_data(tmp_path)
    a,b=FakeVideo(),FakeVideo()
    fgpi=tmp_path/"fgpi";box=tmp_path/"box"
    x=infer_detector_sequence(a,frames,[record],fgpi,mode="fgpi")
    y=infer_detector_sequence(b,frames,[record],box,mode="box_only")
    assert x["status"]==y["status"]=="completed"
    assert x["prompt_frame_index"]==y["prompt_frame_index"]==0
    assert x["prompt"]["box_xyxy"]==y["prompt"]["box_xyxy"]
    assert x["prompt"]["decision"]=="augmented"
    assert y["prompt"]["decision"]=="box_only"
    assert a.received["labels"]==[1,0]
    assert b.received["labels"] is None
    assert x["predicted"]==y["predicted"]==2

def test_uniform_early_features_still_use_box_and_points(tmp_path):
    frames,record,_=make_data(tmp_path)
    class UniformVideo(FakeVideo):
        def _get_image_feature(self,state,frame_idx,batch_size):
            return None,{"backbone_fpn":[torch.ones(1,2,16,16)]},None,None,None
    predictor=UniformVideo()
    result=infer_detector_sequence(predictor,frames,[record],tmp_path/"uniform",mode="fgpi")
    assert result["prompt"]["decision"] == "augmented"
    assert predictor.received["box"] == [4.,4.,12.,12.]
    assert predictor.received["labels"] == [1,0]

def test_joint_token_budget_reaches_native_sam2_prompt_interface(tmp_path):
    frames,record,_=make_data(tmp_path)
    predictor=FakeVideo()
    result=infer_detector_sequence(
        predictor,frames,[record],tmp_path/"multi",mode="fgpi",
        positives=2,negatives=2,max_anchors=3)
    assert predictor.received["labels"] == [1,1,0,0]
    assert len(predictor.received["points"]) == 4
    assert len(result["prompt"]["anchor_coords"]) == 3

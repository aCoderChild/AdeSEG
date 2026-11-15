import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace
import scripts.audit_unarchived_no_detection_targets_v1 as a


def test_trace_checks_all_frames_and_rejects_any_detection():
    frames=[{'image_sha256':'a'},{'image_sha256':'b'}]
    rows=[{'frame':0,'image_sha256':'a','box':None,'confidence':None},
          {'frame':1,'image_sha256':'b','box':None,'confidence':None}]
    a.no_detection_trace(rows,frames)
    rows[1]['box']=[0,0,1,1]
    with pytest.raises(ValueError,match='trace mismatch'):a.no_detection_trace(rows,frames)
    with pytest.raises(ValueError,match='coverage'):a.no_detection_trace(rows[:1],frames)


def test_duplicate_screen_rejects_missing_or_duplicate_sequence(tmp_path,monkeypatch):
    data=tmp_path/'data';data.write_text('manifest');monkeypatch.setattr(a,'DATA',data)
    videos={}
    for i,name in enumerate(sorted(a.EXPECTED)):
        p=tmp_path/f'{name}.png';Image.fromarray(np.full((3,4,3),i*7,dtype=np.uint8)).save(p)
        videos[name]=[SimpleNamespace(frame_index=0,image_path=p)]
    train={'seq2','seq3','seq4','seq5','seq12','seq13','seq14','seq9','seq15'}
    frozen={'manifest_sha256':a.sha(data),'sequences':{s:{'frames':1,'image_sha256':[a.sha(videos[s][0].image_path)]} for s in train}}
    rows,counts=a.screen(videos,frozen);assert len(rows)==len(counts)==23
    with pytest.raises(ValueError,match='coverage'):
        a.screen({k:v for k,v in videos.items() if k!='seq23'},frozen)
    videos['seq7'][0].image_path.write_bytes(videos['seq1'][0].image_path.read_bytes())
    with pytest.raises(ValueError,match='duplicates'):
        a.screen(videos,frozen)

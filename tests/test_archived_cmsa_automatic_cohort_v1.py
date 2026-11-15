import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from evaluation.segmentation import calculate_scores
from scripts.memory_token_io import digest
from scripts import score_archived_cmsa_automatic_cohort_v1 as scorer


def test_score_rows_uses_automatic_prompt_and_existing_threshold(tmp_path, monkeypatch):
    batch=tmp_path/'batch';(batch/'scores').mkdir(parents=True)
    cmsa=tmp_path/'cmsa';cmsa.mkdir();prior=tmp_path/'prior.json';prior.write_text('{}')
    frames=[];meta=[];old=[];refs=[];predictions=[];images=[]
    for i in range(3):
        source=tmp_path/f'image{i}.png';target=tmp_path/f'label{i}.png';pred=tmp_path/f'pred{i}.png'
        Image.new('RGB',(4,4)).save(source)
        truth=np.zeros((4,4),np.uint8);truth[0,0]=1;Image.fromarray(truth).save(target)
        pixels=np.zeros((4,4),np.uint8);pixels[0,0]=128;pixels[0,1]=127;Image.fromarray(pixels).save(pred)
        mask=pixels>=128;metrics=calculate_scores(mask,truth==1)
        frames.append(SimpleNamespace(frame_index=i*5,image_path=source,mask_path=target))
        meta.append({'label_sha256':digest(target)});images.append(digest(source))
        old.append({'sequence':'seq2','frame':i,'present':True,'dice':metrics['dice'],'iou':metrics['iou']})
        refs.append({'native':{'present':True}});predictions.append({'path':str(pred),'sha256':digest(pred)})
    metadata=tmp_path/'metadata.json';metadata.write_text(json.dumps(meta))
    (cmsa/'frames.json').write_text(json.dumps(old));(batch/'scores/results.json').write_text(json.dumps({'rows':{'seq2':refs}}))
    monkeypatch.setattr(scorer,'CMSA',cmsa);monkeypatch.setattr(scorer,'PRIOR',prior)
    monkeypatch.setattr(scorer,'SEQUENCES',('seq2',));monkeypatch.setattr(scorer,'PRIORITY',())
    monkeypatch.setattr(scorer,'load_manifest',lambda _,split: {'seq2':frames} if split=='dev' else {})
    bound={'batch':str(batch),'batch_hashes':{'score':digest(batch/'scores/results.json')},
           'prior_sha256':digest(prior),'cmsa_frames_sha256':digest(cmsa/'frames.json'),
           'records':{'seq2':{'prompt_frame':1,'frame_metadata':{'path':str(metadata),'sha256':digest(metadata)},
                              'images':images,'predictions':predictions,'reference_arm':'native'}}}
    rows=scorer.score_rows(bound)['seq2']
    assert [r['post_prompt'] for r in rows]==[False,False,True]
    assert rows[1]['initialization_frame'] and rows[2]['cmsa']['dice']==1.0
    assert rows[2]['cmsa']['predicted_pixels']==1


def test_scoring_does_not_open_labels_if_verification_fails(tmp_path,monkeypatch):
    verification=tmp_path/'verification.json';verification.write_text('{}')
    batch=tmp_path/'batch';batch.mkdir()
    monkeypatch.setattr(scorer,'new_output',lambda p:p)
    monkeypatch.setattr(scorer,'install_phase_mask_guard',lambda _: {'open_attempts':0,'active':True})
    monkeypatch.setattr(scorer,'read_bound',lambda *_:{'passed':True,'label_guard':{'open_attempts':0}})
    monkeypatch.setattr(scorer,'validate_inputs',lambda *_: (_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_:pytest.fail('target opened early'))
    monkeypatch.setattr(scorer.sys,'argv',['score','--mode','score','--batch',str(batch),
        '--verification',str(verification),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='not frozen'):scorer.main()


def test_declared_cohort_and_transitive_source_closure():
    assert len(scorer.SEQUENCES)==20 and 'seq23' not in scorer.SEQUENCES and 'seq7' not in scorer.SEQUENCES
    assert set(scorer.SEQUENCES)==set(scorer.ADDITIONAL)|set(scorer.PRIORITY)
    bound=scorer.source_hashes()
    assert 'evaluation/davis2017/metrics.py' in bound and 'scripts/local_import_closure_v1.py' in bound


def test_phase_frozen_proof_and_score_cannot_be_swapped():
    rows=[{'phase':'verify','artifact_sha256':'verify-old'},
          {'phase':'score','artifact_sha256':'score-old'},
          {'phase':'audit','artifact_sha256':'audit-old'}]
    hashes={'verification':'verify-old','score':'score-old','audit':'audit-old'}
    scorer.require_phase_artifacts(rows,hashes)
    for name in ('verification','score','audit'):
        with pytest.raises(ValueError,match='phase-frozen'):
            scorer.require_phase_artifacts(rows, hashes|{name:'swapped'})

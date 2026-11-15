import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import audit_additional_author_comparators_v3 as auditor
from scripts.memory_token_io import digest
from scripts.score_boundary_video_factorial_v1 import metric,summary


def test_auditor_accepts_per_sequence_metadata_list_and_detects_label_tamper(tmp_path,monkeypatch):
    seq='seq2';methods=('native','samurai','sam2long');batch=tmp_path/'batch';batch.mkdir()
    scores=batch/'scores';scores.mkdir();report=scores/'report.md';report.write_text('report')
    images=[];metadata=[];rows=[];mapping={seq:{}};receipts={seq:{}}
    for i in range(2):
        image=tmp_path/f'image{i}.png';mask=tmp_path/f'label{i}.png'
        Image.new('RGB',(24,24)).save(image)
        truth=np.zeros((24,24),np.uint8);truth[3:8,3:8]=1 if i else 0
        Image.fromarray(truth).save(mask);images.append(SimpleNamespace(frame_index=i,image_path=image,mask_path=mask))
        metadata.append({'label_sha256':digest(mask)})
    meta=tmp_path/'frames.json';meta.write_text(json.dumps(metadata))
    for method in methods:
        folder=batch/method;folder.mkdir();records=[]
        for i,frame in enumerate(images):
            pred=np.zeros((24,24),np.uint8)
            if i:pred[3:8,3:8]=255
            file=folder/f'{i:06d}.png';Image.fromarray(pred).save(file)
            records.append({'frame':i,'mask_sha256':digest(file)})
        receipt=folder/'receipt.json';receipt.write_text(json.dumps({'records':records}))
        receipts[seq][method]={'path':str(receipt),'sha256':digest(receipt)};mapping[seq][method]=str(folder)
    run_map=batch/'run_map.json';run_map.write_text(json.dumps(mapping))
    for i,frame in enumerate(images):
        truth=np.asarray(Image.open(frame.mask_path))==1;pred=truth.copy()
        row={'frame':i,'source_frame_index':i,'post_prompt':i>0,'initialization_frame':i==0}
        for method in methods:row[method]=metric(pred,truth)
        rows.append(row)
    proof={'passed':True,'selection':[seq],'run_map_sha256':digest(run_map),'launch_manifest_sha256':'fake',
           'receipts':receipts,'references':{seq:{}},'inputs':{seq:{'frame_metadata':{'path':str(meta),'sha256':digest(meta)},
           'images':[digest(f.image_path) for f in images]}},'scorer_sha256':'scorer'}
    verify=batch/'verification.json';verify.write_text(json.dumps(proof))
    results={'passed':True,'verification_sha256':digest(verify),'run_map_sha256':digest(run_map),
             'scorer_sha256':'scorer','report_sha256':digest(report),'rows':{seq:rows},
             'summaries':{seq:{m:summary(rows,m) for m in methods}},
             'all_frame_summaries':{seq:{m:summary(rows,m,post_only=False) for m in methods}},
             'paired_descriptive':{seq:{}},
             'preprompt':{seq:{'frames':0,'labeled_present':0,'present_misses':{m:0 for m in methods}}}}
    (scores/'results.json').write_text(json.dumps(results))
    monkeypatch.setattr(auditor,'METHODS',methods)
    monkeypatch.setattr(auditor,'install_phase_mask_guard',lambda _: {'active':True,'open_attempts':0})
    monkeypatch.setattr(auditor,'validate_frozen',lambda *_args,**_kwargs:proof)
    monkeypatch.setattr(auditor,'input_data',lambda *_:(images,0,None,None))
    result=auditor.audit(run_map,verify,scores)
    assert result['passed'] and result['label_files_checked']==2 and result['prediction_files_checked']==6
    Image.new('L',(24,24),1).save(images[1].mask_path)
    with pytest.raises(ValueError,match='changed'):
        auditor.audit(run_map,verify,scores)

#!/usr/bin/env python3
"""Independent no-detection score audit; predictions/gates precede label reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.audit_unarchived_no_detection_targets_v1 import audit as audit_targets

DATA=ROOT/'data/polypgen_sequence.jsonl';ARMS=('A','B','C','D');SEQUENCES=('seq1','seq7')
BASE=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment').resolve()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def protect_labels():
    blocked=set()
    for line in DATA.read_text().splitlines():
        if line.strip():
            path=Path(json.loads(line)['mask']);blocked.add(os.path.realpath(path if path.is_absolute() else DATA.parent/path))
    state={'blocked_paths':len(blocked),'precheck_attempts':0,'allow_score':False}
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and not state['allow_score'] and os.path.realpath(os.fsdecode(args[0])) in blocked:
            state['precheck_attempts']+=1;raise PermissionError('label read before complete evidence check')
    sys.addaudithook(audit);return state


def actual_mask(path):
    """Independently decode JPEG using OpenCV and check project's JPEG semantics."""
    require(Path(path).suffix.lower() in ('.jpg','.jpeg'),'unexpected label format')
    decoded=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    require(decoded is not None and decoded.ndim==2,'label decode failed')
    with Image.open(path) as image:
        raw=np.asarray(image)
        if raw.ndim==3 or (raw.max()>1 and image.format=='JPEG'):
            gray=np.asarray(image.convert('L'))
            require(((gray>64)&(gray<192)).mean()<=0.05,'ambiguous grayscale annotation')
            expected=gray>127
            actual=decoded>127
        else:
            require(set(np.unique(raw).tolist()).issubset({0,1}),'unexpected indexed labels')
            expected=raw==1
            actual=decoded>0
    require(np.array_equal(actual,expected),'independent decoder mask differs')
    return actual


def run(score_dir,target_a,target_b,gate,reference_audit,target_audit,guard):
    score_dir=Path(score_dir).resolve(strict=True);target_a=Path(target_a).resolve(strict=True);target_b=Path(target_b).resolve(strict=True)
    gate=Path(gate).resolve(strict=True);reference_audit=Path(reference_audit).resolve(strict=True);target_audit=Path(target_audit).resolve(strict=True)
    require(all(p.is_relative_to(BASE) for p in (score_dir,target_a,target_b,gate,reference_audit,target_audit)),'artifact outside experiment folder')
    done=read(score_dir/'completion.json'); score=read(score_dir/'results.json'); frozen_audit=read(target_audit)
    require(done.get('passed') is True and done.get('results_sha256')==sha(score_dir/'results.json') and
            done.get('report_sha256')==sha(score_dir/'report.md') and score.get('passed') is True,'score/report receipt differs')
    require(score.get('scorer_sha256')==sha(ROOT/'scripts/score_unarchived_no_detection_v1.py') and
            score.get('generator_sha256')==sha(ROOT/'scripts/generate_unarchived_auto_prompt_v1.py') and
            score.get('data_manifest_sha256')==sha(DATA) and score.get('target_audit_sha256')==sha(target_audit) and
            score.get('target_audit_source_sha256')==sha(ROOT/'scripts/audit_unarchived_no_detection_targets_v1.py') and
            score.get('gate_sha256')==sha(gate/'gate.json') and score.get('reference_audit_sha256')==sha(reference_audit) and
            score.get('verification_guard',{}).get('verification_open_attempts')==0,'score source/gate/guard differs')
    require(frozen_audit.get('passed') is True and frozen_audit.get('audit_source_sha256')==sha(ROOT/'scripts/audit_unarchived_no_detection_targets_v1.py') and
            frozen_audit.get('target_roots')==[str(target_a),str(target_b)] and frozen_audit.get('label_guard',{}).get('open_attempts')==0,
            'saved target audit differs')
    # Recheck all source/image/duplicate inputs and both sets of frozen zero
    # masks BEFORE enabling label reads. This calls no model or detector.
    fresh=audit_targets([target_a,target_b],gate,reference_audit)
    expected={k:v for k,v in frozen_audit.items() if k!='label_guard'}
    require(fresh==expected and fresh['prediction_masks_checked']==672 and fresh['input_frames_checked']==84,'independent target audit replay differs')
    require(guard['precheck_attempts']==0,'label access attempted before evidence validation')
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')};guard['allow_score']=True
    checked=0;results={}
    for seq in SEQUENCES:
        frames=videos[seq];rows=score['rows'][seq];require(len(frames)==len(rows)>0,'frame coverage differs')
        true_present=0;false_alarms={a:0 for a in ARMS}
        for i,(f,row) in enumerate(zip(frames,rows,strict=True)):
            require(f.frame_index==i==row['frame']==row['source_frame_index'] and sha(f.image_path)==row['image_sha256'] and sha(f.mask_path)==row['label_sha256'],'frame/label hash mismatch')
            truth=actual_mask(f.mask_path);require(sha(f.mask_path)==row['label_sha256'],'label changed during decode')
            with Image.open(f.image_path) as im:require(truth.shape==(im.height,im.width),'label geometry differs')
            present=bool(truth.any()); true_present+=present
            require(row['present']==present and row['predicted_pixels']==0 and row['dice_on_present']==(0.0 if present else None) and
                    row['recall_on_present']==(0.0 if present else None) and row['empty_false_alarm']==(None if present else False),'per-frame no-detection metric differs')
            for arm in ARMS:
                path=target_a/seq/arm/f'{i:06d}.png'; require(sha(path)==row['prediction_sha256'][arm],'prediction SHA differs')
                img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);require(img is not None and img.shape==truth.shape and not img.any(),'prediction is not empty')
                if not present:false_alarms[arm]+=bool(img.any())
                checked+=1
        empty=len(frames)-true_present;summary=score['summaries'][seq]
        expected_summary={'frame_count':len(frames),'detector_calls':len(frames),'initialization':'no_detection','prompt_frame':None,
            'post_prompt_frame_count':0,'post_prompt_present_dice':None,'all_frame_present':true_present,'all_frame_empty':empty,
            'all_frame_present_dice':0.0 if true_present else None,'all_frame_present_recall':0.0 if true_present else None,
            'pre_initialization_present_missed':true_present,'zero_overlap_present':true_present,'labeled_empty_false_alarms':0,
            'labeled_empty_denominator':empty,'empty_predicted_pixels_mean':0.0 if empty else None}
        require(all(summary.get(k)==v for k,v in expected_summary.items()) and not any(false_alarms.values()),'summary mismatch')
        results[seq]=expected_summary
    return {'passed':True,'scope':'Independent CPU OpenCV label/mask rescore after rechecking all target artifacts; no detector/model calls.',
        'audit_source_sha256':sha(Path(__file__)),'scores_sha256':sha(score_dir/'results.json'),'report_sha256':sha(score_dir/'report.md'),
        'target_audit_sha256':sha(target_audit),'predictions_rescored':checked,'frozen_masks_rechecked':fresh['prediction_masks_checked'],
        'labels_checked':sum(len(videos[s]) for s in SEQUENCES),'precheck_label_attempts':guard['precheck_attempts'],'summaries':results,
        'limitations':['No object initialized: these scores do not evaluate video refinement.',
                       'Labels first hash-bound after inference; empty annotation does not establish clinical absence.',
                       'CPython open guard does not intercept native-code I/O; model/detector were not called.']}


def main():
    p=argparse.ArgumentParser()
    for k in ('score','target-a','target-b','gate','reference-audit','target-audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); out=validate_output(a.output);require(out.is_relative_to(BASE) and not out.exists(),'invalid audit output')
    guard=protect_labels();result=run(a.score,a.target_a,a.target_b,a.gate,a.reference_audit,a.target_audit,guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'labels':result['labels_checked'],'predictions':result['predictions_rescored']}))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent saved-artifact audit of two no-detection repeats; no labels or model calls."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import PIL
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard

DATA=ROOT/'data/polypgen_sequence.jsonl'
BASE=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment').resolve()
EXPECTED={f'seq{i}' for i in range(1,24)}
TARGET=('seq1','seq7');ARMS=('A','B','C','D')
CHECKPOINT=BASE/'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
DETECTOR=ROOT/'checkpoints/polypgen_yolov8n.pt'
TRAIN=BASE/'contrast_refinement_v1/source_group_capture_preflight_attempt03'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def no_detection_trace(attempts,frames):
    require(len(attempts)==len(frames),'detector coverage mismatch')
    for i,(attempt,frame) in enumerate(zip(attempts,frames,strict=True)):
        require(attempt.get('frame')==i and attempt.get('image_sha256')==frame['image_sha256'] and
                attempt.get('box') is None and attempt.get('confidence') is None, 'no-detection trace mismatch')


def screen(videos,training):
    require(set(videos)==EXPECTED,'full source sequence coverage mismatch')
    require(training['manifest_sha256']==sha(DATA),'training manifest mismatch')
    train={'seq2','seq3','seq4','seq5','seq12','seq13','seq14','seq9','seq15'}
    raw={};decoded={};rows={};counts={}
    for sequence,frames in videos.items():
        counts[sequence]=len(frames);require(bool(frames),'empty source sequence')
        old=training['sequences'].get(sequence) if sequence in train else None
        if sequence in train:require(old is not None and old['frames']==len(frames) and len(old['image_sha256'])==len(frames),'frozen training counts mismatch')
        for i,f in enumerate(frames):
            require(f.frame_index==i,'source ordinal mismatch');data_sha=sha(f.image_path)
            if old is not None:require(data_sha==old['image_sha256'][i],'frozen training image differs')
            with Image.open(f.image_path) as image:
                rgb=image.convert('RGB'); pix_sha=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
            require(sha(f.image_path)==data_sha,'source changed during decoding')
            rows[(sequence,i)]={'sequence':sequence,'frame':i,'image_sha256':data_sha,'decoded_rgb_sha256':pix_sha}
            for owners,key in ((raw,data_sha),(decoded,pix_sha)):
                prior=owners.get(key,set())
                require(not any(x!=sequence and (x in TARGET or sequence in TARGET) for x in prior),'target image duplicates another sequence')
                owners.setdefault(key,set()).add(sequence)
    return rows,counts


def audit(roots,gate_folder,reference_audit):
    roots=[Path(p).resolve(strict=True) for p in roots]; gate_folder=Path(gate_folder).resolve(strict=True);reference_audit=Path(reference_audit).resolve(strict=True)
    require(len(roots)==2 and roots[0]!=roots[1] and all(p.is_relative_to(BASE) for p in [*roots,gate_folder,reference_audit]),'invalid replay/gate paths')
    gate=read(gate_folder/'gate.json');done=read(gate_folder/'completion.json');ind=read(reference_audit)
    require(gate.get('passed') is True and done.get('passed') is True and done.get('gate_sha256')==sha(gate_folder/'gate.json') and
        ind.get('passed') is True and ind.get('gate_sha256')==sha(gate_folder/'gate.json') and
        ind.get('audit_source_sha256')==sha(ROOT/'scripts/audit_unarchived_prompt_gate_v1.py') and
        ind.get('label_guard',{}).get('open_attempts')==0 and ind.get('reference_runs_checked')==2,
        'reference gate or independent audit mismatch')
    for ref,key,entry in zip((gate['reference_a'],gate['reference_b']),('reference_a_sha256','reference_b_sha256'),ind['runs'],strict=True):
        require(sha(Path(ref)/'completion.json')==gate[key]==entry['completion_sha256'],'reference artifact changed')
    signature=gate['signature']; author=signature['author']
    require(signature['manifest_sha256']==sha(DATA) and signature['checkpoint_sha256']==sha(CHECKPOINT) and
        signature['detector_sha256']==sha(DETECTOR) and signature['pythonhashseed']=='0' and signature['torch_num_threads']==2 and
        author['python']==sys.version and author['torch']==torch.__version__ and author['numpy']==np.__version__ and
        author['ultralytics']==version('ultralytics') and signature['pillow']==PIL.__version__ and signature['opencv']==cv2.__version__,
        'current recorded environment/checkpoint differs')
    for name,value in signature['source_sha256'].items():
        path=(ROOT/name).resolve(strict=True);require(path.is_relative_to(ROOT) and sha(path)==value,'generator source changed')
    training=read(TRAIN/'manifest.json');train_done=read(TRAIN/'completion.json')
    require(train_done.get('passed') is True and train_done.get('manifest_sha256')==sha(TRAIN/'manifest.json'),'training preflight changed')
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')};screen_rows,counts=screen(videos,training)
    evidence={};all_ids=[]; mask_count=0; attempts_total=0
    for sequence in TARGET:
        evidence[sequence]=[];items=[]
        for root in roots:
            folder=(root/sequence).resolve(strict=True); require(folder.is_relative_to(root),'target folder outside replay')
            rec=read(folder/'completion.json');prompt=read(folder/'prompt.json');frames=read(folder/'frames.json');attempts=read(folder/'detections.json');saved_screen=read(folder/'duplicate_screen.json')
            require(rec.get('passed') is True and rec.get('mode')=='target' and rec.get('sequence')==sequence and rec.get('signature')==signature and
                rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('label_guard',{}).get('blocked_paths',0)>0 and
                rec.get('reference_gate_sha256')==sha(gate_folder/'gate.json') and rec.get('independent_audit_sha256')==sha(reference_audit), 'target identity or label guard mismatch')
            require(type(rec.get('pid')) is int and rec.get('run_uuid') and (rec['pid'],rec['run_uuid']) not in all_ids,'target process identity mismatch');all_ids.append((rec['pid'],rec['run_uuid']))
            for field,name in (('prompt_sha256','prompt.json'),('frames_sha256','frames.json'),('detections_sha256','detections.json'),('duplicate_screen_sha256','duplicate_screen.json')):
                require(rec[field]==sha(folder/name),'target artifact hash mismatch')
            require(saved_screen.get('passed') is True and saved_screen.get('target_sequences')==sorted(TARGET) and
                saved_screen.get('counts')==counts and saved_screen.get('training_preflight_sha256')==sha(TRAIN/'manifest.json') and
                saved_screen.get('target_cross_sequence_exact_or_decoded_matches')==0, 'duplicate screen metadata mismatch')
            saved_rows=saved_screen['rows'];require(len(saved_rows)==len(screen_rows),'duplicate screen row count mismatch')
            seen=set()
            for row in saved_rows:
                key=(row['sequence'],row['frame']);require(key not in seen and screen_rows.get(key)==row,'duplicate screen input mismatch');seen.add(key)
            expected=videos[sequence];require(len(frames)==len(expected) and len(frames)>0,'target frame count mismatch')
            for i,(row,f) in enumerate(zip(frames,expected,strict=True)):
                require(row=={'frame':i,'source_frame_index':f.frame_index,'image_path':str(f.image_path.resolve()),'image_sha256':sha(f.image_path)},'target frame identity mismatch')
            require(prompt.get('status')=='no_detection' and prompt.get('first_detection_index') is None and prompt.get('box') is None and
                prompt.get('confidence') is None and prompt.get('points') is None and prompt.get('point_status')=='no_detection' and
                prompt.get('lean_mask_status')=='no_detection' and prompt.get('post_prompt_frames')==0,'prompt is not no-detection')
            no_detection_trace(attempts,frames);attempts_total+=len(attempts)
            require(set(prompt['empty_arm_masks'])==set(ARMS),'arm coverage mismatch')
            for arm in ARMS:
                masks=prompt['empty_arm_masks'][arm];require([r['frame'] for r in masks]==list(range(len(frames))),'mask coverage mismatch')
                for row,frame in zip(masks,expected,strict=True):
                    path=folder/arm/f"{row['frame']:06d}.png";require(sha(path)==row['sha256'],'empty mask SHA mismatch')
                    with Image.open(frame.image_path) as source: size=source.size
                    with Image.open(path) as image:require(image.format=='PNG' and image.size==size and not np.asarray(image.convert('L')).any(),'mask is not image-sized empty PNG')
                    mask_count+=1
            evidence[sequence].append({'folder':str(folder),'completion_sha256':sha(folder/'completion.json'),
                'pid':rec['pid'],'run_uuid':rec['run_uuid'],'prompt_sha256':rec['prompt_sha256'],
                'frames_sha256':rec['frames_sha256'],'detections_sha256':rec['detections_sha256'],
                'duplicate_screen_sha256':rec['duplicate_screen_sha256']})
            items.append((prompt,frames,attempts))
        require(items[0]==items[1],f'{sequence}: fresh replay differs')
    return {'passed':True,'scope':'Independent no-detection artifact check; no detector/model calls or label mask reads.',
      'audit_source_sha256':sha(Path(__file__)),'gate_sha256':sha(gate_folder/'gate.json'),'reference_audit_sha256':sha(reference_audit),
      'target_roots':[str(x) for x in roots],'completion_sha256':evidence,'input_frames_checked':sum(len(videos[s]) for s in TARGET),
      'detector_attempts_checked':attempts_total,'prediction_masks_checked':mask_count,'screen_frames_checked':sum(counts.values()),
      'duplicate_screen_sha256':{s:[e['duplicate_screen_sha256'] for e in evidence[s]] for s in TARGET},
      'limitations':['No-detection is not clinical absence.','Python open guard does not cover arbitrary native I/O.',
        'Exact and decoded-pixel checks do not rule out near duplicates or prior model exposure.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--target-a',type=Path,required=True);p.add_argument('--target-b',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--reference-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    guard=install_mask_open_guard(DATA);outcome=audit([a.target_a,a.target_b],a.gate,a.reference_audit);require(guard['open_attempts']==0,'label-mask open attempted');outcome['label_guard']=guard
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(outcome,indent=2)+'\n');print(json.dumps({'passed':True,'inputs':outcome['input_frames_checked'],'attempts':outcome['detector_attempts_checked'],'masks':outcome['prediction_masks_checked'],'output':str(out)}))

if __name__=='__main__':main()

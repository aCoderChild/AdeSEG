#!/usr/bin/env python3
"""Independently inspect saved seq6 prompt replays; no detector or label reads."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import PIL
import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard

DATA = ROOT / 'data/polypgen_sequence.jsonl'
ARCHIVE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/lean_auto_multiscale_v1/cohort_part_a_attempt01/seq6')
CHECKPOINT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt')
DETECTOR = ROOT / 'checkpoints/polypgen_yolov8n.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, text):
    if not ok:
        raise ValueError(text)


def verify_trace(attempts, frames, prompt, size):
    first = prompt.get('first_detection_index')
    require(type(first) is int and 0 <= first < len(frames), 'invalid first detection')
    require(len(attempts) == first + 1, 'detector-call count differs')
    for i, row in enumerate(attempts):
        require(row['frame'] == i and row['image_sha256'] == frames[i]['image_sha256'], 'detector input/order differs')
        if i < first:
            require(row['box'] is None and row['confidence'] is None, 'earlier detection was skipped')
    last = attempts[-1]
    box = last['box']; conf = last['confidence']
    require(isinstance(box, list) and len(box) == 4 and isinstance(conf, (int, float)) and
            all(math.isfinite(float(x)) for x in [*box, conf]), 'invalid terminal detection')
    require(0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1] and
            box == prompt['box'] and conf == prompt['confidence'], 'terminal prompt differs')
    return first


def image_metadata(path):
    indices=[]; hashes=[]
    index=re.compile(rb'^\s*"source_frame_index"\s*:\s*(\d+)\s*,?\s*$')
    image=re.compile(rb'^\s*"image_sha256"\s*:\s*"([a-f0-9]{64})"\s*,?\s*$')
    with Path(path).open('rb') as stream:
        for line in stream:
            if b'"source_frame_index"' in line:
                m=index.fullmatch(line.rstrip(b'\r\n')); require(m is not None, 'unexpected archived index syntax');indices.append(int(m.group(1)))
            elif b'"image_sha256"' in line:
                m=image.fullmatch(line.rstrip(b'\r\n')); require(m is not None, 'unexpected archived image syntax');hashes.append(m.group(1).decode())
    require(len(indices)==len(hashes) and len(indices)>0, 'archived image metadata incomplete')
    return indices,hashes


def audit(root):
    root=Path(root).resolve(strict=True); gate_file=root/'gate/gate.json'; done=read(root/'gate/completion.json');gate=read(gate_file)
    require(done.get('passed') is True and done.get('gate_sha256') == sha(gate_file) and gate.get('passed') is True, 'invalid gate')
    a,b=(Path(gate[k]).resolve(strict=True) for k in ('reference_a','reference_b'))
    require(a!=b and a.is_relative_to(root) and b.is_relative_to(root), 'reference paths not distinct/in scope')
    require(sha(a/'completion.json')==gate['reference_a_sha256'] and sha(b/'completion.json')==gate['reference_b_sha256'], 'reference completion mismatch')
    recs=[read(x/'completion.json') for x in (a,b)]
    require(all(x.get('passed') is True and x.get('mode')=='reference' and x.get('sequence')=='seq6' and x.get('signature')==gate['signature'] and x.get('label_guard',{}).get('open_attempts')==0 for x in recs), 'reference identity/guard differs')
    require(all(type(x.get('pid')) is int and x.get('run_uuid') for x in recs) and recs[0]['pid']!=recs[1]['pid'] and recs[0]['run_uuid']!=recs[1]['run_uuid'], 'reference processes not distinct')
    sig=gate['signature'];author=sig['author']
    archived_signature=read(ARCHIVE/'setup.json')['signature']
    require(all(archived_signature.get(k)==v for k,v in author.items()), 'archived environment/signature differs')
    require(author.get('python')==sys.version and author.get('torch')==torch.__version__ and
            author.get('numpy')==np.__version__ and author.get('ultralytics')==version('ultralytics') and
            sig.get('pillow')==PIL.__version__ and sig.get('opencv')==cv2.__version__, 'current runtime version differs')
    require(sig['checkpoint_sha256']==sha(CHECKPOINT) and sig['detector_sha256']==sha(DETECTOR) and sig['manifest_sha256']==sha(DATA) and sig['pythonhashseed']=='0' and sig['torch_num_threads']==2, 'frozen model/input/env mismatch')
    for name,value in sig['source_sha256'].items():
        file=(ROOT/name).resolve(strict=True); require(file.is_relative_to(ROOT) and sha(file)==value, 'source hash changed')
    repo=ROOT/'external/Lean-SAM2'; revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True)
    require(revision==author['revision'] and not dirty,'author fork changed')
    for key,file in (('archived_setup_sha256','setup.json'),('archived_records_sha256','records.json'),('archived_completion_sha256','completion.json'),('archived_prompt_sha256','seq6/prompt.json'),('archived_frames_sha256','seq6/frames.json'),('archived_mask_sha256','seq6/prompt_refined.png')):
        require(sig[key]==sha(ARCHIVE/file),'archive changed')
    frames=load_manifest(DATA,'dev')['seq6']; indices,hashes=image_metadata(ARCHIVE/'seq6/frames.json'); require(len(frames)==len(indices)==len(hashes),'source count differs')
    old=read(ARCHIVE/'seq6/prompt.json'); runs=[]
    for folder,rec in zip((a,b),recs,strict=True):
        for k,name in (('prompt_sha256','prompt.json'),('frames_sha256','frames.json'),('detections_sha256','detections.json')):require(rec[k]==sha(folder/name),'artifact hash differs')
        prompt=read(folder/'prompt.json'); inventory=read(folder/'frames.json'); attempts=read(folder/'detections.json'); require(len(inventory)==len(frames),'saved input count differs')
        for i,(f,row) in enumerate(zip(frames,inventory,strict=True)):
            require(f.frame_index==i==indices[i]==row['frame']==row['source_frame_index'] and str(f.image_path.resolve())==row['image_path'] and sha(f.image_path)==hashes[i]==row['image_sha256'],'source image differs')
        with Image.open(frames[0].image_path) as im: size=im.size
        first=verify_trace(attempts,inventory,prompt,size)
        for k in ('first_detection_index','box','confidence','point_status','points','lean_mask_status','feature_shape','early_feature_sha256','provisional_mask_sha256','refined_mask_sha256'):
            require(prompt.get(k)==old.get(k),f'archived prompt field differs: {k}')
        require(prompt.get('status')=='detected_nonempty' and prompt.get('post_prompt_frames')==len(frames)-first-1,'reference status differs')
        points=prompt['points'];require(len(points)==2 and all(0<=x<size[0] and 0<=y<size[1] for x,y in points),'point geometry differs')
        for filename,key in (('prompt_provisional.png','provisional_mask_sha256'),('prompt_refined.png','refined_mask_sha256')):
            file=folder/filename; ref=ARCHIVE/'seq6'/filename; require(sha(file)==prompt[key]==sha(ref),'prompt PNG differs')
            with Image.open(file) as actual, Image.open(ref) as archived:
                require(actual.size==archived.size==size and np.array_equal(np.asarray(actual.convert('L')),np.asarray(archived.convert('L'))),'prompt pixels differ')
        require(sha(folder/'model_state.json')==prompt['model_state_sha256'] and len(read(folder/'model_state.json'))>0,'state fingerprint differs')
        runs.append({'completion_sha256':sha(folder/'completion.json'),'pid':rec['pid'],'run_uuid':rec['run_uuid'],
                     'frames_sha256':sha(folder/'frames.json'),'detections_sha256':sha(folder/'detections.json'),
                     'model_state_sha256':sha(folder/'model_state.json'),'prompt_sha256':sha(folder/'prompt.json')})
    for key in ('prompt_sha256','frames_sha256','detections_sha256','model_state_sha256'):
        require(runs[0][key]==runs[1][key],f'reference replays differ: {key}')
    return {'passed':True,'scope':'Independent saved-artifact audit; no detector, model inference or target-label opens. Not a prospective efficacy test.',
            'gate_sha256':sha(gate_file),'audit_source_sha256':sha(Path(__file__)),'input_frames_checked':len(frames),
            'detector_attempts_checked':first+1,'reference_runs_checked':len(runs),'runs':runs,
            'limitations':['Historical detector-helper, Pillow and OpenCV versions were not recorded in archived receipt.',
                           'Distinct PIDs/UUIDs and source artifacts support but do not by themselves prove execution independence.',
                           'CPython guard cannot intercept arbitrary native-code I/O.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    guard=install_mask_open_guard(DATA); result=audit(a.run);require(guard['open_attempts']==0,'label-mask open attempted');result['label_guard']=guard
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'passed':True,'output':str(out),'inputs':result['input_frames_checked'],'attempts':result['detector_attempts_checked']}))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independently score frozen automatic SAMURAI and project-Lean masks.

Both systems receive the same archived predicted-mask initialization. The
SAMURAI run starts at local frame zero; original pre-prompt frames are scored
as empty predictions, matching the automatic Lean protocol.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import MANIFEST


def read(path): return json.loads(path.read_text())


def score(pred, truth):
    if pred.shape != truth.shape: raise ValueError('Prediction/label geometry differs')
    present=bool(truth.any()); area=int(pred.sum()); denominator=area+int(truth.sum())
    return {'present':present,'dice':2*int((pred & truth).sum())/denominator if denominator else 1.,
            'false_alarm':bool(area) if not present else None,'predicted_pixels':area}


def summary(rows, arm, *, post_only):
    chosen=[r for r in rows if not post_only or r['post_prompt']]
    present=[r[arm]['dice'] for r in chosen if r[arm]['present']]
    empty=[r[arm] for r in chosen if not r[arm]['present']]
    return {'frames':len(chosen),'present_frames':len(present),
            'mean_present_frame_dice':float(np.mean(present)) if present else None,
            'empty_frames':len(empty),'false_alarm_frames':sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels':float(np.mean([x['predicted_pixels'] for x in empty])) if empty else None}


def verify_frozen(sequence, samurai, lean, frames):
    receipt=read(samurai/'receipt.json')
    if not receipt.get('passed') or not receipt.get('full_video') or receipt['sequence']!=sequence:
        raise ValueError('SAMURAI run is incomplete or wrong sequence')
    if receipt['runner_sha256'] != digest(ROOT/'scripts/run_samurai_automatic_priority.py'):
        raise ValueError('SAMURAI runner changed')
    start=receipt['prompt_frame']; records=receipt['records']
    if len(records)!=len(frames)-start or receipt['frames']!=len(records): raise ValueError('SAMURAI coverage differs')
    for local, row in enumerate(records):
        idx=start+local
        if row['local_index']!=local or row['frame']!=idx or row['source_frame_index']!=frames[idx].frame_index or row['image_sha256']!=digest(frames[idx].image_path) or row['mask_sha256']!=digest(samurai/f'{idx:06d}.png'):
            raise ValueError('SAMURAI frame, source or saved prediction differs')
    done=read(lean/'completion.json')
    if not done.get('passed'):raise ValueError('Lean reference incomplete')
    if sequence=='seq23':
        if digest(lean/'frames.json')!=done['frames_sha256'] or digest(lean/'prompt.json')!=done['prompt_sha256']:
            raise ValueError('Cropped Lean receipt differs')
        archived=read(lean/'frames.json')
        if len(archived)!=len(frames):raise ValueError('Cropped Lean coverage differs')
        mask_hashes=[row['mask_sha256'] for row in archived]
        image_hashes=[row['image_sha256'] for row in archived]
        label_hashes=[row['label_sha256'] for row in archived]
        if digest(lean/'prompt.png') != receipt['prompt']['prompt_mask_sha256']:
            raise ValueError('Cropped prompt differs')
    else:
        if digest(lean/'records.json')!=done['records_sha256']:raise ValueError('Lean reference receipt differs')
        record=read(lean/'records.json')
        if record['sequence']!=sequence or record['frames']!=len(frames) or digest(lean/'frames.json')!=record['frames_sha256']:
            raise ValueError('Lean reference geometry or scoring changed')
        archived=read(lean/'frames.json');mask_hashes=record['mask_sha256']
        if len(archived)!=len(frames) or len(mask_hashes)!=len(frames):raise ValueError('Lean reference coverage differs')
        # Original archived automatic prompt metadata binds source and label hashes.
        prompt_meta=Path(receipt['prompt']['prompt_file']).parent
        original=read(prompt_meta/'frames.json')
        image_hashes=[row['image_sha256'] for row in original];label_hashes=[row['label_sha256'] for row in original]
    for i, frame in enumerate(frames):
        if digest(frame.image_path)!=image_hashes[i] or digest(lean/f'{i:06d}.png')!=mask_hashes[i]:
            raise ValueError('Lean reference input or prediction differs')
    return start,label_hashes,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=('seq11','seq20','seq23'),required=True)
    for name in ('samurai','lean','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};frames=videos[a.sequence]
    # No ground-truth pixels are read until every prediction and input hash passes.
    start,label_hashes,receipt=verify_frozen(a.sequence,a.samurai,a.lean,frames)
    rows=[]
    for i,frame in enumerate(frames):
        if digest(frame.mask_path)!=label_hashes[i]:raise ValueError('Evaluation label changed')
        truth=load_label_mask(frame.mask_path,[1])==1
        lp=np.asarray(Image.open(a.lean/f'{i:06d}.png').convert('L'))>0
        sp=(np.asarray(Image.open(a.samurai/f'{i:06d}.png').convert('L'))>0) if i>=start else np.zeros_like(truth)
        ls,ss=score(lp,truth),score(sp,truth)
        if ls['present']!=ss['present']:raise ValueError('Presence mismatch')
        rows.append({'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,
                     'initialization_frame':i==start,'lean':ls,'samurai':ss})
    result={'scope':'Exploratory same archived automatic-mask initialization; no new detector calls. Not independent validation.',
      'sequence':a.sequence,'prompt_frame':start,'frames':len(rows),
      'pre_prompt_present_misses':sum(r['lean']['present'] for r in rows if r['frame']<start),
      'samurai_receipt_sha256':digest(a.samurai/'receipt.json'),'lean_completion_sha256':digest(a.lean/'completion.json'),
      'scorer_sha256':digest(Path(__file__)),'summaries':{arm:{'all':summary(rows,arm,post_only=False),
      'post_prompt':summary(rows,arm,post_only=True)} for arm in ('lean','samurai')},'rows':rows,
      'limitations':['Historical prompt uses YOLO scan until first detection; no model is applied before initialization.',
                     'SAMURAI and Lean are distinct systems and checkpoints; no single memory mechanism is isolated.',
                     'SAMURAI includes the author memory-selection thresholds; no new project gate or redetection is added.',
                     'Sequence-specific crop is retrospective; patient independence and detector overlap are unknown.',
                     'MPS and archived CPU Lean run on different devices; short CPU/MPS tests do not ensure full-trajectory equivalence.']}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'prompt_frame':start,'summaries':result['summaries']}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independently score frozen same-device automatic-prompt Lean/SAMURAI.

Labels are read only after every source image and both full prediction sets
have passed provenance and checksum checks. Pre-prompt frames are scored as
empty predictions. This is an exploratory development comparison.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.run_samurai_automatic_priority import MANIFEST,prepare_archived_prompt
from scripts.score_samurai_automatic_priority import score,summary


def read(path):return json.loads(path.read_text())


def check_frozen(sequence,frames,lean,samurai,lean_gate,samurai_gate,prompt):
    l=read(lean/'receipt.json');s=read(samurai/'receipt.json')
    if not l.get('passed') or not s.get('passed') or not l.get('full_video') or not s.get('full_video'):
        raise ValueError('Incomplete predictions')
    if l['sequence']!=sequence or s['sequence']!=sequence or l['prompt_frame']!=s['prompt_frame'] or l['prompt']!=prompt or s['prompt']!=prompt:
        raise ValueError('Initialization differs')
    if (l['identity']['device']!='mps' or s['device']!='mps' or l['identity']['fallback']!='0' or s['fallback']!='0'
        or l['identity']['fast_math']!='0' or s['fast_math']!='0' or l['identity']['torch']!=s['torch']
        or l['identity']['checkpoint_sha256']!=s['checkpoint_sha256']):
        raise ValueError('Device, checkpoint or backend settings differ')
    if (l['identity']['runner_sha256']!=digest(ROOT/'scripts/run_lean_window_baseline_mps.py')
        or s['runner_sha256']!=digest(ROOT/'scripts/run_samurai_automatic_priority.py')
        or l['gate_sha256']!=digest(lean_gate) or s['gate_sha256']!=digest(samurai_gate)):
        raise ValueError('Runner or engineering gate changed')
    lg=read(lean_gate);sg=read(samurai_gate)
    if (not lg.get('passed') or not sg.get('passed') or lg['sequence']!=sequence or sg['sequence']!=sequence
        or lg['prompt']!=prompt or sg['prompt']!=prompt or lg['identity']!=l['identity']
        or lg['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_window_baseline_mps_gate.py')
        or sg['verifier_sha256']!=digest(ROOT/'scripts/verify_samurai_automatic_gate.py')
        or s['kalman_adapter_sha256']!=digest(ROOT/'modeling/samurai_mps_kalman.py')):
        raise ValueError('Engineering gate identity differs')
    if l['state_sha256']!=lg['state_sha256'] or s['state_sha256']!=sg['state_sha256']:
        raise ValueError('Live model state differs from engineering gate')
    start=l['prompt_frame'];lr=l['records'];sr=s['records']
    if l['end_frame']!=len(frames)-1 or len(lr)!=len(frames) or s['frames']!=len(frames)-start or len(sr)!=s['frames']:
        raise ValueError('Full-video coverage differs')
    for i,frame in enumerate(frames):
        row=lr[i]
        if row['frame']!=i or row['source_frame_index']!=frame.frame_index or row['image_sha256']!=digest(frame.image_path) or row['mask_sha256']!=digest(lean/f'{i:06d}.png'):
            raise ValueError('Lean image, index or mask changed')
        if i>=start:
            item=sr[i-start]
            if item['frame']!=i or item['local_index']!=i-start or item['source_frame_index']!=frame.frame_index or item['image_sha256']!=row['image_sha256'] or item['mask_sha256']!=digest(samurai/f'{i:06d}.png'):
                raise ValueError('SAMURAI image, rebase or mask changed')
        else:
            with Image.open(lean/f'{i:06d}.png') as image:
                if np.asarray(image.convert('L')).any():raise ValueError('Nonempty pre-prompt Lean output')
    return start,l,s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=('seq11','seq20','seq23'),required=True)
    for k in ('lean','samurai','lean-gate','samurai-gate','preflight','cropped-reference','cropped-backup','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};frames=videos[a.sequence]
    _,_,_,prompt=prepare_archived_prompt(a.sequence,frames,a.preflight,a.cropped_reference,a.cropped_backup)
    # Do not open any label pixels before both full mask sets are frozen and checked.
    start,l,s=check_frozen(a.sequence,frames,a.lean,a.samurai,a.lean_gate,a.samurai_gate,prompt)
    pre=read(a.preflight/'manifest.json');folder=Path(pre['sequences'][a.sequence]['source'])/a.sequence
    meta=read(folder/'frames.json');rows=[]
    if len(meta)!=len(frames):raise ValueError('Label metadata coverage differs')
    for i,frame in enumerate(frames):
        if digest(frame.mask_path)!=meta[i]['label_sha256']:raise ValueError('Label changed')
        truth=load_label_mask(frame.mask_path,[1])==1
        lp=np.asarray(Image.open(a.lean/f'{i:06d}.png').convert('L'))>0
        sp=(np.asarray(Image.open(a.samurai/f'{i:06d}.png').convert('L'))>0) if i>=start else np.zeros_like(truth)
        ls,ss=score(lp,truth),score(sp,truth)
        rows.append({'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,
          'initialization_frame':i==start,'lean':ls,'samurai':ss})
    result={'scope':'Exploratory same-MPS same archived automatic-mask comparison; no detector rerun or independent-generalization claim.',
      'sequence':a.sequence,'prompt_frame':start,'frames':len(rows),
      'pre_prompt_present_misses':sum(r['lean']['present'] for r in rows if r['frame']<start),
      'lean_receipt_sha256':digest(a.lean/'receipt.json'),'samurai_receipt_sha256':digest(a.samurai/'receipt.json'),
      'scorer_sha256':digest(Path(__file__)),'summaries':{arm:{'all':summary(rows,arm,post_only=False),
      'post_prompt':summary(rows,arm,post_only=True)} for arm in ('lean','samurai')},'rows':rows,
      'limitations':['Historical detector scanned successive frames until first detection; no new detector call or redetection was used.',
        'SAMURAI and project token-window Lean are different full systems; SAMURAI author memory thresholds are retained.',
        'The SAMURAI MPS Kalman branch uses an explicitly labeled float32 compatibility cast.',
        'Sequence 23 viewport is retrospective, patient independence and detector training overlap are unknown.',
        'Saved source frame indices are ordered manifest indices and do not establish native acquisition timing.']}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'prompt_frame':start,'summaries':result['summaries']}))


if __name__=='__main__':main()

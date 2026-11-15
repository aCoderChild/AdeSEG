#!/usr/bin/env python3
"""Score frozen full-video offline SAM2Long against same-MPS project Lean.

Verify both masks, prompt, source image hashes, device/config and checkpoint
before opening annotations. SAM2Long uses final offline pathway selection.
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
from scripts.run_sam2long_automatic_priority import verify_gate as long_gate,PATHWAY_CONFIG,CONFIG,dependency_hashes
from scripts.run_lean_window_baseline_mps import verify_gate as lean_gate
from scripts.score_samurai_automatic_priority import score,summary


def read(path):return json.loads(path.read_text())


def check_frozen(sequence,frames,lean,long,lean_gate_path,long_gate_path,prompt):
    l=read(lean/'receipt.json');x=read(long/'receipt.json')
    if not l.get('passed') or not x.get('passed') or not l.get('full_video') or not x.get('full_video') or not x.get('noncausal'):
        raise ValueError('Full Lean/SAM2Long predictions are not frozen')
    if l['sequence']!=sequence or x['sequence']!=sequence or l['prompt_frame']!=x['prompt_frame'] or l['prompt']!=prompt or x['prompt']!=prompt:
        raise ValueError('Prompt/crop identity differs')
    li=l['identity'];xi=x['identity']
    if (li['device']!='mps' or xi['device']!='mps' or li['fallback']!='0' or xi['fallback']!='0'
        or li['fast_math']!='0' or xi['fast_math']!='0' or li['torch']!=xi['torch']
        or li['checkpoint_sha256']!=xi['checkpoint_sha256'] or xi['pathways']!=PATHWAY_CONFIG
        or xi['config_sha256']!=digest(CONFIG) or xi['dependencies']!=dependency_hashes()):
        raise ValueError('Checkpoint, MPS or author pathway settings differ')
    if (li['runner_sha256']!=digest(ROOT/'scripts/run_lean_window_baseline_mps.py')
        or xi['runner_sha256']!=digest(ROOT/'scripts/run_sam2long_automatic_priority.py')
        or l['gate_sha256']!=digest(lean_gate_path) or x['gate_sha256']!=digest(long_gate_path)):
        raise ValueError('Runner or gate differs')
    lg=lean_gate(lean_gate_path,sequence,prompt,li);xg=long_gate(long_gate_path,sequence,prompt,xi)
    if l['state_sha256']!=lg['state_sha256'] or x['state_sha256']!=xg['state_sha256']:
        raise ValueError('Model state differs from short-clip verification')
    if l['end_frame']!=len(frames)-1 or x['end_frame']!=len(frames)-1 or len(l['records'])!=len(frames) or len(x['records'])!=len(frames):
        raise ValueError('Full-video frame coverage differs')
    pa=x.get('pathway_audit',{})
    if (pa.get('audited_post_prompt_frames')!=len(frames)-1-l['prompt_frame']
        or pa.get('pathways_after_initial_step')!=3
        or pa.get('checked_tensor_fields')!=['pred_masks','maskmem_features','obj_ptr','object_score_logits','ious']):
        raise ValueError('Full-video pathway state was not audited')
    for i,frame in enumerate(frames):
        for folder,run in ((lean,l),(long,x)):
            row=run['records'][i]
            if row['frame']!=i or row['source_frame_index']!=frame.frame_index or row['image_sha256']!=digest(frame.image_path) or row['mask_sha256']!=digest(folder/f'{i:06d}.png'):
                raise ValueError('Input image, frame index or frozen prediction changed')
            if i<l['prompt_frame']:
                with Image.open(folder/f'{i:06d}.png') as image:
                    if np.asarray(image.convert('L')).any():raise ValueError('Pre-prompt prediction must be empty')
    return l['prompt_frame']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=('seq11','seq20','seq23'),required=True)
    for key in ('lean','sam2long','lean-gate','long-gate','preflight','cropped-reference','cropped-backup','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};frames=videos[a.sequence]
    _,_,_,prompt=prepare_archived_prompt(a.sequence,frames,a.preflight,a.cropped_reference,a.cropped_backup)
    start=check_frozen(a.sequence,frames,a.lean,a.sam2long,a.lean_gate,a.long_gate,prompt)
    pre=read(a.preflight/'manifest.json');meta=read(Path(pre['sequences'][a.sequence]['source'])/a.sequence/'frames.json')
    if len(meta)!=len(frames):raise ValueError('Label metadata coverage differs')
    rows=[]
    for i,frame in enumerate(frames):
        if digest(frame.mask_path)!=meta[i]['label_sha256']:raise ValueError('Label bytes changed')
        truth=load_label_mask(frame.mask_path,[1])==1
        l=np.asarray(Image.open(a.lean/f'{i:06d}.png').convert('L'))>0
        x=np.asarray(Image.open(a.sam2long/f'{i:06d}.png').convert('L'))>0
        rows.append({'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,
          'initialization_frame':i==start,'lean':score(l,truth),'sam2long':score(x,truth)})
    result={'scope':'Exploratory same-MPS automatic-mask comparison; SAM2Long final pathway selection is offline.',
      'sequence':a.sequence,'prompt_frame':start,'frames':len(rows),
      'pre_prompt_present_misses':sum(r['lean']['present'] for r in rows if r['frame']<start),
      'lean_receipt_sha256':digest(a.lean/'receipt.json'),'sam2long_receipt_sha256':digest(a.sam2long/'receipt.json'),
      'scorer_sha256':digest(Path(__file__)),'summaries':{arm:{'all':summary(rows,arm,post_only=False),
      'post_prompt':summary(rows,arm,post_only=True)} for arm in ('lean','sam2long')},'rows':rows,
      'limitations':['SAM2Long selects final pathway after the full video; it is not a causal streaming output.',
       'Author SAM2Long pathway selection parameters include native thresholds; no new project gate or detector is added.',
       'The initial automatic mask itself used historical YOLO scanning until first detection and is not a ground-truth mask.',
       'Sequence-specific seq23 crop is retrospective; patient independence, detector exposure and label completeness are unknown.']}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'prompt_frame':start,'summaries':result['summaries']}))


if __name__=='__main__':main()

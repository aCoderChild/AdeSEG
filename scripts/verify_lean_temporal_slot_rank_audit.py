#!/usr/bin/env python3
"""Independently recompute saved slot-diagnostic mask scores and changes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('audit','replay','verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());d=read(a.audit/'results.json');done=read(a.audit/'completion.json');r=read(a.replay/'completion.json');v=read(a.verification/'verification.json')
if not all(x['passed'] for x in (done,r,v)) or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/audit_lean_temporal_slot_rank.py') or d['replay_sha256']!=digest(a.replay/'completion.json') or d['verification_sha256']!=digest(a.verification/'verification.json'):raise ValueError('Audit provenance changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23']
for x,e in zip(d['frames'],r['events'],strict=True):
 i=x['frame'];f=frames[i]
 if i!=e['frame'] or digest(f.image_path)!=x['image_sha256'] or digest(f.mask_path)!=x['label_sha256'] or x['total_keys']!=e['total_keys'] or x['pointer_keys']!=e['pointer_keys']:raise ValueError('Frame or keys differ')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any());masks={}
 if present!=x['present']:raise ValueError('Presence differs')
 for arm,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{arm}.png'
  if digest(path)!=sha:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0;n=int(pred.sum());g=int(gt.sum());tp=int((pred&gt).sum());dice=2*tp/(n+g) if present else None;fa=bool(n) if not present else None;z=x['scores'][arm]
  if z['predicted_pixels']!=n or z['false_alarm']!=fa or (z['dice'] is None)!=(dice is None) or dice is not None and abs(z['dice']-dice)>1e-12:raise ValueError('Score differs')
  masks[arm]=pred
 for arm in ('rank','permuted'):
  if x['changed_pixels'][arm]!=int(np.count_nonzero(masks[arm]!=masks['noop'])):raise ValueError('Changed pixel count differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['frames']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All slot-diagnostic mask scores independently verified.')

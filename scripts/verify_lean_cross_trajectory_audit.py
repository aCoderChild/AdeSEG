#!/usr/bin/env python3
"""Independently recalculate cross-trajectory mask metrics and differences."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('audit','replay','replay-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json');r=read(a.replay/'completion.json');v=read(a.replay_verification/'verification.json')
if not all(x['passed'] for x in (done,r,v)) or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['replay_sha256']!=digest(a.replay/'completion.json') or d['verification_sha256']!=digest(a.replay_verification/'verification.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_lean_cross_trajectory_memory.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_cross_trajectory_repeats.py'):raise ValueError('Audit provenance changed')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos['seq23']
for row,e in zip(d['frames'],r['events'],strict=True):
 i=row['frame'];f=frames[i]
 if i!=e['frame'] or digest(f.mask_path)!=row['label_sha256'] or row['total_keys']!=e['total_keys'] or row['pointer_keys']!=e['pointer_keys']:raise ValueError('Frame or read geometry differs')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any())
 if present!=row['present']:raise ValueError('Presence differs')
 masks={}
 for arm,expected in e['masks'].items():
  path=a.replay/f'{i:06d}_{arm}.png'
  if digest(path)!=expected:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0;n=int(np.count_nonzero(pred));g=int(np.count_nonzero(gt));tp=int(np.count_nonzero(pred&gt));score=row['scores'][arm]
  dice=2*tp/(n+g) if present else None;false=bool(n) if not present else None
  if score['predicted_pixels']!=n or score['false_alarm']!=false or (score['dice'] is None)!=(dice is None) or dice is not None and abs(score['dice']-dice)>1e-12:raise ValueError('Score differs')
  masks[arm]=pred
 for key,a0,b0 in (('raw_with_fused_vs_raw','raw_with_fused','raw_noop'),('fused_with_raw_vs_fused','fused_with_raw','fused_noop')):
  if row['changed_pixels'][key]!=int(np.count_nonzero(masks[a0]!=masks[b0])):raise ValueError('Changed-pixel count differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['frames']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All cross-trajectory mask scores and pixel differences independently verified.')

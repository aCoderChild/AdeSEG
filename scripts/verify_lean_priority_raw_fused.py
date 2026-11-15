#!/usr/bin/env python3
"""Independently recompute scores, matched-read counts and saved-mask hashes."""
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
for k in ('audit','baseline','raw-window','fused-window','raw-two','output'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--sequence',choices=('seq8','seq20'),required=True)
a=p.parse_args();done=read(a.audit/'completion.json');result=read(a.audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['source_sha256']!=digest(ROOT/'scripts/audit_lean_priority_raw_fused.py') or result['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_fused_priority_followup_v1.json'):raise ValueError('Audit provenance mismatch')
runs={'baseline':a.baseline,'raw_window':a.raw_window,'fused_window':a.fused_window,'raw_two':a.raw_two};frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[a.sequence]
if len(frames)!=len(result['frames']):raise ValueError('Frame count differs')
records={name:read(path/'records.json') for name,path in runs.items()}
for i,(f,row) in enumerate(zip(frames,result['frames'],strict=True)):
 if row['frame']!=i:raise ValueError('Frame order differs')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any())
 if row['present']!=present:raise ValueError('Label presence differs')
 for name,path in runs.items():
  predpath=path/f'{i:06d}.png';score=row['scores'][name]
  if digest(predpath)!=records[name]['mask_sha256'][i] or score['mask_sha256']!=records[name]['mask_sha256'][i]:raise ValueError('Mask hash mismatch')
  pred=np.asarray(Image.open(predpath))>0
  if pred.shape!=gt.shape:raise ValueError('Mask dimensions differ')
  den=np.count_nonzero(pred)+np.count_nonzero(gt);dice=2*np.count_nonzero(pred&gt)/den if den else 1.;fa=bool(pred.any()) if not present else None
  if abs(score['dice']-dice)>1e-12 or score['false_alarm']!=fa:raise ValueError('Per-frame score mismatch')
for name,path in runs.items():
 scores=[r['scores'][name]['dice'] for r in result['frames'] if r['matched']];empty=[r['scores'][name]['false_alarm'] for r in result['frames'] if not r['present']];z=result['conditions'][name]
 if len(scores)!=z['matched_present_count'] or abs(float(np.mean(scores))-z['matched_present_dice'])>1e-12 or sum(empty)!=z['false_alarms'] or len(empty)!=z['empty_count']:raise ValueError('Aggregated score mismatch')
a0=read(a.fused_window/'positions.json');b0=read(a.raw_two/'positions.json')
if len(a0)!=len(b0) or len(a0)!=result['read_count'] or sum(x['spatial_keys']==y['spatial_keys'] for x,y in zip(a0,b0))!=result['matched_spatial_key_counts'] or sum(x['pointer_keys']==y['pointer_keys'] for x,y in zip(a0,b0))!=result['matched_pointer_key_counts']:raise ValueError('Read counts mismatch')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':a.sequence,'frames':len(frames),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('Per-frame scores, summaries and matched-read counts verified.')

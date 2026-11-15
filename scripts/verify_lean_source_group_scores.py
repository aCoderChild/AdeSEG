#!/usr/bin/env python3
"""Independently recompute per-mask confusion counts for frozen group results."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('audit','replay','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());done=read(a.audit/'completion.json');r=read(a.audit/'results.json');replay=read(a.replay/'completion.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or r['runner_sha256']!=digest(ROOT/'scripts/audit_lean_source_group_deletion.py') or r['replay_sha256']!=digest(a.replay/'completion.json') or r['sequence']!=replay['sequence']:raise ValueError('Audit provenance differs')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[r['sequence']];events={e['frame']:e for e in replay['events']};count=0
for item in r['frames']:
 i=item['frame'];f=frames[i]
 if digest(f.mask_path)!=item['label_sha256']:raise ValueError('Label differs')
 truth=load_label_mask(f.mask_path,[1])==1;positive=int(np.count_nonzero(truth))
 if bool(positive)!=item['present']:raise ValueError('Presence differs')
 for arm,score in item['scores'].items():
  path=a.replay/f'{i:06d}_{arm}.png'
  if digest(path)!=events[i]['masks'][arm]:raise ValueError('Mask differs')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=truth.shape:raise ValueError('Geometry differs')
  tp=int(np.count_nonzero(pred&truth));fp=int(np.count_nonzero(pred&~truth));fn=int(np.count_nonzero(~pred&truth));den=2*tp+fp+fn;dice=2*tp/den if den else 1.
  if abs(dice-score['dice'])>1e-12 or score['predicted_pixels']!=tp+fp or score['false_alarm']!=(bool(tp+fp) if not positive else None):raise ValueError('Score differs')
  count+=1
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'sequence':r['sequence'],'verified_scores':count,'results_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))};(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

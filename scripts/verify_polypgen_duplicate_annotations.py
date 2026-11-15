#!/usr/bin/env python3
"""Independently verify exact duplicate groups and label disagreements."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('input-audit','duplicate-audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();src=read(a.input_audit/'results.json');done=read(a.duplicate_audit/'completion.json');d=read(a.duplicate_audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.duplicate_audit/'results.json') or done['report_sha256']!=digest(a.duplicate_audit/'report.md') or d['input_sha256']!=digest(a.input_audit/'results.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_duplicate_annotations.py'):raise ValueError('Audit changed')
groups=collections.defaultdict(list)
for x in src['single_frame_files']:groups[x['pixel_sha256']].append(x['image'])
expected={k:sorted(v) for k,v in groups.items() if len(v)>1}
observed={x['pixel_sha256']:sorted(item['image'] for item in x['items']) for x in d['groups']}
if expected!=observed:raise ValueError('Duplicate groups differ')
for g in d['groups']:
 imgs=[];masks=[]
 for x in g['items']:
  ip=Path(x['image']);mp=Path(x['mask'])
  if digest(ip)!=x['image_sha256'] or digest(mp)!=x['mask_sha256']:raise ValueError('Input changed')
  with Image.open(ip) as f:imgs.append(np.asarray(f.convert('RGB')).copy())
  with Image.open(mp) as f:a0=np.asarray(f.convert('L'))
  masks.append(a0>(0 if a0.max()<=1 else 127))
 if any(not np.array_equal(imgs[0],img) for img in imgs[1:]):raise ValueError('Images differ')
 for pair in g['pairs']:
  x=masks[pair['first']];y=masks[pair['second']];den=int((x|y).sum());iou=int((x&y).sum())/den if den else 1.
  if abs(iou-pair['mask_iou'])>1e-12 or int(x.sum())!=pair['first_mask_pixels'] or int(y.sum())!=pair['second_mask_pixels'] or (g['items'][pair['first']]['boxes']==g['items'][pair['second']]['boxes'])!=pair['same_boxes']:raise ValueError('Label disagreement differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'groups':len(observed),'results_sha256':digest(a.duplicate_audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Duplicate groups and annotation disagreement independently reproduced.')

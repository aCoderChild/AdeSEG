#!/usr/bin/env python3
"""Independently verify two fixed-image matched-detector engineering runs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','inputs','original','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');src=read(a.inputs/'inputs.json')
if not x['passed'] or x!=y or x['mode']!='gate' or x['inputs_sha256']!=digest(a.inputs/'inputs.json') or x['original_sha256']!=digest(a.original) or x['candidate_sha256']!=digest(a.candidate) or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_finetune_c6.py') or x['helper_sha256']!=digest(ROOT/'scripts/run_prompt_feature_refinement.py') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json') or len(x['events'])!=2:raise ValueError('Gate differs')
for e,row in zip(x['events'],src['inputs'][:2],strict=True):
 if e['id']!=row['id'] or e['image_sha256']!=digest(Path(row['image'])) or set(e['boxes'])!={'original','candidate'}:raise ValueError('Gate input changed')
 for b in e['boxes'].values():
  if b['box'] is None:
   if b['confidence'] is not None:raise ValueError('Invalid missing detection')
  else:
   x0,y0,x1,y1=b['box']
   if not(0<=x0<x1<=row['width'] and 0<=y0<y1<=row['height'] and np.isfinite(b['confidence'])):raise ValueError('Box outside image')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'inputs_sha256':x['inputs_sha256'],'original_sha256':x['original_sha256'],'candidate_sha256':x['candidate_sha256'],'runner_sha256':x['runner_sha256'],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Two-image original and candidate detector gates reproduced.')

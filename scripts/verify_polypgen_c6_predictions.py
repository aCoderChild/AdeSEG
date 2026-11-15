#!/usr/bin/env python3
"""Audit repeated paired C6 detector predictions before label scoring."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','inputs','gate','gate-reference','original','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');src=read(a.inputs/'inputs.json');g=read(a.gate/'verification.json');prior=read(a.gate_reference/'completion.json')
if not all(z['passed'] for z in (x,y,src,g,prior)) or x!=y or x['mode']!='candidate' or x['inputs_sha256']!=digest(a.inputs/'inputs.json') or x['original_sha256']!=digest(a.original) or x['candidate_sha256']!=digest(a.candidate) or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_c6_detector.py') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_c6_detector_comparison_v1.json') or g['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_c6_detector_gate.py') or g['runner_sha256']!=x['runner_sha256'] or prior['mode']!='gate' or x['events'][:2]!=prior['events'] or x['model_state_sha256']!=prior['model_state_sha256']:raise ValueError('Prediction provenance or reproducibility differs')
if len(x['events'])!=88 or len(src['inputs'])!=88:raise ValueError('C6 scope differs')
for e,row in zip(x['events'],src['inputs'],strict=True):
 if e['id']!=row['id'] or e['image_sha256']!=row['image_sha256'] or digest(Path(row['image']))!=row['image_sha256'] or set(e['boxes'])!={'original','candidate'}:raise ValueError('Input identity differs')
 for b in e['boxes'].values():
  if b['box'] is None:
   if b['confidence'] is not None:raise ValueError('Invalid missing result')
  else:
   x0,y0,x1,y1=b['box']
   if not (0<=x0<x1<=row['width'] and 0<=y0<y1<=row['height'] and isinstance(b['confidence'],float) and np.isfinite(b['confidence'])):raise ValueError('Invalid box geometry')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'images':88,'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'inputs_sha256':digest(a.inputs/'inputs.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All 88 paired detector predictions repeated and verified.')

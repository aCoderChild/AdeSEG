#!/usr/bin/env python3
"""Independently verify repeatability and artifacts of label-free image gates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('first','second','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args()
r=[json.loads((x/'completion.json').read_text()) for x in (a.first,a.second)]
if not all(x['passed'] and x['mode']=='gate' for x in r) or r[0]['identity']!=r[1]['identity'] or r[0]['events']!=r[1]['events'] or len(r[0]['events'])!=3:raise ValueError('Gate results differ')
for root,report in zip((a.first,a.second),r):
 for e in report['events']:
  if e['sequence']!='seq3' or set(e['masks'])!={'fixed_initial','synthetic_control'}:raise ValueError('Invalid gate scope')
  for name,expected in e['masks'].items():
   if digest(root/f"{e['sequence']}_{e['frame']:06d}_{name}.png")!=expected:raise ValueError('Mask hash changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'identity':r[0]['identity'],'events':r[0]['events'],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('Three fixed and synthetic image-prompt pairs reproduced exactly.')

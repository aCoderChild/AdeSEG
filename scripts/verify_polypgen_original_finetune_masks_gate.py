#!/usr/bin/env python3
"""Check repeated fixed-frame SAM2 box and feature-point decoding."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','checkpoint','fixed-box-reference','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');old=read(a.fixed_box_reference/'completion.json')
if not x['passed'] or x!=y or x['mode']!='gate' or x['sam2_checkpoint_sha256']!=digest(a.checkpoint) or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_original_finetune_masks.py') or x['feature_helper_sha256']!=digest(ROOT/'modeling/early_feature_prompt.py') or x['selector_sha256']!=digest(ROOT/'modeling/prompt_feature_refinement.py') or len(x['events'])!=1:raise ValueError('Mask gate differs')
e=x['events'][0];prior=old['events'][0];seq=e['sequence'];i=e['frame']
if seq!='seq3' or i!=prior['frame'] or e['original_box_viewport']!=e['candidate_box_viewport'] or e['points']['original']!=e['points']['candidate'] or e['masks']['original_box']!=prior['masks']['fixed_initial'] or e['masks']['original_box']!=e['masks']['candidate_box'] or e['masks']['original_points']!=e['masks']['candidate_points']:raise ValueError('Mask no-op or point control differs')
for name,sha in e['masks'].items():
 for folder in (a.first,a.second):
  if digest(folder/f'{seq}_{i:06d}_{name}.png')!=sha:raise ValueError('Gate mask changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':digest(a.checkpoint),'runner_sha256':x['runner_sha256'],'verifier_sha256':digest(Path(__file__)),'feature_helper_sha256':x['feature_helper_sha256'],'selector_sha256':x['selector_sha256']},indent=2)+'\n');print('SAM2 box and early-feature point masks reproduced exactly.')

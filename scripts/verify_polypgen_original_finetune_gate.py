#!/usr/bin/env python3
"""Verify repeated original-detector inference at its saved seq3 prompt."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json')
if not x['passed'] or x!=y or x['mode']!='gate' or x['checkpoint_sha256']!=digest(a.checkpoint) or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_original_finetune.py') or x['detector_helper_sha256']!=digest(ROOT/'scripts/run_prompt_feature_refinement.py') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json') or len(x['events'])!=1:raise ValueError('Gate repetition or provenance changed')
e=x['events'][0]
if e['sequence']!='seq3' or e['viewport'] or e['candidate_box_viewport'] is None or not np.allclose(e['candidate_box_viewport'],e['original_box_viewport'],rtol=0,atol=1e-5) or abs(e['candidate_confidence']-e['original_confidence'])>1e-6:raise ValueError('Saved original prompt does not reproduce')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'preflight_sha256':x['preflight_sha256'],'crop_sha256':x['crop_sha256'],'protocol_sha256':x['protocol_sha256'],'checkpoint_sha256':x['checkpoint_sha256'],'model_state_sha256':x['model_state_sha256'],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Original seq3 detector output reproduced twice without target labels.')

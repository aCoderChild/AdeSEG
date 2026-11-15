#!/usr/bin/env python3
"""Audit repeated one-pass detector predictions before opening target labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','checkpoint','preflight','crop-reference','training-verification','gate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');t=read(a.training_verification/'verification.json');g=read(a.gate_verification/'verification.json');m=read(a.preflight/'manifest.json');crop=read(a.crop_reference/'prompt.json')
if not x['passed'] or x!=y or x['mode']!='candidate' or not t['passed'] or t['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_detector_finetune.py') or t['best_sha256']!=digest(a.checkpoint) or x['checkpoint_sha256']!=t['best_sha256'] or not g['passed'] or g['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_gate.py') or g['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_original_finetune.py') or x['runner_sha256']!=g['runner_sha256'] or x['preflight_sha256']!=digest(a.preflight/'manifest.json') or x['crop_sha256']!=digest(a.crop_reference/'prompt.json') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json'):raise ValueError('Prediction provenance or repeat differs')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};events=x['events']
if [e['sequence'] for e in events]!=['seq8','seq11','seq20','seq23']:raise ValueError('Scope differs')
for e in events:
 seq=e['sequence'];src=Path(m['sequences'][seq]['source'])/seq;prompt=read(src/'prompt.json')
 if digest(src/'prompt.json')!=m['sequences'][seq]['prompt_sha256']:raise ValueError('Original prompt changed')
 expected=crop if seq=='seq23' else prompt;box=expected['box_viewport_xyxy'] if seq=='seq23' else expected['box']
 if e['frame']!=expected['first_detection_index'] or e['original_box_viewport']!=box or e['original_confidence']!=expected['confidence'] or e['viewport']!=(seq=='seq23') or e['image_sha256']!=digest(videos[seq][e['frame']].image_path):raise ValueError('Initial frame or prompt differs')
 pred=e['candidate_box_viewport'];full=e['candidate_box_full'];conf=e['candidate_confidence']
 if pred is None:
  if full is not None or conf is not None:raise ValueError('No detection status differs')
 else:
  if len(pred)!=4 or not np.isfinite(pred).all() or not isinstance(conf,float) or not np.isfinite(conf):raise ValueError('Invalid detector box')
  with Image.open(videos[seq][e['frame']].image_path) as image:width,height=image.size
  if seq=='seq23':width=VIEWPORT[2]-VIEWPORT[0];height=VIEWPORT[3]-VIEWPORT[1]
  if not (0<=pred[0]<pred[2]<=width and 0<=pred[1]<pred[3]<=height):raise ValueError('Detector box outside viewport')
  expected_full=[pred[0]+VIEWPORT[0],pred[1]+VIEWPORT[1],pred[2]+VIEWPORT[0],pred[3]+VIEWPORT[1]] if seq=='seq23' else pred
  if full!=expected_full:raise ValueError('Viewport restoration differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'predictions_sha256':digest(a.first/'completion.json'),'repeat_sha256':digest(a.second/'completion.json'),'checkpoint_sha256':digest(a.checkpoint),'frames':len(events),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Four one-pass initial-frame predictions reproduced and independently verified.')

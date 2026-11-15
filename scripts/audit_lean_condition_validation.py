#!/usr/bin/env python3
"""Score verified saved conditioning-region predictions after replay."""
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
for k in ('capture','replay','verification','merged','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'condition_validation_capture.json');replay=read(a.replay/'completion.json');v=read(a.verification/'verification.json')
if not all(x['passed'] for x in (cap,replay,v)) or v['first_sha256']!=digest(a.replay/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_validation_repeats.py') or replay['capture_sha256']!=digest(a.capture/'condition_validation_capture.json'):raise ValueError('Replay verification changed')
seq=cap['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq]
if cap.get('viewport',False):
 folder=Path(cap['reference']);done=read(folder/'completion.json')
 if not done['passed'] or digest(folder/'frames.json')!=done['frames_sha256']:raise ValueError('Cropped reference changed')
 rows=read(folder/'frames.json')
else:
 source=Path(read(a.merged/'locations.json')[seq]);done=read(source/'completion.json');record=read(source/'records.json')[0]
 if not done['passed'] or digest(source/'records.json')!=done['records_sha256'] or digest(source/seq/'frames.json')!=record['frames_sha256']:raise ValueError('Source changed')
 rows=read(source/seq/'frames.json')
result=[]
for event in replay['events']:
 i=event['frame'];frame=frames[i]
 if digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Label changed')
 gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any());scores={}
 for name,h in event['masks'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=h:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=gt.shape:raise ValueError('Mask dimensions differ')
  n=int(pred.sum())+int(gt.sum());scores[name]={'dice':2*int((pred&gt).sum())/n if n else 1.,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
 result.append({'frame':i,'present':present,'label_sha256':rows[i]['label_sha256'],'scores':scores})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'results.json').write_text(json.dumps({'sequence':seq,'scope':'Post-hoc fixed-read conditioning deletion; not propagated.','replay_sha256':digest(a.replay/'completion.json'),'runner_sha256':digest(Path(__file__)),'frames':result},indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n');print(seq,'masks scored')

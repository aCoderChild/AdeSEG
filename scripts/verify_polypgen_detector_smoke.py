#!/usr/bin/env python3
"""Independently verify engineering-only YOLO training smoke artifacts."""
import argparse,csv,json,math,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('smoke','dataset','dataset-verification','pretrained','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.smoke/'completion.json');setup=read(a.smoke/'setup.json');dv=read(a.dataset_verification/'verification.json')
if not done['passed'] or done['runner_sha256']!=digest(ROOT/'scripts/run_polypgen_detector_smoke.py') or done['setup_sha256']!=digest(a.smoke/'setup.json') or setup['dataset_sha256']!=digest(a.dataset/'dataset.json') or setup['dataset_verification_sha256']!=digest(a.dataset_verification/'verification.json') or not dv['passed'] or dv['dataset_sha256']!=setup['dataset_sha256'] or dv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or setup['pretrained_sha256']!=digest(a.pretrained) or done['results_sha256']!=digest(a.smoke/'train/results.csv') or done['checkpoint_sha256']!=digest(a.smoke/'train/weights/last.pt') or setup['mode']!='engineering_only':raise ValueError('Smoke provenance changed')
with (a.smoke/'train/results.csv').open(newline='') as f:rows=list(csv.DictReader(f))
if len(rows)!=1 or any(value and not math.isfinite(float(value)) for value in rows[0].values()):raise ValueError('Invalid training metrics')
args=setup['training_args']
if args['epochs']!=1 or args['device']!='mps' or args['amp'] is not False or args['seed']!=20261114 or args['workers']!=0:raise ValueError('Smoke configuration changed')
os.environ['YOLO_CONFIG_DIR']=str(a.smoke/'ultralytics_config')
from ultralytics import YOLO
base=YOLO(str(a.pretrained));trained=YOLO(str(a.smoke/'train/weights/last.pt'))
if base.task!='detect' or len(base.names)!=80 or trained.task!='detect' or trained.names!={0:'polyp'} or torch.equal(base.model.state_dict()['model.0.conv.weight'].detach().cpu(),trained.model.state_dict()['model.0.conv.weight'].detach().cpu()):raise ValueError('Smoke checkpoint did not train expected model')
for split,records in setup['selected'].items():
 image_dir=a.smoke/'smoke_data'/'images'/split;label_dir=a.smoke/'smoke_data'/'labels'/split
 if len(list(image_dir.glob('*.jpg')))!=len(records) or len(list(label_dir.glob('*.txt')))!=len(records):raise ValueError('Smoke subset changed')
 actual=[]
 for img in sorted(image_dir.glob('*.jpg')):
  label=label_dir/(img.stem+'.txt')
  if img.is_symlink() or not label.is_file():raise ValueError('Smoke source not isolated')
  actual.append((digest(img),digest(label)))
 if sorted(actual)!=sorted((x['image_sha256'],x['label_sha256']) for x in records):raise ValueError('Smoke inputs changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'smoke_setup_sha256':digest(a.smoke/'setup.json'),'dataset_sha256':setup['dataset_sha256'],'pretrained_sha256':setup['pretrained_sha256'],'smoke_checkpoint_sha256':done['checkpoint_sha256'],'verifier_sha256':digest(Path(__file__)),'scope':'Engineering-only; no efficacy claim.'},indent=2)+'\n');print('Official YOLO smoke artifacts and isolated inputs independently verified.')

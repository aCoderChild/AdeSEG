#!/usr/bin/env python3
"""Independently verify the completed C1-C5/C6 detector training pilot."""
import argparse,csv,json,math,os,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('training','dataset','dataset-verification','smoke-verification','pretrained','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.training/'completion.json');setup=read(a.training/'setup.json');status=read(a.training/'status.json');data=read(a.dataset/'dataset.json');dv=read(a.dataset_verification/'verification.json');sv=read(a.smoke_verification/'verification.json')
if not done['passed'] or status['status']!='completed' or done['setup_sha256']!=digest(a.training/'setup.json') or done['runner_sha256']!=digest(ROOT/'scripts/train_polypgen_one_shot_detector.py') or setup['runner_sha256']!=done['runner_sha256'] or setup['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json') or not dv['passed'] or dv['dataset_sha256']!=digest(a.dataset/'dataset.json') or dv['config_sha256']!=digest(a.dataset/'data.yaml') or dv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or not sv['passed'] or sv['dataset_sha256']!=dv['dataset_sha256'] or sv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_smoke.py') or setup['pretrained_sha256']!=digest(a.pretrained) or sv['pretrained_sha256']!=setup['pretrained_sha256']:raise ValueError('Training provenance changed')
results=a.training/'train/results.csv';best=a.training/'train/weights/best.pt';last=a.training/'train/weights/last.pt'
if digest(results)!=done['results_sha256'] or digest(best)!=done['best_sha256'] or digest(last)!=done['last_sha256']:raise ValueError('Training artifacts changed')
args=setup['training_args'];actual=yaml.safe_load((a.training/'train/args.yaml').read_text());expected={'epochs':20,'imgsz':640,'batch':8,'seed':20261114,'workers':0,'device':'mps','amp':False,'cache':False,'deterministic':True,'patience':0,'plots':False,'save':True,'val':True,'close_mosaic':10,'optimizer':'auto'}
for key,value in expected.items():
 if args[key]!=value or actual[key]!=value:raise ValueError('Configuration differs: '+key)
if Path(actual['data']).resolve()!=(a.dataset/'data.yaml').resolve() or Path(actual['model']).resolve()!=a.pretrained.resolve():raise ValueError('Training inputs differ')
with results.open(newline='') as f:rows=list(csv.DictReader(f))
if len(rows)!=20 or [int(float(r['epoch'])) for r in rows]!=list(range(1,21)) or any(x and not math.isfinite(float(x)) for row in rows for x in row.values()):raise ValueError('Epoch metrics incomplete or nonfinite')
for x in data['included']:
 image=Path(x['staged_image']);label=Path(x['staged_label'])
 if image.is_symlink() or digest(image)!=x['image_sha256'] or digest(label)!=x['label_sha256']:raise ValueError('Staged training input changed')
os.environ['YOLO_CONFIG_DIR']=str(a.training/'ultralytics_config')
from ultralytics import YOLO
selected=YOLO(str(best));final=YOLO(str(last));base=YOLO(str(a.pretrained))
if selected.task!='detect' or selected.names!={0:'polyp'} or final.names!={0:'polyp'} or len(base.names)!=80 or torch.equal(base.model.state_dict()['model.0.conv.weight'].detach().cpu(),final.model.state_dict()['model.0.conv.weight'].detach().cpu()):raise ValueError('Model format or training update invalid')
fitness=float(selected.ckpt['train_metrics']['fitness']);highest=max(float(x['metrics/mAP50-95(B)']) for x in rows)
if abs(fitness-highest)>2e-5:raise ValueError('Selected checkpoint differs from center-C6 validation fitness')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'epochs':20,'best_sha256':digest(best),'last_sha256':digest(last),'results_sha256':digest(results),'training_completion_sha256':digest(a.training/'completion.json'),'dataset_sha256':digest(a.dataset/'dataset.json'),'pretrained_sha256':digest(a.pretrained),'validation_max_map50_95':highest,'verifier_sha256':digest(Path(__file__)),'scope':'Reproducibility and provenance only; not an independent-patient efficacy claim.'}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('20-epoch training, validation selection and source integrity independently verified.')

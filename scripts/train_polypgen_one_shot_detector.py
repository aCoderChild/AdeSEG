#!/usr/bin/env python3
"""Train the frozen one-shot YOLO pilot on the verified center split."""
import argparse,csv,json,math,os,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def check_staged(data):
 for row in data['included']:
  image=Path(row['staged_image']);label=Path(row['staged_label'])
  if image.is_symlink() or digest(image)!=row['image_sha256'] or digest(label)!=row['label_sha256']:raise ValueError('Staged training input changed')

def main():
 p=argparse.ArgumentParser()
 for k in ('dataset','dataset-verification','smoke-verification','pretrained','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();d=read(a.dataset/'dataset.json');v=read(a.dataset_verification/'verification.json');smoke=read(a.smoke_verification/'verification.json');pretrained_sha=digest(a.pretrained)
 if not v['passed'] or v['dataset_sha256']!=digest(a.dataset/'dataset.json') or v['config_sha256']!=digest(a.dataset/'data.yaml') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or not smoke['passed'] or smoke['dataset_sha256']!=v['dataset_sha256'] or smoke['pretrained_sha256']!=pretrained_sha or smoke['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_smoke.py') or d['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json') or pretrained_sha!='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36':raise ValueError('Training preflight changed')
 if not torch.backends.mps.is_available():raise ValueError('MPS device unavailable')
 check_staged(d);out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config');os.environ['WANDB_MODE']='disabled'
 from ultralytics import YOLO,__version__ as ultralytics_version
 if ultralytics_version!='8.4.95' or str(torch.__version__)!='2.12.0':raise ValueError('Training runtime changed since smoke test')
 model=YOLO(str(a.pretrained));before=model.model.state_dict()['model.0.conv.weight'].detach().cpu().clone()
 if model.task!='detect' or len(model.names)!=80 or model.ckpt.get('train_args',{}).get('data')!='coco.yaml':raise ValueError('Unexpected COCO initialization')
 args={'data':str(a.dataset/'data.yaml'),'epochs':20,'imgsz':640,'batch':8,'seed':20261114,'workers':0,'device':'mps','amp':False,'cache':False,'deterministic':True,'patience':0,'plots':False,'save':True,'val':True,'project':str(out),'name':'train','exist_ok':False,'close_mosaic':10,'optimizer':'auto'}
 setup={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'dataset_sha256':v['dataset_sha256'],'config_sha256':v['config_sha256'],'dataset_verification_sha256':digest(a.dataset_verification/'verification.json'),'smoke_verification_sha256':digest(a.smoke_verification/'verification.json'),'pretrained_sha256':pretrained_sha,'pretrained_source':'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt','protocol_sha256':digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'),'training_args':args,'ultralytics':ultralytics_version,'torch':str(torch.__version__),'runner_sha256':digest(Path(__file__)),'caution':'MPS training may not be bitwise reproducible; no sequence labels are used.'}
 (out/'setup.json').write_text(json.dumps(setup,indent=2)+'\n');(out/'status.json').write_text(json.dumps({'status':'running','pid':os.getpid(),'started_utc':setup['started_utc']},indent=2)+'\n')
 try:
  model.train(**args)
  results=out/'train/results.csv';last=out/'train/weights/last.pt';best=out/'train/weights/best.pt'
  if not all(q.is_file() for q in (results,last,best)):raise ValueError('Training output missing')
  with results.open(newline='') as f:rows=list(csv.DictReader(f))
  if len(rows)!=20 or any(value and not math.isfinite(float(value)) for row in rows for value in row.values()):raise ValueError('Training did not complete 20 finite epochs')
  selected=YOLO(str(best));final=YOLO(str(last))
  if selected.names!={0:'polyp'} or final.names!={0:'polyp'} or selected.task!='detect' or torch.equal(before,final.model.state_dict()['model.0.conv.weight'].detach().cpu()):raise ValueError('Training checkpoint invalid')
  check_staged(d)
  complete={'passed':True,'epochs':len(rows),'setup_sha256':digest(out/'setup.json'),'results_sha256':digest(results),'best_sha256':digest(best),'last_sha256':digest(last),'runner_sha256':digest(Path(__file__)),'scope':'Exploratory center-separated detector training; no priority-video labels used.'}
  (out/'completion.json').write_text(json.dumps(complete,indent=2)+'\n');(out/'status.json').write_text(json.dumps({'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print('One-shot detector training completed and input hashes rechecked.')
 except BaseException as e:
  (out/'status.json').write_text(json.dumps({'status':'failed','error':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run an isolated one-epoch Ultralytics dataset/device smoke test."""
import argparse,csv,json,math,os,shutil,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('dataset','dataset-verification','pretrained','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();d=read(a.dataset/'dataset.json');v=read(a.dataset_verification/'verification.json')
 if not v['passed'] or v['dataset_sha256']!=digest(a.dataset/'dataset.json') or v['config_sha256']!=digest(a.dataset/'data.yaml') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or d['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'):raise ValueError('Verified dataset changed')
 if digest(a.pretrained)!='858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e' or not torch.backends.mps.is_available():raise ValueError('Original detector or MPS unavailable')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config');os.environ['WANDB_MODE']='disabled'
 from ultralytics import YOLO,__version__ as ultralytics_version
 base=YOLO(str(a.pretrained));before=base.model.state_dict()['model.0.conv.weight'].detach().cpu().clone()
 if base.task!='detect' or base.names!={0:'polyp'}:raise ValueError('Expected original one-class PolypGen detector')
 chosen={}
 for split,positive,negative in (('train',4,2),('val',2,1)):
  rows=sorted((x for x in d['included'] if x['split']==split),key=lambda x:x['staged_image']);chosen[split]=[x for x in rows if x['boxes_xyxy']][:positive]+[x for x in rows if not x['boxes_xyxy']][:negative]
  if len(chosen[split])!=positive+negative:raise ValueError('Smoke class coverage unavailable')
  for kind in ('images','labels'):(out/'smoke_data'/kind/split).mkdir(parents=True,exist_ok=True)
  for x in chosen[split]:
   original=Path(x['staged_image']);label=Path(x['staged_label']);name=original.name
   if original.is_symlink() or digest(original)!=x['image_sha256'] or digest(label)!=x['label_sha256']:raise ValueError('Smoke input changed')
   shutil.copyfile(original,out/'smoke_data'/'images'/split/name);(out/'smoke_data'/'labels'/split/(Path(name).stem+'.txt')).write_bytes(label.read_bytes())
 config={'path':str(out/'smoke_data'),'train':'images/train','val':'images/val','names':{0:'polyp'}};(out/'smoke_data'/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 args={'data':str(out/'smoke_data'/'data.yaml'),'epochs':1,'imgsz':640,'batch':2,'seed':20261114,'workers':0,'device':'mps','amp':False,'cache':False,'deterministic':True,'patience':0,'plots':False,'save':True,'val':True,'project':str(out),'name':'train','exist_ok':False,'close_mosaic':0,'optimizer':'AdamW','lr0':0.0001,'lrf':0.1,'warmup_epochs':1}
 (out/'setup.json').write_text(json.dumps({'mode':'engineering_only','pretrained_sha256':digest(a.pretrained),'dataset_sha256':digest(a.dataset/'dataset.json'),'dataset_verification_sha256':digest(a.dataset_verification/'verification.json'),'selected':{s:[{'image_sha256':x['image_sha256'],'label_sha256':x['label_sha256']} for x in rows] for s,rows in chosen.items()},'training_args':args,'ultralytics':ultralytics_version,'torch':torch.__version__,'runner_sha256':digest(Path(__file__)),'adaptation_protocol_sha256':digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json')},indent=2)+'\n')
 base.train(**args)
 results=out/'train'/'results.csv';last=out/'train'/'weights'/'last.pt'
 if not results.is_file() or not last.is_file():raise ValueError('Smoke run missing outputs')
 with results.open(newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=1 or not rows[0]:raise ValueError('Expected one smoke epoch')
 for name,value in rows[0].items():
  if value and not math.isfinite(float(value)):raise ValueError('Nonfinite smoke metric: '+name)
 trained=YOLO(str(last))
 if trained.task!='detect' or len(trained.names)!=1 or trained.names[0]!='polyp' or torch.equal(before,trained.model.state_dict()['model.0.conv.weight'].detach().cpu()):raise ValueError('Smoke training did not update the detector')
 (out/'completion.json').write_text(json.dumps({'passed':True,'setup_sha256':digest(out/'setup.json'),'results_sha256':digest(results),'checkpoint_sha256':digest(last),'runner_sha256':digest(Path(__file__)),'scope':'Engineering-only smoke checkpoint, not a research candidate.'},indent=2)+'\n');print('One-epoch MPS smoke training passed; checkpoint is engineering only.')
if __name__=='__main__':main()

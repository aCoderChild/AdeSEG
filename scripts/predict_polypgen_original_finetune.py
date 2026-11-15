#!/usr/bin/env python3
"""Run one detector pass at saved initialization times; never read labels."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT,crop_viewport
from scripts.memory_token_io import digest,validate_output
from scripts.run_prompt_feature_refinement import detect_box

PROTOCOL=ROOT/'protocols/polypgen_original_detector_finetune_v1.json'
def read(p):return json.loads(p.read_text())
def param_hash(model):
 h=hashlib.sha256()
 for key,t in sorted(model.state_dict().items()):
  h.update(key.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('checkpoint','preflight','crop-reference','gate-verification','training-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();manifest=read(a.preflight/'manifest.json');pre=read(a.preflight/'completion.json');crop=read(a.crop_reference/'completion.json')
 if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py') or not crop['passed'] or crop['prompt_sha256']!=digest(a.crop_reference/'prompt.json') or crop['setup_sha256']!=digest(a.crop_reference/'setup.json'):raise ValueError('Frame or viewport provenance changed')
 if a.mode=='gate':
  if digest(a.checkpoint)!=digest(ROOT/'checkpoints/polypgen_yolov8n.pt'):raise ValueError('Engineering gate requires original detector')
  names=('seq3',)
 else:
  gate=read(a.gate_verification/'verification.json');training=read(a.training_verification/'verification.json')
  if not gate['passed'] or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_gate.py') or gate['runner_sha256']!=digest(Path(__file__)) or gate['preflight_sha256']!=digest(a.preflight/'manifest.json') or gate['crop_sha256']!=digest(a.crop_reference/'prompt.json') or gate['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Detector engineering gate changed')
  if not training['passed'] or training['best_sha256']!=digest(a.checkpoint) or training['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_detector_finetune.py'):raise ValueError('Trained detector verification missing')
  names=('seq8','seq11','seq20','seq23')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};cases=[]
 for seq in names:
  meta=manifest['sequences'][seq];source=Path(meta['source'])/seq;path=source/'prompt.json';original=read(path)
  if digest(path)!=meta['prompt_sha256']:raise ValueError('Archived original prompt changed')
  if seq=='seq23':prompt=read(a.crop_reference/'prompt.json');i=prompt['first_detection_index'];original_box=prompt['box_viewport_xyxy']
  else:prompt=original;i=prompt['first_detection_index'];original_box=prompt['box']
  if i is None or not 0<=i<len(videos[seq]) or original_box is None:raise ValueError('Missing original initialization')
  path=videos[seq][i].image_path;cases.append({'sequence':seq,'frame':i,'image_sha256':digest(path),'original_box_viewport':original_box,'original_confidence':prompt['confidence'],'viewport':seq=='seq23'})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config');os.environ['WANDB_MODE']='disabled';torch.set_num_threads(2)
 from ultralytics import YOLO,__version__ as version
 detector=YOLO(str(a.checkpoint));detector.model.eval().float().fuse(verbose=False)
 if detector.task!='detect' or detector.names!={0:'polyp'}:raise ValueError('Expected trained one-class detector')
 before=param_hash(detector.model);events=[]
 with torch.inference_mode():
  for c in cases:
   path=videos[c['sequence']][c['frame']].image_path
   if digest(path)!=c['image_sha256']:raise ValueError('Image changed')
   with Image.open(path) as im:image=np.asarray(im.convert('RGB'))
   if c['viewport']:image=crop_viewport(image)
   box,confidence=detect_box(detector,image)
   if a.mode=='gate' and (box is None or not np.allclose(box,c['original_box_viewport'],rtol=0,atol=1e-5) or abs(confidence-c['original_confidence'])>1e-6):raise ValueError('Original YOLO first-frame prediction differs')
   full=box
   if box is not None and c['viewport']:
    x0,y0,_,_=VIEWPORT;full=[box[0]+x0,box[1]+y0,box[2]+x0,box[3]+y0]
   events.append({**c,'candidate_box_viewport':box,'candidate_box_full':full,'candidate_confidence':confidence})
 if before!=param_hash(detector.model):raise ValueError('Detector weights changed during inference')
 result={'passed':True,'mode':a.mode,'events':events,'checkpoint_sha256':digest(a.checkpoint),'preflight_sha256':digest(a.preflight/'manifest.json'),'crop_sha256':digest(a.crop_reference/'prompt.json'),'protocol_sha256':digest(PROTOCOL),'runner_sha256':digest(Path(__file__)),'detector_helper_sha256':digest(ROOT/'scripts/run_prompt_feature_refinement.py'),'model_state_sha256':before,'ultralytics':version,'torch':str(torch.__version__),'scope':'One fixed saved initialization frame per sequence; no target labels, later detector search, or video propagation.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'frames':len(events),'detected':sum(e['candidate_box_viewport'] is not None for e in events)}))
if __name__=='__main__':main()

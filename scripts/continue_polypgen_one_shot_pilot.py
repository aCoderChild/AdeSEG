#!/usr/bin/env python3
"""Continue verified one-shot detector evaluation when its training finishes."""
import json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import validate_output

BASE=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
P=validate_output(BASE/'lean_one_shot_detector_pilot_v1');TRAIN=P/'training_attempt01';CHECK=P/'prompt_check_v1';CHECK.mkdir(exist_ok=True)
RAW=BASE/'lean_token_window_candidate_v1';OLD=BASE/'lean_fixed_box_oracle_diagnostic_v1';PRE=BASE/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03';CROP=BASE/'lean_seq23_no_inset_v1/full_attempt02';PRIOR=BASE/'lean_priority_prompt_quality_v1';DATA=P/'dataset_attempt02';DV=P/'dataset_verification_attempt02';SV=P/'smoke_verification_attempt01';PRETRAIN=P/'assets/yolov8n_coco_v8.2.0.pt';SAM=BASE/'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt';PY=str(ROOT/'.venv/bin/python')
status_path=P/'evaluation_status.json';env=os.environ.copy();env['YOLO_CONFIG_DIR']=str(P/'ultralytics_config');env['WANDB_MODE']='disabled'
def stamp(stage,**extra):
 status_path.write_text(json.dumps({'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def run(stage,args):
 stamp(stage)
 with (CHECK/(stage+'.log')).open('w') as log:subprocess.run([PY,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
 stamp('waiting_for_training')
 while True:
  state=json.loads((TRAIN/'status.json').read_text());state_name=state['status']
  if state_name=='completed':break
  if state_name!='running':raise RuntimeError('Training is not running or completed: '+state_name)
  try:os.kill(int(state['pid']),0)
  except ProcessLookupError as exc:raise RuntimeError('Training process disappeared without completion') from exc
  time.sleep(10)
 tv=P/'training_verification_attempt01'
 run('training_verification',['scripts/verify_polypgen_one_shot_training.py','--training',TRAIN,'--dataset',DATA,'--dataset-verification',DV,'--smoke-verification',SV,'--pretrained',PRETRAIN,'--output',tv])
 best=TRAIN/'train/weights/best.pt';dg=CHECK/'gate_verification_attempt01'
 pred=[]
 for i in (1,2):
  target=CHECK/f'candidate_attempt0{i}';pred.append(target)
  run(f'detector_candidate_attempt0{i}',['scripts/predict_polypgen_one_shot_detector.py','--mode','candidate','--checkpoint',best,'--preflight',PRE,'--crop-reference',CROP,'--gate-verification',dg,'--training-verification',tv,'--output',target])
 pv=CHECK/'candidate_verification_attempt01'
 run('detector_candidate_verification',['scripts/verify_polypgen_one_shot_predictions.py','--first',pred[0],'--second',pred[1],'--checkpoint',best,'--preflight',PRE,'--crop-reference',CROP,'--training-verification',tv,'--gate-verification',dg,'--output',pv])
 masks=[];mg=CHECK/'mask_gate_verification_attempt02'
 for i in (1,2):
  target=CHECK/f'masks_attempt0{i}';masks.append(target)
  run(f'mask_candidate_attempt0{i}',['scripts/predict_polypgen_one_shot_masks.py','--mode','candidate','--checkpoint',SAM,'--checkpoint-verification',RAW/'gate_verification_attempt01','--fixed-box-gate',OLD/'gate_verification_attempt01','--fixed-box-reference',OLD/'gate_attempt01','--predictions',pred[0],'--prediction-verification',pv,'--mask-gate',mg,'--output',target])
 mv=CHECK/'mask_verification_attempt01'
 run('mask_candidate_verification',['scripts/verify_polypgen_one_shot_masks.py','--first',masks[0],'--second',masks[1],'--predictions',pred[0],'--prediction-verification',pv,'--mask-gate',mg,'--checkpoint',SAM,'--output',mv])
 oracle=[]
 for i in (1,2):
  target=CHECK/f'oracle_attempt0{i}';oracle.append(target)
  run(f'oracle_attempt0{i}',['scripts/predict_polypgen_one_shot_oracle.py','--checkpoint',SAM,'--checkpoint-verification',RAW/'gate_verification_attempt01','--predictions',pred[0],'--prediction-verification',pv,'--masks',masks[0],'--mask-verification',mv,'--preflight',PRE,'--crop-reference',CROP,'--output',target])
 ov=CHECK/'oracle_verification_attempt01'
 run('oracle_verification',['scripts/verify_polypgen_one_shot_oracle.py','--first',oracle[0],'--second',oracle[1],'--predictions',pred[0],'--masks',masks[0],'--mask-verification',mv,'--output',ov])
 ba=CHECK/'box_audit_attempt01';bv=CHECK/'box_verification_attempt01'
 run('box_audit',['scripts/audit_polypgen_one_shot_boxes.py','--predictions',pred[0],'--verification',pv,'--preflight',PRE,'--crop-reference',CROP,'--output',ba])
 run('box_audit_verification',['scripts/verify_polypgen_one_shot_boxes.py','--audit',ba,'--predictions',pred[0],'--output',bv])
 ma=CHECK/'mask_audit_attempt01';mav=CHECK/'mask_audit_verification_attempt01'
 run('mask_audit',['scripts/audit_polypgen_one_shot_masks.py','--predictions',pred[0],'--box-audit',ba,'--box-verification',bv,'--masks',masks[0],'--mask-verification',mv,'--oracle',oracle[0],'--oracle-verification',ov,'--preflight',PRE,'--crop-reference',CROP,'--priority-audit',PRIOR/'audit_attempt01','--priority-verification',PRIOR/'verification_attempt01','--output',ma])
 run('mask_audit_verification',['scripts/verify_polypgen_one_shot_mask_scores.py','--audit',ma,'--masks',masks[0],'--oracle',oracle[0],'--preflight',PRE,'--crop-reference',CROP,'--priority-audit',PRIOR/'audit_attempt01','--output',mav])
 stamp('completed',box_report=str(ba/'report.md'),mask_report=str(ma/'report.md'))
 print('Training and all fixed-frame detector, mask and oracle audits completed.')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  stamp('failed',error=type(e).__name__,message=str(e),traceback=traceback.format_exc());raise

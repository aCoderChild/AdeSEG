#!/usr/bin/env python3
"""Label-free matched detector predictions on fixed C6 still images."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_prompt_feature_refinement import detect_box
from scripts.predict_polypgen_one_shot_detector import param_hash

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('inputs','original','candidate','training-verification','original-gate','candidate-verification','gate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();src=read(a.inputs/'inputs.json');done=read(a.inputs/'completion.json');tv=read(a.training_verification/'verification.json');og=read(a.original_gate/'verification.json');cv=read(a.candidate_verification/'verification.json')
 if not src['passed'] or not done['passed'] or done['inputs_sha256']!=digest(a.inputs/'inputs.json') or src['runner_sha256']!=digest(ROOT/'scripts/prepare_polypgen_c6_detector_inputs.py') or src['protocol_sha256']!=digest(ROOT/'protocols/polypgen_c6_detector_comparison_v1.json') or not tv['passed'] or src['training_verification_sha256']!=digest(a.training_verification/'verification.json') or src['dataset_sha256']!=tv['dataset_sha256'] or tv['best_sha256']!=digest(a.candidate) or tv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_training.py') or not og['passed'] or og['checkpoint_sha256']!=digest(a.original) or og['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_detector_gate.py') or not cv['passed'] or cv['checkpoint_sha256']!=digest(a.candidate) or cv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_predictions.py'):raise ValueError('Inputs or verified models changed')
 if a.mode=='candidate':
  gate=read(a.gate_verification/'verification.json')
  if not gate['passed'] or gate['inputs_sha256']!=digest(a.inputs/'inputs.json') or gate['original_sha256']!=digest(a.original) or gate['candidate_sha256']!=digest(a.candidate) or gate['runner_sha256']!=digest(Path(__file__)) or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_c6_detector_gate.py'):raise ValueError('C6 engineering gate changed')
 rows=src['inputs'][:2] if a.mode=='gate' else src['inputs']
 if len(src['inputs'])!=88 or [r['id'] for r in src['inputs']]!=list(range(88)):raise ValueError('Input manifest scope differs')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config');os.environ['WANDB_MODE']='disabled';torch.set_num_threads(2)
 from ultralytics import YOLO,__version__ as version
 models={'original':YOLO(str(a.original)),'candidate':YOLO(str(a.candidate))}
 for m in models.values():
  m.model.eval().float().fuse(verbose=False)
  if m.task!='detect' or m.names!={0:'polyp'}:raise ValueError('Unexpected detector')
 before={k:param_hash(m.model) for k,m in models.items()};events=[]
 with torch.inference_mode():
  for row in rows:
   path=Path(row['image'])
   if path.is_symlink() or digest(path)!=row['image_sha256']:raise ValueError('C6 image changed')
   with Image.open(path) as im:image=np.asarray(im.convert('RGB'))
   if image.shape[:2]!=(row['height'],row['width']):raise ValueError('Image dimensions changed')
   boxes={}
   for name,model in models.items():
    box,confidence=detect_box(model,image);boxes[name]={'box':box,'confidence':confidence}
   events.append({'id':row['id'],'image_sha256':row['image_sha256'],'boxes':boxes})
 if before!={k:param_hash(m.model) for k,m in models.items()}:raise ValueError('Detector weights changed')
 result={'passed':True,'mode':a.mode,'events':events,'model_state_sha256':before,'inputs_sha256':digest(a.inputs/'inputs.json'),'original_sha256':digest(a.original),'candidate_sha256':digest(a.candidate),'training_verification_sha256':digest(a.training_verification/'verification.json'),'protocol_sha256':digest(ROOT/'protocols/polypgen_c6_detector_comparison_v1.json'),'helper_sha256':digest(ROOT/'scripts/run_prompt_feature_refinement.py'),'runner_sha256':digest(Path(__file__)),'torch':str(torch.__version__),'ultralytics':version,'scope':'One prediction per detector per still image; no labels or video propagation.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'images':len(events)}))
if __name__=='__main__':main()

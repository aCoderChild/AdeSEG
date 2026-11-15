#!/usr/bin/env python3
"""Score frozen image-level original, candidate and oracle prompt masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def metrics(pred,gt):
 n=int(pred.sum())+int(gt.sum());tp=int((pred&gt).sum());present=bool(gt.any())
 return {'dice':2*tp/n if present and n else None,'precision':tp/int(pred.sum()) if present and pred.any() else (0. if present else None),'recall':tp/int(gt.sum()) if present else None,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
def main():
 p=argparse.ArgumentParser()
 for k in ('predictions','box-audit','box-verification','masks','mask-verification','oracle','oracle-verification','preflight','crop-reference','priority-audit','priority-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();pred=read(a.predictions/'completion.json');bd=read(a.box_audit/'results.json');bv=read(a.box_verification/'verification.json');mask=read(a.masks/'completion.json');mv=read(a.mask_verification/'verification.json');oracle=read(a.oracle/'completion.json');ov=read(a.oracle_verification/'verification.json');pre=read(a.preflight/'manifest.json');crop=read(a.crop_reference/'completion.json')
 if not all(x['passed'] for x in (pred,bv,mask,mv,oracle,ov,crop)) or bv['audit_sha256']!=digest(a.box_audit/'results.json') or bv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_boxes.py') or mv['mask_receipt_sha256']!=digest(a.masks/'completion.json') or mv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_masks.py') or ov['oracle_sha256']!=digest(a.oracle/'completion.json') or ov['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_oracle.py') or mask['predictions_sha256']!=digest(a.predictions/'completion.json') or oracle['predictions_sha256']!=digest(a.predictions/'completion.json') or oracle['mask_receipt_sha256']!=mv['mask_receipt_sha256'] or crop['prompt_sha256']!=digest(a.crop_reference/'prompt.json'):raise ValueError('Frozen inputs or verification changed')
 pv=read(a.priority_verification/'verification.json');prior=read(a.priority_audit/'results.json')
 if not pv['passed'] or pv['audit_results_sha256']!=digest(a.priority_audit/'results.json') or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_priority_initial_prompts.py'):raise ValueError('Saved prompt audit changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results=[]
 for e,b,m,o in zip(pred['events'],bd['results'],mask['events'],oracle['events'],strict=True):
  seq=e['sequence'];i=e['frame'];frame=videos[seq][i]
  if any((x['sequence'],x['frame'])!=(seq,i) for x in (b,m,o)) or digest(frame.mask_path)!=b['label_sha256'] or b['label_sha256']!=o['label_sha256'] or digest(frame.image_path)!=e['image_sha256']:raise ValueError('Frame or labels changed')
  gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any());source=Path(pre['sequences'][seq]['source'])/seq
  original=a.crop_reference/'prompt.png' if e['viewport'] else source/'prompt_refined.png'
  if not e['viewport'] and digest(original)!=pre['sequences'][seq]['prompt_mask_sha256']:raise ValueError('Original Lean prompt changed')
  old=next(x for x in prior['results'] if x['sequence']==seq)
  if old['frame']!=i or old['prompt_mask_sha256']!=digest(original) or old['label_sha256']!=b['label_sha256']:raise ValueError('Previously verified initial prompt changed')
  saved=np.asarray(Image.open(original))>0
  if saved.shape!=gt.shape:raise ValueError('Original prompt geometry differs')
  scores={'saved_lean_prompt':metrics(saved,gt)}
  for name,sha in m['masks'].items():
   path=a.masks/f'{seq}_{i:06d}_{name}.png'
   if digest(path)!=sha:raise ValueError('Mask changed')
   result=np.asarray(Image.open(path))>0
   if result.shape!=gt.shape:raise ValueError('Mask geometry differs')
   scores[name]=metrics(result,gt)
  if o['mask_sha256'] is not None:
   path=a.oracle/f'{seq}_{i:06d}_oracle_box.png'
   if digest(path)!=o['mask_sha256']:raise ValueError('Oracle mask changed')
   scores['oracle_box']=metrics(np.asarray(Image.open(path))>0,gt)
  else:scores['oracle_box']=None
  results.append({'sequence':seq,'frame':i,'present':present,'label_sha256':b['label_sha256'],'original_box_detected':b['original']['detected'],'candidate_box_detected':b['candidate']['detected'],'points':m['points'],'scores':scores})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Image-level prompt masks on fixed initialization frames; original stored Lean prompt and SAM2 image-predictor arms have distinct preprocessing.','results':results,'box_audit_sha256':digest(a.box_audit/'results.json'),'mask_receipt_sha256':digest(a.masks/'completion.json'),'oracle_sha256':digest(a.oracle/'completion.json'),'runner_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# One-shot prompt masks on fixed initialization frames','','The saved Lean prompt uses the original video predictor and earlier-feature points. The other masks use a common SAM2 image predictor with box-only or box-plus-early-feature-point inputs; its image preprocessing differs from the video predictor. A missing candidate box produces an empty candidate mask, not a later detector search. Oracle boxes use target labels and are not available at inference. Empty-label false alarms are reported separately; labels may be incomplete.','','| Sequence | Frame | Label | Saved Lean prompt Dice | Original box / points Dice | Candidate box / points Dice | Oracle box Dice | Empty false alarm: saved / original box / candidate box |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
 for r in results:
  s=r['scores'];fmt=lambda item:'—' if item is None or item['dice'] is None else f"{item['dice']:.4f}";fa='—' if r['present'] else ' / '.join('yes' if s[k]['false_alarm'] else 'no' for k in ('saved_lean_prompt','original_box','candidate_box'))
  lines.append(f"| {r['sequence']} | {r['frame']} | {'present' if r['present'] else 'empty'} | {fmt(s['saved_lean_prompt'])} | {fmt(s['original_box'])} / {fmt(s['original_points'])} | {fmt(s['candidate_box'])} / {fmt(s['candidate_points'])} | {fmt(s['oracle_box'])} | {fa} |")
 lines+=['','These four previously inspected frames do not establish full-video performance, patient-level independence or an achievable oracle bound. The new detector checkpoint was selected on C6 validation; its initial COCO weights and training data were not the original PolypGen detector checkpoint.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

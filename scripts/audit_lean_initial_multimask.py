#!/usr/bin/env python3
"""Independently score saved native multimask initialization predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',type=Path,action='append',required=True)
 for x in ('merged','verification','gate','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();v=read(a.verification);merge=read(a.merged/'completion.json'); protocol=read(ROOT/'protocols/lean_initial_multimask_v1.json')
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_initial_multimask.py')
     or not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256'] or digest(a.gate/'records.json')!=v['records_sha256']): raise ValueError('Invalid input receipts')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};loc=read(a.merged/'locations.json');results={}
 for part in a.part:
  setup,done=read(part/'setup.json'),read(part/'completion.json');records=read(part/'records.json')
  if (setup['mode']!='cohort' or setup['identity']!=v['identity'] or not done['passed'] or digest(part/'records.json')!=done['records_sha256']
      or setup['sequences']!=[x['sequence'] for x in records]): raise ValueError('Partition invalid')
  for item in records:
   name=item['sequence']
   if name in results: raise ValueError('Duplicate sequence')
   folder=part/name
   if item!=read(folder/'record.json') or any(not read(folder/f'parameter_integrity_{c}.json')['unchanged'] for c in ('baseline','candidate')): raise ValueError('Invalid condition record')
   source=Path(loc[name]);oldrec=read(source/'records.json')[0];olddone=read(source/'completion.json');old_audit=read(source/'independent_audit.json');oldfolder=source/name
   if (not olddone['passed'] or not old_audit['passed'] or digest(source/'records.json')!=olddone['records_sha256']
       or old_audit['records_sha256']!=olddone['records_sha256'] or item['original_records_sha256']!=olddone['records_sha256']
       or digest(oldfolder/'frames.json')!=oldrec['frames_sha256'] or digest(oldfolder/'prompt.json')!=oldrec['prompt_sha256']): raise ValueError('Original source invalid')
   oldrows=read(oldfolder/'frames.json');prompt=read(oldfolder/'prompt.json');i=item['first_detection_index'];frame=videos[name][i]
   if (i!=prompt['first_detection_index'] or digest(frame.image_path)!=item['image_sha256'] or item['image_sha256']!=oldrows[i]['image_sha256']
       or digest(frame.mask_path)!=oldrows[i]['label_sha256']): raise ValueError('Input changed')
   truth=load_label_mask(frame.mask_path,[1])==1;present=bool(truth.any()); scores={}
   for cond in ('baseline','candidate'):
    data=item[cond]
    for suffix in ('provisional','refined'):
     path=folder/f'{cond}_{suffix}.png'
     if digest(path)!=data[suffix+'_sha256']: raise ValueError('Mask hash changed')
    if not read(folder/f'parameter_integrity_{cond}.json')['unchanged']: raise ValueError('Weight changed')
    pred=np.asarray(Image.open(folder/f'{cond}_refined.png'))>0
    if pred.shape!=truth.shape: raise ValueError('Mask dimensions changed')
    den=int(pred.sum()+truth.sum());scores[cond]={'dice':2*int((pred&truth).sum())/den if den else 1., 'false_alarm':bool(pred.any()) if not present else None, 'pred_pixels':int(pred.sum())}
   if not np.array_equal(np.asarray(Image.open(folder/'baseline_refined.png'))>0,np.asarray(Image.open(oldfolder/'prompt_refined.png'))>0):raise ValueError('Baseline mask differs')
   events=item['candidate']['decoder_events']
   if not events or any(len(x['predicted_iou'])!=3 or x['selected']!=max(range(3),key=lambda j:x['predicted_iou'][j]) for x in events):raise ValueError('Candidate decoder invalid')
   if name=='seq3' and item!=read(a.gate/'records.json')[0]: raise ValueError('Cohort seq3 differs from engineering replay')
   box=np.asarray(prompt['box'],dtype=float)
   if present:
    yy,xx=np.nonzero(truth);gtbox=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
    inter=np.maximum(0,np.minimum(box[2:],gtbox[2:])-np.maximum(box[:2],gtbox[:2]));ia=float(np.prod(inter));ba=float(np.prod(box[2:]-box[:2]));ga=float(np.prod(gtbox[2:]-gtbox[:2]));box_iou=ia/(ba+ga-ia) if ba+ga-ia else 0.
   else:box_iou=None
   results[name]={'frame':i,'present':present,'gt_pixels':int(truth.sum()),'box_iou':box_iou,'baseline':scores['baseline'],'candidate':scores['candidate'],
    'same_refined_mask':item['baseline']['refined_sha256']==item['candidate']['refined_sha256'],'same_points':item['baseline']['points']==item['candidate']['points'],
    'candidate_decoder_events':events}
 if set(results)!=set(protocol['cohort']):raise ValueError('Incomplete frozen cohort')
 rows=list(results.values());present=[x for x in rows if x['present']];empty=[x for x in rows if not x['present']]
 deltas=np.array([x['candidate']['dice']-x['baseline']['dice'] for x in present]);rng=np.random.default_rng(20261114);boot=rng.choice(deltas,(10000,len(deltas)),replace=True).mean(axis=1)
 summary={'scope':'One automatically selected initialization frame per video. Ground truth used only after all prompt masks are saved. This is not video propagation and is not generalization evidence.',
  'sequences':results,'present_frames':len(present),'empty_frames':len(empty),'baseline_present_dice':float(np.mean([x['baseline']['dice'] for x in present])),
  'candidate_present_dice':float(np.mean([x['candidate']['dice'] for x in present])),'paired_sequence_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
  'baseline_empty_false_alarms':sum(x['baseline']['false_alarm'] for x in empty),'candidate_empty_false_alarms':sum(x['candidate']['false_alarm'] for x in empty),
  'changed_masks':sum(not x['same_refined_mask'] for x in rows),'changed_points':sum(not x['same_points'] for x in rows),
  'present_wins':int(np.sum(deltas>1e-12)),'present_losses':int(np.sum(deltas< -1e-12)),'present_ties':int(np.sum(np.abs(deltas)<=1e-12))}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Native SAM2 multimask: first automatic prompt study','',summary['scope'],'','| Sequence | First detection | Label | Box IoU | Baseline Dice / empty alarm | Multimask Dice / empty alarm | Points changed |', '| --- | ---: | --- | ---: | ---: | ---: | --- |']
 for name,x in results.items():
  if x['present']:
   box=f"{x['box_iou']:.3f}";old=f"{x['baseline']['dice']:.3f}";new=f"{x['candidate']['dice']:.3f}"
  else:
   box='—';old=str(x['baseline']['false_alarm']);new=str(x['candidate']['false_alarm'])
  lines.append(f"| {name} | {x['frame']} | {'present' if x['present'] else 'empty'} | {box} | {old} | {new} | {not x['same_points']} |")
 lines += ['',f"On {len(present)} labeled-present prompt frames, mean Dice: {summary['baseline_present_dice']:.4f} baseline, {summary['candidate_present_dice']:.4f} multimask. Paired sequence-bootstrap 95% interval for difference: {summary['paired_sequence_bootstrap_ci95']}. On {len(empty)} labeled-empty prompt frames, false alarms: {summary['baseline_empty_false_alarms']} / {summary['candidate_empty_false_alarms']}.",
  f"Refined masks changed on {summary['changed_masks']}/21 sequences, feature points on {summary['changed_points']}/21. Present-frame wins/losses/ties: {summary['present_wins']}/{summary['present_losses']}/{summary['present_ties']}.",
  'The box, frame and original no-object decision are unchanged. The author decoder selects by predicted IoU, not ground truth; an incorrect box or prompt on an empty frame can still fail. One-frame Dice cannot predict full-video behavior and this inspected cohort is not independent.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='sequences'},indent=2))
if __name__=='__main__':main()

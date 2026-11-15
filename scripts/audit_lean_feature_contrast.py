#!/usr/bin/env python3
"""Score saved Lean feature rankings after exact reference-video replays."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.weighted_rank_auc import weighted_rank_auc
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def resize(mask,shape):
 x=torch.from_numpy(np.asarray(mask,dtype=np.float32).copy())[None,None]
 return F.interpolate(x,size=shape,mode='area')[0,0].numpy()
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',type=Path,action='append',required=True)
 for name in ('merged','indexed-root','verification','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();v=read(a.verification);protocol=read(ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json')
 if not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_feature_contrast.py') or v['observer_sha256']!=digest(ROOT/'scripts/observe_lean_feature_contrast.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json'):
  raise ValueError('Engineering verification changed')
 loc=read(a.merged/'locations.json');merge=read(a.merged/'completion.json')
 if not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256']:raise ValueError('Merge changed')
 indexed=read(a.indexed_root/'audited_cohort_attempt01/completion.json')
 if not indexed['passed'] or digest(a.indexed_root/'audited_cohort_attempt01/results.json')!=indexed['results_sha256']:raise ValueError('Indexed audit changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results={}
 for part in a.part:
  setup,done=read(part/'setup.json'),read(part/'completion.json')
  if not done['passed'] or done['sequences']!=setup['sequences'] or setup['verification_sha256']!=digest(a.verification) or setup['runner_sha256']!=v['observer_sha256']:raise ValueError('Partition invalid')
  for name in setup['sequences']:
   if name in results:raise ValueError('Duplicate sequence')
   run=part/name;o=read(run/'contrast_observation.json');rec=read(run/'records.json');run_done=read(run/'completion.json')
   if (not o['passed'] or o['sequence']!=name or o['label_free_gate'] or o['source_sha256']!=v['observer_sha256']
       or o['protocol_sha256']!=v['protocol_sha256'] or not run_done['passed'] or digest(run/'records.json')!=run_done['records_sha256']
       or not read(run/'parameter_integrity_after.json')['unchanged']):raise ValueError('Observation invalid')
   refs=[d/name for d in (a.indexed_root/'part_a_attempt01',a.indexed_root/'part_b_attempt01') if (d/name/'completion.json').exists()]
   if len(refs)!=1:raise ValueError('Reference missing')
   ref=refs[0];rr=read(ref/'records.json');rd=read(ref/'completion.json')
   if not rd['passed'] or digest(ref/'records.json')!=rd['records_sha256'] or o['reference_records_sha256']!=rd['records_sha256'] or o['replay_records_sha256']!=digest(run/'records.json'):raise ValueError('Reference changed')
   for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256','frames_sha256'):
    if rec[key]!=rr[key]:raise ValueError('Observer changed reference')
   for i,h in enumerate(rec['mask_sha256']):
    if digest(run/f'{i:06d}.png')!=h or digest(ref/f'{i:06d}.png')!=h:raise ValueError('Video mask changed')
   source=Path(loc[name]);old=read(source/'records.json')[0];olddone=read(source/'completion.json');oldfolder=source/name
   if digest(source/'records.json')!=olddone['records_sha256'] or digest(oldfolder/'frames.json')!=old['frames_sha256'] or digest(oldfolder/'prompt.json')!=old['prompt_sha256']:raise ValueError('Source input changed')
   oldrows=read(oldfolder/'frames.json');prompt=read(oldfolder/'prompt.json');frames=videos[name];first=prompt['first_detection_index'];targets=sorted(set(first+round((len(frames)-1-first)*i/4) for i in range(5)))
   if o['target_frames']!=targets or [x['frame'] for x in o['captures']]!=targets:raise ValueError('Frame schedule changed')
   rows=[]
   for cap in o['captures']:
    i=cap['frame'];frame=frames[i];path=run/'contrast'/f'{i:06d}.npz'
    if digest(path)!=cap['map_sha256'] or digest(frame.image_path)!=oldrows[i]['image_sha256'] or digest(frame.mask_path)!=oldrows[i]['label_sha256']:raise ValueError('Feature or label input changed')
    gt=load_label_mask(frame.mask_path,[1])==1;pred=np.asarray(Image.open(ref/f'{i:06d}.png'))>0
    if gt.shape!=pred.shape:raise ValueError('Mask shape changed')
    row={'frame':i,'present':bool(gt.any()),'gt_pixels':int(gt.sum()),'pred_pixels':int(pred.sum()),'levels':{}}
    with np.load(path,allow_pickle=False) as arrays:
     if set(arrays.files)!=set(cap['arrays']):raise ValueError('Maps changed')
     for key,meta in cap['arrays'].items():
      arr=arrays[key]
      if list(arr.shape)!=meta['shape'] or hashlib.sha256(arr.tobytes()).hexdigest()!=meta['sha256'] or not np.isfinite(arr).all():raise ValueError('Feature score changed')
     for level in range(3):
      keys=[f'level{level}_positive',f'level{level}_contrast']
      if not all(k in arrays for k in keys):row['levels'][str(level)]=None;continue
      shape=arrays[keys[0]].shape;weights={'global':(resize(gt,shape),resize(~gt,shape)), 'within_prediction':(resize(gt&pred,shape),resize((~gt)&pred,shape))}
      row['levels'][str(level)]={scope:{kind:weighted_rank_auc(arrays[f'level{level}_{kind}'],*weights[scope]) for kind in ('positive','contrast')} for scope in weights}
    rows.append(row)
   results[name]={'first_detection_index':first,'prototype_status':o['prototype_status'],'frames':rows}
 if set(results)!=set(protocol['cohort']):raise ValueError('Diagnostic cohort incomplete')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);summary={'scope':'Read-only scores from exact automatic indexed Lean replays. AUC is soft area-weighted per feature cell; labels are evaluation only. Frames are fixed-spaced after the automatic prompt, not independently sampled.','sequences':results}
 (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Frozen feature contrast: exploratory rank diagnostic','',summary['scope'],'','Each cell contributes its fractional polyp and background pixel occupancy. The within-prediction AUC compares true and false predicted foreground; it is undefined if either is absent. Empty-label frames have no positive AUC. No threshold, gating, or propagation change was tested.','','| Sequence | Frame | Label | Predicted pixels | FPN level | Whole-frame AUC positive / contrast | Within-prediction AUC positive / contrast |','| --- | ---: | --- | ---: | ---: | ---: | ---: |']
 fmt=lambda x:'—' if x is None else f'{x:.3f}'
 for name,record in results.items():
  for row in record['frames']:
   for level,z in row['levels'].items():
    if z is None:whole=inside='unavailable'
    else:whole=f"{fmt(z['global']['positive'])} / {fmt(z['global']['contrast'])}";inside=f"{fmt(z['within_prediction']['positive'])} / {fmt(z['within_prediction']['contrast'])}"
    lines.append(f"| {name} | {row['frame']} | {'present' if row['present'] else 'empty'} | {row['pred_pixels']} | {level} | {whole} | {inside} |")
 lines += ['', 'These are descriptive observations on four previously inspected sequences. The first automatic prompt may be incorrect or labeled empty, and the later labels may be incomplete. AUC measures feature ranking, not attainable mask Dice or video improvement.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'auc_sha256':digest(ROOT/'evaluation/weighted_rank_auc.py'),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
 print('Audited',len(results),'sequences and',sum(len(x['frames']) for x in results.values()),'fixed frames')
if __name__=='__main__':main()

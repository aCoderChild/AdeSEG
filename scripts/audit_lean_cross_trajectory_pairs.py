#!/usr/bin/env python3
"""Score frozen cross-trajectory swap masks after independent replay checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def metric(pred,gt):
 n=int(pred.sum())+int(gt.sum());present=bool(gt.any())
 return {'dice':2*int((pred&gt).sum())/n if present else None,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
def main():
 p=argparse.ArgumentParser()
 for k in ('replay','verification','raw-capture','fused-capture','crop-reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');raw=read(a.raw_capture/'raw_two_capture_receipt.json');fused=read(a.fused_capture/'fusion_capture_receipt.json');crop=read(a.crop_reference/'completion.json')
 if not all(x['passed'] for x in (r,v,raw,fused,crop)) or r['mode']!='candidate' or r['raw_capture_sha256']!=digest(a.raw_capture/'raw_two_capture_receipt.json') or r['fused_capture_sha256']!=digest(a.fused_capture/'fusion_capture_receipt.json') or v['runner_sha256']!=digest(ROOT/'scripts/replay_lean_cross_trajectory_pairs.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_cross_trajectory_pair_repeats.py') or v['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_pairs.py') or crop['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen swap inputs changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos['seq23'];refs=read(a.crop_reference/'frames.json');rows=[]
 for e in r['events']:
  i=e['frame'];f=frames[i]
  if digest(f.image_path)!=refs[i]['image_sha256'] or digest(f.mask_path)!=refs[i]['label_sha256']:raise ValueError('Input or label changed')
  gt=load_label_mask(f.mask_path,[1])==1
  if gt[:,:VIEWPORT[0]].any():raise ValueError('Annotated foreground outside viewport')
  masks={};scores={}
  for name,sha in e['masks'].items():
   path=a.replay/f'{i:06d}_{name}.png'
   if digest(path)!=sha:raise ValueError('Prediction changed')
   m=np.asarray(Image.open(path))>0
   if m.shape!=gt.shape:raise ValueError('Prediction geometry differs')
   masks[name]=m;scores[name]=metric(m,gt)
  changes={name:int(np.count_nonzero(mask^masks['raw_noop' if name.startswith('raw_') else 'fused_noop'])) for name,mask in masks.items() if not name.endswith('_noop')}
  rows.append({'frame':i,'present':bool(gt.any()),'label_sha256':refs[i]['label_sha256'],'total_keys':e['total_keys'],'pointer_keys':e['pointer_keys'],'scores':scores,'changed_pixels':changes})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Previously inspected same-frame memory swaps, no propagation or independent generalization.','replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__)),'frames':rows}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Sequence 23: post-hoc pairwise block swaps','', 'Index-matched slices are swapped in both directions across different trajectories. Source values and positions move together; the recipient prompt, current query, decoder inputs and pointer tail are retained. These selected local reads do not propagate and are not an independent efficacy test.', '', '| Frame | Label | Recipient no-op | First + middle | First + last | Middle + last |', '| ---: | --- | ---: | ---: | ---: | ---: |']
 for x in rows:
  for recipient,donor in (('fused','raw'),('raw','fused')):
   scores=x['scores'];fmt=lambda z: ('FA='+str(z['false_alarm'])) if z['dice'] is None else f"{z['dice']:.6f}"
   vals=[fmt(scores[f'{recipient}_noop'])]+[fmt(scores[f'{recipient}_with_{donor}_{k}']) for k in ('first_middle','first_last','middle_last')]
   lines.append(f"| {x['frame']} | {'present' if x['present'] else 'empty'}; {recipient} | "+' | '.join(vals)+' |')
 lines+=['', 'First=[4096,7168), middle=[7168,8192), last=[8192,12288). Both listed blocks were swapped. The fused middle block is the captured fused skeleton; the matching raw slice need not have the same source age. These blocks were selected after the whole-swap findings. The results cannot identify fusion or history length as the cause. Full changed-pixel counts are in results.json.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

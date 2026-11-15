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
 if not all(x['passed'] for x in (r,v,raw,fused,crop)) or r['mode']!='candidate' or r['raw_capture_sha256']!=digest(a.raw_capture/'raw_two_capture_receipt.json') or r['fused_capture_sha256']!=digest(a.fused_capture/'fusion_capture_receipt.json') or v['runner_sha256']!=digest(ROOT/'scripts/replay_lean_cross_trajectory_memory.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_cross_trajectory_repeats.py') or v['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_swap.py') or crop['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen swap inputs changed')
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
  changes={'raw_with_fused_vs_raw':int(np.count_nonzero(masks['raw_with_fused']^masks['raw_noop'])),'fused_with_raw_vs_fused':int(np.count_nonzero(masks['fused_with_raw']^masks['fused_noop']))}
  rows.append({'frame':i,'present':bool(gt.any()),'label_sha256':refs[i]['label_sha256'],'total_keys':e['total_keys'],'pointer_keys':e['pointer_keys'],'scores':scores,'changed_pixels':changes})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Previously inspected same-frame memory swaps, no propagation or independent generalization.','replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__)),'frames':rows}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Sequence 23: cross-trajectory same-read spatial-memory swap','','The two captured trajectories have the same automatic prompt, current image query and decoder image features. At each selected frame the entire nonconditioning spatial memory and its positions are exchanged at equal key count, while each recipient keeps its own prompt memory and object pointers. The swapped histories include different trajectory-derived values and potentially different source ages, so this is not an isolated fusion ablation. These previously inspected reads have no feedback into later frames.','','| Frame | Label | Spatial / pointer keys | Fused no-op | Fused with raw memory | Raw no-op | Raw with fused memory | Changed pixels: fused / raw |','| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for x in rows:
  s=x['scores'];fmt=lambda z:'—' if z['dice'] is None else f"{z['dice']:.6f}";change=x['changed_pixels']
  lines.append(f"| {x['frame']} | {'present' if x['present'] else 'empty'} | {x['total_keys']-x['pointer_keys']} / {x['pointer_keys']} | {fmt(s['fused_noop'])} | {fmt(s['fused_with_raw'])} | {fmt(s['raw_noop'])} | {fmt(s['raw_with_fused'])} | {change['fused_with_raw_vs_fused']} / {change['raw_with_fused_vs_raw']} |")
 for x in rows:
  if not x['present']:lines.append(f"\nFrame {x['frame']} labeled-empty false alarms (fused / fused with raw / raw / raw with fused): "+' / '.join(str(x['scores'][k]['false_alarm']) for k in ('fused_noop','fused_with_raw','raw_noop','raw_with_fused'))+'.')
 lines+=['','A positive local swap result cannot establish that fusion or history length caused the full-video difference; changed memory values, positions and past errors are entangled. The object-pointer tails remain from their respective original trajectories. Patient independence is unknown and labels may be incomplete.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

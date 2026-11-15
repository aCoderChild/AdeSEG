#!/usr/bin/env python3
"""Score frozen fixed-box and explicit oracle image predictions after inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output
from scripts.run_prompt_feature_refinement import gt_box,box_iou

def read(p):return json.loads(p.read_text())
def score(mask,gt):
 n=int(mask.sum())+int(gt.sum())
 return 2*int((mask&gt).sum())/n if n else 1.0

def main():
 p=argparse.ArgumentParser()
 for k in ('fixed-a','fixed-b','oracle-a','oracle-b','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();f=read(a.fixed_a/'completion.json');fr=read(a.fixed_b/'completion.json');o=read(a.oracle_a/'completion.json');orr=read(a.oracle_b/'completion.json')
 if f!=fr or o!=orr or not f['passed'] or not o['passed'] or len(f['events'])!=20 or len(o['events'])!=20:raise ValueError('Repeated inference differs')
 if o['runner_sha256']!=digest(ROOT/'scripts/run_fixed_box_oracle.py') or o['fixed_completion_sha256']!=digest(a.fixed_a/'completion.json') or o['fixed_repeat_sha256']!=digest(a.fixed_b/'completion.json'):raise ValueError('Oracle provenance changed')
 manifest=read(a.preflight/'manifest.json');videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results=[]
 for fe,oe in zip(f['events'],o['events']):
  seq=fe['sequence'];i=fe['frame'];frame=videos[seq][i];meta=manifest['sequences'][seq];folder=Path(meta['source'])/seq
  if (seq,i)!=(oe['sequence'],oe['frame']) or digest(folder/'frames.json')!=meta['frames_metadata_sha256'] or digest(frame.mask_path)!=oe['label_sha256'] or digest(frame.mask_path)!=read(folder/'frames.json')[i]['label_sha256']:raise ValueError('Input changed')
  gt=load_label_mask(frame.mask_path,[1])==1;view=gt
  if seq=='seq23':
   x0,y0,x1,y1=VIEWPORT;view=gt[y0:y1,x0:x1]
   if int(view.sum())!=int(gt.sum()):raise ValueError('Truth outside crop')
  box=gt_box(view)
  if box!=oe['oracle_box']:raise ValueError('Oracle box mismatch')
  predictions=[]
  for fixed in (a.fixed_a,a.fixed_b):
   path=fixed/f'{seq}_{i:06d}_fixed_initial.png'
   if digest(path)!=fe['masks']['fixed_initial']:raise ValueError('Fixed mask changed')
   predictions.append(np.asarray(Image.open(path))>0)
  if not np.array_equal(*predictions):raise ValueError('Fixed masks differ')
  mask=predictions[0];present=bool(gt.any());oracle=None
  if present:
   preds=[]
   for folder in (a.oracle_a,a.oracle_b):
    path=folder/f'{seq}_{i:06d}_oracle.png'
    if digest(path)!=oe['mask_sha256']:raise ValueError('Oracle mask changed')
    preds.append(np.asarray(Image.open(path))>0)
   if not np.array_equal(*preds):raise ValueError('Oracle masks differ')
   oracle=score(preds[0],gt)
  results.append({'sequence':seq,'frame':i,'present':present,'fixed_dice':score(mask,gt) if present else None,'fixed_false_alarm':bool(mask.any()) if not present else None,'oracle_dice':oracle,'fixed_box_iou':box_iou(fe['box'],box) if present else None})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 lines=['# Fixed saved-box versus explicit oracle box: SAM2 image diagnostic','','The same archived initial automatic box is applied to each target image without detector calls. This is not video propagation; a stale box may miss a moving target. The oracle uses target labels and is not deployable. Five post-initial frames were preselected per sequence. Seq23 uses a fixed, retrospectively selected inset-free viewport.','','| Sequence | Present frames | Fixed-box mean Dice | Oracle-box mean Dice | Mean fixed-box IoU | Fixed false alarms / empty |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for seq in ('seq8','seq11','seq20','seq23'):
  rows=[r for r in results if r['sequence']==seq];pos=[r for r in rows if r['present']];neg=[r for r in rows if not r['present']]
  mean=lambda k: f"{np.mean([r[k] for r in pos]):.4f}" if pos else '—'
  lines.append(f"| {seq} | {len(pos)} | {mean('fixed_dice')} | {mean('oracle_dice')} | {mean('fixed_box_iou')} | {sum(r['fixed_false_alarm'] for r in neg)} / {len(neg)} |")
 lines+=['','These frames and sequences were previously inspected; patient independence is unknown. Box oracle Dice is not a guaranteed upper bound, and this comparison does not establish that a better automatic box can be obtained or that the video method will improve.','']
 (out/'results.json').write_text(json.dumps({'results':results,'auditor_sha256':digest(Path(__file__))},indent=2)+'\n');(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

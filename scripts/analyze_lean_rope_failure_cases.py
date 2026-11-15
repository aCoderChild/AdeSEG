#!/usr/bin/env python3
"""Inspect changed predictions on selected audited PolypGen videos."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p): return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser();
 for x in ('audit','merged','historical','part-a','part-b','output'): p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args(); receipt=read(a.audit/'completion.json')
 if not receipt['passed'] or receipt['results_sha256'] != digest(a.audit/'results.json'): raise ValueError('Audit changed')
 results=read(a.audit/'results.json')['sequences']; loc=read(a.merged/'locations.json'); videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
 names=['seq8','seq11','seq13','seq15','seq17','seq20','seq23']; analysis={}
 for name in names:
  old=Path(loc[name])/name; candidate=next(p/name for p in (a.part_a,a.part_b) if (p/name/'completion.json').exists())
  old_rows=read(old/'frames.json'); new_rows=read(candidate/'frames.json'); frames=videos[name]
  target={x['frame'] for x in (json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()) if x['present']}
  sums={'original':[0,0,0], 'corrected':[0,0,0]}; changes=[]
  for i in sorted(target):
   frame=frames[i]; truth=load_label_mask(frame.mask_path,[1])==1
   masks=[]
   for path,row in ((old/f'{i:06d}.png',old_rows[i]),(candidate/f'{i:06d}.png',new_rows[i])):
    if digest(path) != row['mask_sha256']: raise ValueError(f'{name}/{i} mask changed')
    masks.append(np.asarray(Image.open(path))>0)
   for key,pred in zip(('original','corrected'),masks):
    sums[key][0]+=int((pred & truth).sum()); sums[key][1]+=int((pred & ~truth).sum()); sums[key][2]+=int((~pred & truth).sum())
   changes.append({'frame':i,'dice_change':new_rows[i]['dice']-old_rows[i]['dice'], 'original_dice':old_rows[i]['dice'],'corrected_dice':new_rows[i]['dice']})
  measures={}
  for key,(tp,fp,fn) in sums.items(): measures[key]={'foreground_precision':tp/(tp+fp) if tp+fp else None,'foreground_recall':tp/(tp+fn) if tp+fn else None,'true_positive_pixels':tp,'false_positive_pixels':fp,'false_negative_pixels':fn}
  analysis[name]={'measures_on_matched_present_frames':measures,'largest_decreases':sorted(changes,key=lambda x:x['dice_change'])[:5],
    'largest_increases':sorted(changes,key=lambda x:x['dice_change'],reverse=True)[:5], 'cohort':results[name]}
 out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
 (out/'analysis.json').write_text(json.dumps({'scope':'Selected exploratory videos; pooled pixel precision and recall use matched historical present frames. Descriptive diagnostics do not establish causes.','sequences':analysis},indent=2)+'\n')
 lines=['# Selected failure-case pixel analysis','','Pooled pixel precision and recall below use the historical matched present frames. They weight frames by pixel counts and complement, rather than replace, the equal-frame Dice scores.','', '| Sequence | Precision original / corrected | Recall original / corrected | Largest frame Dice decrease (frame, change) |', '| --- | ---: | ---: | ---: |']
 fmt=lambda x: 'undefined' if x is None else f'{x:.3f}'
 for name,r in analysis.items():
  o=r['measures_on_matched_present_frames']['original']; n=r['measures_on_matched_present_frames']['corrected']; worst=r['largest_decreases'][0]
  lines.append(f"| {name} | {fmt(o['foreground_precision'])} / {fmt(n['foreground_precision'])} | {fmt(o['foreground_recall'])} / {fmt(n['foreground_recall'])} | {worst['frame']}, {worst['dice_change']:+.3f} |")
 lines += ['', 'These comparisons describe mask changes, not causes. Sequence 20 had an empty-frame automatic initialization and sequence 23 initialized on the inset in the previously inspected prompts; the correction does not address detector localization.','']
 (out/'report.md').write_text('\n'.join(lines)); (out/'completion.json').write_text(json.dumps({'passed':True,'analysis_sha256':digest(out/'analysis.json'),'audit_sha256':digest(a.audit/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n'); print('\n'.join(lines))
if __name__=='__main__':main()

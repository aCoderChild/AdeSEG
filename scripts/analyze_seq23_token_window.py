#!/usr/bin/env python3
"""Describe where the audited seq23 token-window masks differ."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('audit','repeat-verification','baseline','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();d=read(a.audit/'completion.json');audit=read(a.audit/'results.json');rep=read(a.repeat_verification/'verification.json')
 if not d['passed'] or digest(a.audit/'results.json')!=d['results_sha256'] or d['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py') or not rep['passed'] or rep['source_sha256']!=digest(ROOT/'scripts/verify_seq23_token_window_repeat.py') or rep['sequence']!='seq23':raise ValueError('Audit or repeat invalid')
 rows=audit['sequences']['seq23']['per_frame'];frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23']
 oldrows=read(a.baseline/'frames.json');newrows=read(a.candidate/'frames.json');windows={'earlier_0_46':range(0,47),'late_47_55':range(47,56)};result={}
 for name,ids in windows.items():
  group=[rows[i] for i in ids];present=[x for x in group if x['present']];empty=[x for x in group if not x['present']]
  entry={'selection':'post-hoc; chosen after observing where gains occurred','present_frames':len(present),'empty_frames':len(empty)}
  for arm,folder,metadata in [('baseline',a.baseline,oldrows),('window',a.candidate,newrows)]:
   tp=fp=fn=0
   for r in present:
    i=r['frame'];mask=folder/f'{i:06d}.png'
    if digest(mask)!=metadata[i]['mask_sha256'] or digest(frames[i].image_path)!=metadata[i]['image_sha256'] or digest(frames[i].mask_path)!=metadata[i]['label_sha256']:raise ValueError('Input or mask changed')
    truth=load_label_mask(frames[i].mask_path,[1])==1;prediction=np.asarray(Image.open(mask))>0
    tp+=int((truth&prediction).sum());fp+=int((~truth&prediction).sum());fn+=int((truth&~prediction).sum())
   entry[arm]={'mean_dice':float(np.mean([r['scores']['candidate' if arm=='window' else 'baseline']['dice'] for r in present])) if present else None,
               'tp':tp,'fp':fp,'fn':fn,'pooled_precision':tp/(tp+fp) if tp+fp else None,'pooled_recall':tp/(tp+fn) if tp+fn else None,
               'false_alarms':sum(r['scores']['candidate' if arm=='window' else 'baseline']['false_alarm'] for r in empty)}
  result[name]=entry
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 payload={'scope':'Retrospective diagnostic on one video, windows selected after seeing results.','windows':result,'audit_sha256':d['results_sha256'],
          'repeat_verification_sha256':digest(a.repeat_verification/'verification.json'),'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# Sequence 23: where the token-window difference occurs','',payload['scope'],'', '| Frame interval | Present / empty | Baseline Dice | Window Dice | Baseline pooled precision / recall | Window pooled precision / recall | Empty false alarms baseline / window |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 fmt=lambda x:'—' if x is None else f'{x:.3f}'
 for name,x in result.items():
  b=x['baseline'];c=x['window'];lines.append(f"| {name} | {x['present_frames']} / {x['empty_frames']} | {fmt(b['mean_dice'])} | {fmt(c['mean_dice'])} | {fmt(b['pooled_precision'])} / {fmt(b['pooled_recall'])} | {fmt(c['pooled_precision'])} / {fmt(c['pooled_recall'])} | {b['false_alarms']} / {c['false_alarms']} |")
 lines+=['','These intervals were selected after seeing the masks and are not independent evidence. Frames are correlated, and the candidate changes old-tail fusion as well as the window; this analysis cannot establish which change caused the difference. The initial detector prompt on frame 0 is on a labeled-empty frame in both arms.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

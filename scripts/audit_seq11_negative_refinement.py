#!/usr/bin/env python3
"""Independently audit negative-only masks against fixed previous controls."""
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
 for key in ('pilot','prior-audit','reference','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();candidate=read(a.pilot/'predictions.json');old=read(a.prior_audit/'results.json');ref=read(a.reference/'records.json');done=read(a.reference/'completion.json')
 if not done['passed'] or digest(a.reference/'records.json')!=done['records_sha256'] or candidate['identity']['reference_sha256']!=done['records_sha256'] or old['reference_sha256']!=done['records_sha256']:
  raise ValueError('Reference changed')
 if candidate['identity']['runner_sha256']!=digest(ROOT/'scripts/run_seq11_negative_refinement.py') or candidate['identity']['protocol_sha256']!=digest(ROOT/'protocols/seq11_temporal_negative_v1.json') or not read(a.pilot/'parameter_integrity_after.json')['unchanged']:
  raise ValueError('Candidate provenance invalid')
 if old['audit_sha256']!=digest(ROOT/'scripts/audit_seq11_temporal_refinement.py'):raise ValueError('Prior audit source differs')
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11'];prior={r['target']:r for r in old['rows']};rows=[]
 for item in candidate['records']:
  i=item['target'];orig=prior[i];frame=frames[i]
  if digest(frame.image_path)!=item['image_sha256'] or digest(frame.mask_path)!=orig['label_sha256'] or item['source']!=orig['source']:raise ValueError('Frame or label differs')
  gt=load_label_mask(frame.mask_path,[1])==1;metrics={k:orig['metrics'][k] for k in ('lean_reference','box_only','random_points','contrast_points')}
  refpath=a.reference/f'{i:06d}.png'
  if digest(refpath)!=ref['mask_sha256'][i] or digest(refpath)!=item['input_mask_sha256']:raise ValueError('Original mask changed')
  for arm,h in item['masks'].items():
   path=a.pilot/f'{i:06d}_{arm}.png'
   if digest(path)!=h:raise ValueError('Candidate mask changed')
   pred=np.asarray(Image.open(path))>0
   if pred.shape!=gt.shape:raise ValueError('Mask shape differs')
   den=int(pred.sum()+gt.sum());score=2*int((pred&gt).sum())/den if den else 1.
   m={'dice':score,'predicted_pixels':int(pred.sum()),'false_alarm':bool(pred.any()) if not gt.any() else None}
   if arm=='box_only':
    if m!=metrics['box_only']:raise ValueError('Box-only scoring changed')
   else:metrics[arm]=m
  rows.append({'source':item['source'],'target':i,'present':bool(gt.any()),'label_sha256':digest(frame.mask_path),'metrics':metrics})
 if [r['target'] for r in rows]!=[r['target'] for r in old['rows']]:raise ValueError('Frame coverage differs')
 summary={}
 for arm in rows[0]['metrics']:
  vals=[r['metrics'][arm]['dice'] for r in rows if r['present']];empty=[r['metrics'][arm]['false_alarm'] for r in rows if not r['present']]
  summary[arm]={'present_mean_dice':float(np.mean(vals)),'present_count':len(vals),'false_alarms':sum(empty),'empty_count':len(empty)}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'summary':summary,'rows':rows,'pilot_sha256':digest(a.pilot/'predictions.json'),'prior_audit_sha256':digest(a.prior_audit/'results.json'),'reference_sha256':done['records_sha256'],'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

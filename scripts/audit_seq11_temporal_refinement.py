#!/usr/bin/env python3
"""Independently rescore the saved temporal-point prompt pilot."""
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
 for k in ('pilot','reference','background-audit','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();run=read(a.pilot/'predictions.json');old=read(a.reference/'records.json');done=read(a.reference/'completion.json')
 if not done['passed'] or digest(a.reference/'records.json')!=done['records_sha256'] or run['identity']['reference_sha256']!=done['records_sha256']:raise ValueError('Reference changed')
 if run['identity']['runner_sha256']!=digest(ROOT/'scripts/run_seq11_temporal_refinement.py') or run['identity']['protocol_sha256']!=digest(ROOT/'protocols/seq11_temporal_refinement_v1.json'):raise ValueError('Implementation changed')
 if not read(a.pilot/'parameter_integrity_after.json')['unchanged']:raise ValueError('Model changed')
 audit=read(a.background_audit/'results.json');targets={r['target']:r for r in audit['rows']}
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11'];rows=[]
 for item in run['records']:
  i=item['target'];frame=frames[i];prior=targets[i]
  if item['source']!=prior['source'] or digest(frame.image_path)!=item['image_sha256'] or digest(frame.mask_path)!=prior['target_label_sha256']:raise ValueError('Input changed')
  truth=load_label_mask(frame.mask_path,[1])==1;predictions={'lean_reference':a.reference/f'{i:06d}.png'}
  predictions.update({k:a.pilot/f'{i:06d}_{k}.png' for k in item['masks']});metrics={}
  for arm,path in predictions.items():
   expected=old['mask_sha256'][i] if arm=='lean_reference' else item['masks'][arm]
   if digest(path)!=expected:raise ValueError('Mask changed')
   pred=np.asarray(Image.open(path))>0
   if pred.shape!=truth.shape:raise ValueError('Mask shape changed')
   den=int(pred.sum()+truth.sum());metrics[arm]={'dice':2*int((pred&truth).sum())/den if den else 1.,'predicted_pixels':int(pred.sum()),'false_alarm':bool(pred.any()) if not truth.any() else None}
  rows.append({'source':item['source'],'target':i,'present':bool(truth.any()),'label_sha256':digest(frame.mask_path),'metrics':metrics})
 arms=list(rows[0]['metrics']);summary={}
 for arm in arms:
  present=[r['metrics'][arm]['dice'] for r in rows if r['present']];empty=[r['metrics'][arm]['false_alarm'] for r in rows if not r['present']]
  summary[arm]={'present_mean_dice':float(np.mean(present)) if present else None,'present_count':len(present),'false_alarms':sum(empty),'empty_count':len(empty)}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'summary':summary,'rows':rows,'pilot_sha256':digest(a.pilot/'predictions.json'),'reference_sha256':done['records_sha256'],
         'background_audit_sha256':digest(a.background_audit/'results.json'),'audit_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

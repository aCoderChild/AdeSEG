#!/usr/bin/env python3
"""Score frozen factorial memory-encoder mixtures on selected sequence-23 reads."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('replay','verification','crop-reference','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');ref=read(a.crop_reference/'completion.json');refs=read(a.crop_reference/'frames.json')
if not all(x['passed'] for x in (r,v,ref)) or r['mode']!='candidate' or v['mode']!='candidate' or v['first_sha256']!=digest(a.replay/'completion.json') or v['runner_sha256']!=digest(ROOT/'scripts/replay_lean_empty_encoder_components.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_empty_component_gates.py') or ref['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen replay or reference changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];rows=[]
for e in r['events']:
 i=e['target_frame'];f=frames[i]
 if digest(f.image_path)!=refs[i]['image_sha256'] or digest(f.mask_path)!=refs[i]['label_sha256']:raise ValueError('Input changed')
 gt=load_label_mask(f.mask_path,[1])==1
 if gt[:,:VIEWPORT[0]].any():raise ValueError('Annotated foreground outside viewport')
 present=bool(gt.any());scores={};masks={}
 if present:raise ValueError('Preselected empty control is no longer labeled empty')
 for name,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=sha:raise ValueError('Prediction changed')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=gt.shape:raise ValueError('Output shape differs')
  n=int(pred.sum())+int(gt.sum());scores[name]={'dice':2*int((pred&gt).sum())/n if present else None,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())};masks[name]=pred
 changes={arm:int(np.count_nonzero(mask!=masks[arm.split('_')[0]+'_rrr'])) for arm,mask in masks.items()}
 rows.append({'source_frame':e['source_frame'],'target_frame':i,'present':present,'image_sha256':refs[i]['image_sha256'],'label_sha256':refs[i]['label_sha256'],'source_object_logits':e['source_object_logits'],'same_source_vision':e['same_source_vision'],'same_source_mask':e['same_source_mask'],'scores':scores,'object_logits':e['object_logits'],'changed_pixels':changes})
summary={name:sum(row['scores'][name]['false_alarm'] for row in rows) for name in rows[0]['scores']}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'summary':summary,'scope':'Post-hoc labeled-empty fixed reads; mixed encoder inputs can be off-manifold. No propagation, independent generalization or inference policy is evaluated.','rows':rows,'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Sequence 23: labeled-empty encoder input mixtures','',result['scope'],'','Each three-letter arm is ordered **image features / predicted mask logits / native object score**. `r` takes the recipient source frame and `d` the other trajectory. The actual pretrained encoder and bfloat16 storage conversion are reused; native target-read presence behavior is unchanged.','','| Target | Recipient | Arm | False alarm | Native target object logit | Pixels changed vs recipient |','| ---: | --- | :---: | ---: | ---: | ---: |']
for row in rows:
 for recipient in ('recent','spread'):
  for arm in ('rrr','drr','rdr','rrd','rdd','drd','ddr','ddd'):
   name=f'{recipient}_{arm}';z=row['scores'][name];score=f"{z['dice']:.4f}" if z['dice'] is not None else 'FA='+str(z['false_alarm']);lines.append(f"| {row['target_frame']} | {recipient} | {arm} | {score} | {row['object_logits'][name]:.4f} | {row['changed_pixels'][name]} |")
lines+=['','## Selected-frame totals','','| Recipient | Arm | Labeled-empty false alarms |','| --- | :---: | ---: |']
for recipient in ('recent','spread'):
 for arm in ('rrr','drr','rdr','rrd','rdd','drd','ddr','ddd'):
  lines.append(f"| {recipient} | {arm} | {summary[recipient+'_'+arm]} / {len(rows)} |")
lines+=['','`rrr` is the recipient no-op and `ddd` is the independently verified full newest-memory exchange. Mixed inputs may be internally inconsistent; effects are conditional and need not add. Source trajectories already differ because of past predictions. This does not establish a deployable policy, full-video improvement, or generalization.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))

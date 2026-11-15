#!/usr/bin/env python3
"""Score frozen recent/spread memory exchange masks after independent replay."""
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
if not all(x['passed'] for x in (r,v,ref)) or r['mode']!='candidate' or v['mode']!='candidate' or v['first_sha256']!=digest(a.replay/'completion.json') or v['runner_sha256']!=digest(ROOT/'scripts/replay_lean_recent_spread_exchange.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_recent_spread_exchange_gates.py') or ref['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen replay or reference changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];rows=[]
for e in r['events']:
 i=e['frame'];f=frames[i]
 if digest(f.image_path)!=refs[i]['image_sha256'] or digest(f.mask_path)!=refs[i]['label_sha256']:raise ValueError('Input changed')
 gt=load_label_mask(f.mask_path,[1])==1
 if gt[:,:VIEWPORT[0]].any():raise ValueError('Annotated foreground outside viewport')
 present=bool(gt.any());scores={};masks={}
 for arm,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{arm}.png'
  if digest(path)!=sha:raise ValueError('Prediction changed')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=gt.shape:raise ValueError('Output geometry differs')
  n=int(pred.sum())+int(gt.sum());scores[arm]={'dice':2*int((pred&gt).sum())/n if present else None,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())};masks[arm]=pred
 changes={arm:int(np.count_nonzero(masks[arm]!=masks['recent_noop' if arm.startswith('recent_') else 'spread_noop'])) for arm in masks if not arm.endswith('_noop')}
 rows.append({'frame':i,'present':present,'image_sha256':refs[i]['image_sha256'],'label_sha256':refs[i]['label_sha256'],'scores':scores,'object_logits':e['object_logits'],'changed_pixels':changes,'same_current_query':e['same_current_query'],'same_decoder_image_inputs':e['same_decoder_image_inputs'],'total_keys':e['total_keys'],'pointer_keys':e['pointer_keys'],'history':e['history']})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'scope':'Previously inspected selected sequence 23 same-read cross-trajectory exchanges; not full propagation or independent generalization.','frames':rows,'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Sequence 23: recent-versus-spread memory exchanges','',result['scope'],'','For each recipient, no-op and swapped inputs use that recipient’s current query, decoder image inputs, conditioning and remaining memory. Exchanged features and positional encodings move together; native presence behavior is unchanged.','','| Frame | Label | Recipient | No-op Dice / object logit | All spatial / object logit | Newest entry / object logit | Pointer tail / object logit |','| ---: | --- | --- | ---: | ---: | ---: | ---: |']
for x in rows:
 for recipient,donor in (('recent','spread'),('spread','recent')):
  arms=[recipient+'_noop']+[f'{recipient}_with_{donor}_{part}' for part in ('spatial','newest','pointers')]
  def fmt(name):
   z=x['scores'][name];value=f"{z['dice']:.4f}" if z['dice'] is not None else 'FA='+str(z['false_alarm'])
   return f"{value} / {x['object_logits'][name]:.4f}"
  lines.append(f"| {x['frame']} | {'present' if x['present'] else 'empty'} | {recipient} | "+' | '.join(fmt(arm) for arm in arms)+' |')
lines+=['',f"Exact current query equality across trajectories: {sum(x['same_current_query'] for x in rows)}/{len(rows)} reads. Exact equality of decoder inputs other than memory-conditioned backbone features: {sum(x['same_decoder_image_inputs'] for x in rows)}/{len(rows)} reads.",'','All results are local conditional interventions on previously inspected histories. Spatial exchanges change source content and time positions together; pointer exchanges change the full object-pointer suffix. Past prediction feedback, source ages and token utility are not isolated. Empty labels may be incomplete.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))

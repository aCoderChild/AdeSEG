#!/usr/bin/env python3
"""Score frozen temporal-slot diagnostic outputs on selected seq23 reads."""
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
a=p.parse_args();read=lambda p:json.loads(p.read_text());r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');ref=read(a.crop_reference/'completion.json');rowsref=read(a.crop_reference/'frames.json')
if not all(x['passed'] for x in (r,v,ref)) or r['mode']!='candidate' or v['runner_sha256']!=digest(ROOT/'scripts/replay_lean_temporal_slot_rank.py') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_slot_rank_repeats.py') or v['module_sha256']!=digest(ROOT/'modeling/lean_temporal_rank_positions.py') or ref['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen replay or reference changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];rows=[]
for e in r['events']:
 i=e['frame'];f=frames[i]
 if digest(f.image_path)!=rowsref[i]['image_sha256'] or digest(f.mask_path)!=rowsref[i]['label_sha256']:raise ValueError('Input changed')
 gt=load_label_mask(f.mask_path,[1])==1
 if gt[:,:VIEWPORT[0]].any():raise ValueError('Annotated target outside viewport')
 present=bool(gt.any());masks={};scores={}
 for name,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=sha:raise ValueError('Prediction changed')
  mask=np.asarray(Image.open(path))>0
  if mask.shape!=gt.shape:raise ValueError('Output geometry differs')
  n=int(mask.sum())+int(gt.sum());scores[name]={'dice':2*int((mask&gt).sum())/n if present else None,'false_alarm':bool(mask.any()) if not present else None,'predicted_pixels':int(mask.sum())};masks[name]=mask
 changes={name:int(np.count_nonzero(masks[name]!=masks['noop'])) for name in ('rank','permuted')}
 rows.append({'frame':i,'present':present,'image_sha256':rowsref[i]['image_sha256'],'label_sha256':rowsref[i]['label_sha256'],'scores':scores,'changed_pixels':changes,'total_keys':e['total_keys'],'pointer_keys':e['pointer_keys'],'segments':e['segments']})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'scope':'Selected, previously inspected fixed reads; slot substitution only, no propagation or independent generalization.','frames':rows,'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Sequence 23: same-read temporal-slot substitution','',result['scope'],'','| Frame | Label | Original slots | Ranked slots | Permuted slots | Changed pixels: rank / permuted |','| ---: | --- | ---: | ---: | ---: | ---: |']
for x in rows:
 fmt=lambda s:f"{s['dice']:.6f}" if s['dice'] is not None else 'FA='+str(s['false_alarm']);z=x['scores'];change=x['changed_pixels'];lines.append(f"| {x['frame']} | {'present' if x['present'] else 'empty'} | {fmt(z['noop'])} | {fmt(z['rank'])} | {fmt(z['permuted'])} | {change['rank']} / {change['permuted']} |")
lines+=['','All arms use the same saved memory values, prompt, image query, pointers and spatial rotary indices. Ranked and permuted controls reuse the same checkpoint slots and slot multiset; rank does not encode real elapsed age. This is not a full-video result, and selected reads cannot prove generalization. Labels may be incomplete.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))

#!/usr/bin/env python3
"""Independently reconstruct retained indices from each source ranking."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.cross_frame_token_pool import TokenGroup,select_global_tokens
from modeling.lean_region_token_pool import region_indices
from modeling.lean_rgb_contrast_control import rgb_reference,rgb_contrast
from modeling.seq23_viewport import crop_viewport,VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','replay','scores','merged','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');r=read(a.replay/'completion.json');scores=read(a.scores/'completion.json')
if not all(x['passed'] for x in (cap,r,scores)) or r['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or r['score_receipt_sha256']!=digest(a.scores/'completion.json') or r['runner_sha256']!=digest(ROOT/'scripts/replay_lean_channel_fixed_retention.py'):raise ValueError('Inputs changed')
seq=cap['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq]
if cap['viewport']:prompt_path=a.capture/'prompt.png';prompt=read(a.capture/'prompt.json')
else:
 folder=Path(read(a.merged/'locations.json')[seq])/seq;prompt_path=folder/'prompt_refined.png';prompt=read(folder/'prompt.json')
if digest(prompt_path)!=r['rgb_prompt_sha256']:raise ValueError('Prompt changed')
def image(i):
 path=frames[i].image_path
 if digest(path)!=r['rgb_source_image_sha256'][str(i)]:raise ValueError('Image changed')
 with Image.open(path) as im:x=np.asarray(im.convert('RGB'))
 return crop_viewport(x) if cap['viewport'] else x
initial=image(prompt['first_detection_index']);mask=np.asarray(Image.open(prompt_path))>0
if cap['viewport']:x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
rgb=rgb_reference(initial,mask);snapshots={x['frame']:x for x in cap['snapshots']};score_files={x['source_frame']:x for x in scores['events']};checks=0
for e in r['events']:
 f=e['frame'];path=a.capture/'source_snapshots'/f'{f:06d}.pt'
 if digest(path)!=snapshots[f]['sources_sha256']:raise ValueError('Source changed')
 source=torch.load(path,map_location='cpu',weights_only=True);groups={m:[] for m in ('usage','contrast','distributed','three_region','foreground','shuffled_contrast','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt')};bounds=[];starts=[];rng=np.random.default_rng(20261114)
 for item in source['sources']:
  i=item['frame'];ids=item['token_ids'].numpy();usage=item['usage'].numpy();base=item['contrast'].numpy();starts.append(item['start']);bounds.append(item['boundary'].numpy())
  sp=a.scores/f'{i:06d}.pt'
  if digest(sp)!=score_files[i]['file_sha256']:raise ValueError('Score changed')
  trained=torch.load(sp,map_location='cpu',weights_only=True);color=rgb_contrast(image(i),rgb)[item['token_ids']].numpy();shuffled=base[rng.permutation(len(ids))]
  for m in groups:
   v=trained[m][item['token_ids']].numpy() if m in trained else color if m=='rgb_contrast' else shuffled if m=='shuffled_contrast' else base
   groups[m].append(TokenGroup(i,ids,usage,v))
 for m,gs in groups.items():
  if m=='three_region':idx=region_indices(gs,bounds,4096,'three_region')
  elif m=='foreground':
   ranked=[]
   for j,g in enumerate(gs):ranked.extend((float(.5*u+.25*(c+1)),j,k) for k,(u,c) in enumerate(zip(g.usage_rank,g.contrast)))
   ranked.sort(key=lambda x:-x[0]);idx=[[] for _ in gs]
   for _,j,k in ranked[:4096]:idx[j].append(k)
  else:idx=select_global_tokens(gs,4096,'usage' if m=='usage' else 'distributed' if m=='distributed' else 'contrast')
  expected=sorted(start+int(k) for start,ids in zip(starts,idx) for k in ids)
  if expected!=e['kept'][m]:raise ValueError(f'Ranking differs: {seq} {f} {m}')
  checks+=1
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':seq,'rankings_checked':checks,'replay_sha256':digest(a.replay/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print(seq,checks,'retention rankings independently reproduced')

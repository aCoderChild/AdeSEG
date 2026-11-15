#!/usr/bin/env python3
"""Independently audit repeated score reconstruction and fixed-read replays."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output

def read(x):return json.loads(x.read_text())
p=argparse.ArgumentParser()
for k in ('capture','scores-a','scores-b','first','second','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');sa=read(a.scores_a/'completion.json');sb=read(a.scores_b/'completion.json');one=read(a.first/'completion.json');two=read(a.second/'completion.json')
if not all(x['passed'] for x in (cap,sa,sb,one,two)) or sa!=sb or one!=two or sa['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or one['capture_sha256']!=sa['capture_sha256'] or one['score_receipt_sha256']!=digest(a.scores_a/'completion.json') or one['score_repeat_sha256']!=digest(a.scores_b/'completion.json') or one['runner_sha256']!=digest(ROOT/'scripts/replay_lean_channel_remaining_intervention.py') or sa['runner_sha256']!=digest(ROOT/'scripts/reconstruct_lean_channel_remaining_scores.py'):raise ValueError('Receipts differ')
for e in sa['events']:
 for folder in (a.scores_a,a.scores_b):
  path=folder/f"{e['source_frame']:06d}.pt"
  if digest(path)!=e['file_sha256']:raise ValueError('Score artifact changed')
  scores=torch.load(path,map_location='cpu',weights_only=True)
  if {k:sha(v) for k,v in scores.items()}!=e['score_sha256']:raise ValueError('Score tensor changed')
methods={'usage','contrast','distributed','three_region','foreground','shuffled','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt'}
snapshots={v['frame']:v for v in cap['snapshots']}
if set(snapshots)!=set(cap['targets']) or {v['frame'] for v in one['events']}!=set(snapshots):raise ValueError('Missing frames')
for event in one['events']:
 f=event['frame'];src=a.capture/'source_snapshots'/f'{f:06d}.pt';snap=a.capture/'snapshots'/f'{f:06d}.pt'
 if digest(src)!=snapshots[f]['sources_sha256'] or digest(snap)!=snapshots[f]['read_sha256']:raise ValueError('Snapshot changed')
 source=torch.load(src,map_location='cpu',weights_only=True);eligible=set()
 for s in source['sources']:
  if s['frame']>=f-1:raise ValueError('Newest token selected')
  eligible.update(range(s['start'],s['start']+len(s['token_ids'])))
 if set(event['deleted'])!=methods or set(event['masks'])!=methods|{'noop'}:raise ValueError('Method coverage')
 for ids in event['deleted'].values():
  if len(ids)!=1024 or len(set(ids))!=1024 or not set(ids)<=eligible:raise ValueError('Deletion differs')
 for name,expected in event['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{f:06d}_{name}.png')!=expected:raise ValueError('Mask changed')
 if event['masks']['noop']!=digest(a.capture/f'{f:06d}.png'):raise ValueError('No-op changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':cap['sequence'],'frames':cap['targets'],'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'scores_sha256':digest(a.scores_a/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('Repeated score tensors, 1024-token deletions, predictions verified against the captured trajectory.')

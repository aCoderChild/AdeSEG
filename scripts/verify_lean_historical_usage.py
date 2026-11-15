#!/usr/bin/env python3
"""Audit stored author usage against frozen attention and source-token identities."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('capture','source-reference','source-verification','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());r=read(a.capture/'historical_usage_receipt.json');prior=read(a.source_reference/'utility_capture_receipt.json');src=read(a.source_reference/'source_capture_receipt.json');v=read(a.source_verification/'verification.json');seq=r['sequence']
if not r['passed'] or not prior['passed'] or not src['passed'] or not v['passed'] or r['reference_sha256']!=digest(a.source_reference/'utility_capture_receipt.json') or r['nested_receipt_sha256']!=digest(a.capture/'utility_capture_receipt.json') or read(a.capture/'utility_capture_receipt.json')!=prior or r['observer_sha256']!=digest(ROOT/'modeling/lean_historical_usage_capture.py') or r['runner_sha256']!=digest(ROOT/'scripts/capture_lean_historical_usage.py') or r['protocol_sha256']!=digest(ROOT/'protocols/lean_historical_usage_capture_v1.json'):raise ValueError('Capture provenance differs')
if seq=='seq3':
 if v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_source_gates.py') or digest(a.source_reference/'source_capture_receipt.json') not in (v['first_sha256'],v['second_sha256']):raise ValueError('Source gate changed')
else:
 if v['runner_sha256']!=digest(ROOT/'scripts/verify_lean_utility_source_candidate.py') or v['sequence']!=seq or v['source_receipt_sha256']!=digest(a.source_reference/'source_capture_receipt.json'):raise ValueError('Source audit changed')
for name in ('records.json','positions.json','selections.json','memory_events.json','token_window_events.json'):
 if digest(a.capture/name)!=digest(a.source_reference/name):raise ValueError('Trajectory trace differs')
masks=sorted(p.name for p in a.source_reference.glob('[0-9]*.png'))
if not masks or masks!=sorted(p.name for p in a.capture.glob('[0-9]*.png')) or any(digest(a.capture/n)!=digest(a.source_reference/n) for n in masks):raise ValueError('Mask trajectory differs')
events={e['source_frame']:e for e in r['events']}
if len(events)!=len(r['events']):raise ValueError('Duplicate source records')
for source,e in events.items():
 path=a.capture/'historical_usage'/e['file']
 if digest(path)!=e['sha256']:raise ValueError('Usage record changed')
 u=torch.load(path,map_location='cpu',weights_only=True)
 if u['source_frame']!=source or u['read_frame']!=e['read_frame'] or u['usage'].shape!=(4096,) or not torch.isfinite(u['usage']).all() or u['keep_idx'].shape!=(1024,) or len(torch.unique(u['keep_idx']))!=1024:raise ValueError('Invalid stored usage')
attention_checks=0;retained_checks=0;missing=[]
for z in src['observations']:
 frame=z['frame'];name=f'{frame:06d}.pt';sp=a.source_reference/'source_snapshots'/name;shot=a.source_reference/'snapshots'/name
 if digest(sp)!=src['source_snapshots'][name]:raise ValueError('Source snapshot differs')
 snapshot=torch.load(sp,map_location='cpu',weights_only=True);data=torch.load(shot,map_location='cpu',weights_only=True)
 for s in snapshot['sources']:
  source=s['frame']
  if source not in events:
   if source<frame-1:missing.append((frame,source))
   continue
  u=torch.load(a.capture/'historical_usage'/events[source]['file'],map_location='cpu',weights_only=True)
  if source==frame-1 and u['read_frame']==frame and len(s['token_ids'])==4096:
   actual=data['attention_outputs'][1][s['start']:s['start']+4096].float()
   if not torch.equal(u['usage'],actual):raise ValueError('Historical usage differs from frozen source-read attention')
   attention_checks+=1
  if source<frame-1:
   if u['read_frame']>=frame or not torch.equal(u['keep_idx'],s['token_ids']):raise ValueError('Historical usage unavailable or retained identities differ')
   retained_checks+=1
if missing or not attention_checks or not retained_checks:raise ValueError(f'Insufficient verified historical coverage: {missing}')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'sequence':seq,'capture_sha256':digest(a.capture/'historical_usage_receipt.json'),'source_reference_sha256':digest(a.source_reference/'source_capture_receipt.json'),'attention_checks':attention_checks,'retained_source_checks':retained_checks,'usage_records':len(events),'protocol_sha256':r['protocol_sha256'],'observer_sha256':r['observer_sha256'],'runner_sha256':r['runner_sha256'],'verifier_sha256':digest(Path(__file__)),'scope':'Pre-pruning usage captured causally; no utility or efficacy claim.'}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

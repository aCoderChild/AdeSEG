#!/usr/bin/env python3
"""Verify repeated cropped prompt and first-frame memory engineering runs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('first','second','baseline','window-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text())
x=read(a.first/'token_window_observation.json');y=read(a.second/'token_window_observation.json');base=read(a.baseline/'completion.json');verify=read(a.window_verification/'verification.json')
if x!=y or not all(v['passed'] for v in (x,y,base,verify)) or x['mode']!='gate' or x['sequence']!='seq23' or x['source_sha256']!=digest(ROOT/'scripts/run_seq23_temporal_spread_window.py') or x['verification_sha256']!=digest(a.window_verification/'verification.json') or x['baseline_completion_sha256']!=digest(a.baseline/'completion.json'):raise ValueError('Crop-gate provenance differs')
gates=[]
for folder in (a.first,a.second):
 gate=read(folder/'gate.json');events=read(folder/'token_window_events.json')
 if digest(folder/'gate.json')!=x['records_sha256'] or digest(folder/'token_window_events.json')!=x['events_sha256'] or any(digest(folder/k)!=v for k,v in gate['files'].items()) or not read(folder/'parameter_integrity_after.json')['unchanged']:raise ValueError('Cropped artifacts differ')
 if digest(folder/'prompt.json')!=base['prompt_sha256'] or digest(folder/'prompt.png')!=digest(a.baseline/'prompt.png') or any(digest(folder/f'{i:06d}.png')!=digest(a.baseline/f'{i:06d}.png') for i in range(3)):raise ValueError('Cropped prompt or prefix differs')
 plans=[e for e in events if e['stage']=='admission_plan']
 if not plans or any(e['planned_reserved_tokens']>16384 or e['condition_tokens']!=4096 or e['new_tokens']!=4096 for e in plans):raise ValueError('Cropped budget differs')
 gates.append((gate,events))
if gates[0]!=gates[1]:raise ValueError('Cropped gate replays differ')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'source_sha256':x['source_sha256'],'window_verification_sha256':x['verification_sha256'],'baseline_completion_sha256':x['baseline_completion_sha256'],'first_observation_sha256':digest(a.first/'token_window_observation.json'),'second_observation_sha256':digest(a.second/'token_window_observation.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('Cropped prompt, first masks, budget and repeated gates verified.')

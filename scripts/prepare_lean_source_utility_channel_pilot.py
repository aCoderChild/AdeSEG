#!/usr/bin/env python3
"""Create label-free, channel-resolved descriptors for frozen deletion groups."""
import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('captures','deletions','scores','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);samples=[];max_error=0.
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 cap=a.captures/('source_gate_attempt02' if seq=='seq3' else f'{seq}_source_capture_attempt01');rep=a.deletions/('gate_attempt01' if seq=='seq3' else f'{seq}_attempt01');sc=a.scores/f'{seq}_scores_attempt02/scores.json';repeat=a.scores/f'{seq}_scores_attempt03/scores.json';x=read(sc);r=read(rep/'completion.json');receipt=read(cap/'source_capture_receipt.json')
 if x!=read(repeat) or not x['passed'] or not r['passed'] or not receipt['passed'] or x['sequence']!=seq or x['replay_sha256']!=digest(rep/'completion.json') or x['capture_sha256']!=digest(cap/'source_capture_receipt.json') or x['runner_sha256']!=digest(ROOT/'scripts/score_lean_source_utility_features.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_source_utility_channel_pilot_v1.json'):raise ValueError('Frozen source or score provenance changed')
 events={e['frame']:e for e in r['events'] if 'groups' in e}
 for event in x['events']:
  frame=event['frame'];item=events[frame];snapshot=cap/'source_snapshots'/f'{frame:06d}.pt'
  if digest(snapshot)!=receipt['source_snapshots'][snapshot.name]:raise ValueError('Source snapshot changed')
  sources=torch.load(snapshot,map_location='cpu',weights_only=True)['sources'];f=torch.load(cap/'features'/f'{frame:06d}.pt',map_location='cpu',weights_only=True);fg=f['foreground'].float();bg=f['background'].float();channels={}
  for s in sources:
   i=s['frame']
   if i>=frame-1:continue
   path=cap/'source_features'/f'{i:06d}.pt'
   if digest(path)!=receipt['source_features'][path.name]:raise ValueError('Source feature changed')
   feature=torch.load(path,map_location='cpu',weights_only=True)['feature'].float()
   # Linear channel contributions sum exactly to the pre-existing unweighted contrast.
   per_channel=F.normalize(feature,dim=0)*(fg-bg)[:,None,None]*.5
   grid=F.interpolate(per_channel[None],size=(64,64),mode='area')[0].reshape(32,4096).T
   for j,token in enumerate(s['token_ids'].tolist()):channels[s['start']+j]=grid[token]
  vectors=torch.stack([torch.stack([channels[j] for j in group]).mean(0) for group in item['groups']]);baseline=vectors.sum(1);prior=torch.tensor(event['scores']['unweighted']);error=float((baseline-prior).abs().max());max_error=max(max_error,error)
  if error>1e-6:raise ValueError('Channel decomposition does not reproduce original score')
  name=f'{seq}_{frame:06d}.pt';torch.save({'sequence':seq,'frame':frame,'foreground':fg,'background':bg,'contributions':vectors,'controls':event['scores']},out/name)
  samples.append({'sequence':seq,'frame':frame,'group':'development' if seq in ('seq2','seq3','seq4') else 'validation','file':name,'file_sha256':digest(out/name),'scores_sha256':digest(sc),'replay_sha256':digest(rep/'completion.json'),'capture_sha256':digest(cap/'source_capture_receipt.json')})
manifest={'passed':True,'samples':samples,'max_unweighted_error':max_error,'protocol_sha256':digest(ROOT/'protocols/lean_source_utility_channel_pilot_v1.json'),'runner_sha256':digest(Path(__file__)),'meaning':'Label-free feature descriptors only; no utility outcomes were opened.'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'passed':True,'samples':len(samples),'max_error':max_error}))

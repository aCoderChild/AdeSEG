#!/usr/bin/env python3
"""Audit frozen LOSO gates then score against independently verified outcomes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('runs','features','deletions','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());manifest=read(a.features/'manifest.json');samples={(r['sequence'],r['frame']):r for r in manifest['samples']};seqs=('seq2','seq3','seq4','seq9','seq10');frozen={};fold_hashes={}
# Verify every fold and checkpoint before opening any evaluation outcome in this script.
for seq in seqs:
 first=a.runs/f'{seq}_attempt01';second=a.runs/f'{seq}_attempt02';x=read(first/'predictions.json');y=read(second/'predictions.json')
 if x!=y or not x['passed'] or x['holdout']!=seq or x['training_sequences']!=sorted(set(seqs)-{seq}) or seq in x['training_outcome_sha256'] or x['feature_manifest_sha256']!=digest(a.features/'manifest.json') or x['author_sha256']!=author_sha256() or x['runner_sha256']!=digest(ROOT/'scripts/train_lean_source_utility_loso.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_source_utility_channel_loso_v1.json') or any(digest(f/'research_gate.pt')!=x['checkpoint_sha256'] for f in (first,second)):raise ValueError('Fold or repeat invalid')
 state=torch.load(first/'research_gate.pt',map_location='cpu',weights_only=True)
 if state['format']!='lean_source_utility_loso_research_only_v1' or state['engineering_only'] is not True or state['holdout']!=seq:raise ValueError('Checkpoint invalid')
 model=PromptContrastChannelAttention(initialize_for_training=True);model.gate.load_state_dict(state['state_dict'],strict=True);model.train()
 if parameter_hash(model)!=x['final_parameter_sha256']:raise ValueError('Gate parameters changed')
 for item in x['predictions']:
  feature=samples[(seq,item['frame'])];path=a.features/feature['file']
  if digest(path)!=feature['file_sha256']:raise ValueError('Features changed')
  f=torch.load(path,map_location='cpu',weights_only=True)
  with torch.enable_grad():scores=(f['contributions']@model.channel_weights(f['foreground'],f['background'])).detach().numpy()
  if not np.array_equal(scores,np.asarray(item['scores'],dtype=np.float32)) or f['controls']!=item['controls']:raise ValueError('Prediction replay differs')
  frozen[(seq,item['frame'])]=item
 fold_hashes[seq]=digest(first/'predictions.json')
rows=[];details=[];arms=('utility_trained','unweighted','cbam','rgb','shuffled','age')
for seq in seqs:
 folder=a.deletions/f'{seq}_audit_attempt01';v=read(a.deletions/f'{seq}_score_verification_attempt01/verification.json');data=read(folder/'results.json')
 if not v['passed'] or v['results_sha256']!=digest(folder/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_scores.py'):raise ValueError('Held-out outcomes changed')
 per={k:[] for k in arms};delta={k:[] for k in arms};empty={k:0 for k in arms};nempty=0
 for f in data['frames']:
  x=frozen[(seq,f['frame'])];scores={'utility_trained':x['scores'],**x['controls']};chosen={k:min(range(5),key=lambda i:(scores[k][i],i)) for k in arms};m=[f['scores'][f'group{i}'] for i in range(5)]
  if f['present']:
   u=[f['scores']['noop']['dice']-z['dice'] for z in m];mean=sum(z['dice'] for z in m)/5
   for k in arms:
    delta[k].append(m[chosen[k]]['dice']-mean);v0=[]
    for i in range(5):
     for j in range(i+1,5):
      if u[i]==u[j]:continue
      z=(scores[k][i]-scores[k][j])*(u[i]-u[j]);v0.append(1. if z>0 else 0. if z<0 else .5)
    if v0:per[k].append(float(np.mean(v0)))
  else:
   nempty+=1
   for k in arms:empty[k]+=int(m[chosen[k]]['false_alarm'])
  details.append({'sequence':seq,'frame':f['frame'],'present':f['present'],'chosen':chosen})
 rows.append({'sequence':seq,'present_reads':len(delta['utility_trained']),'empty_reads':nempty,'concordance':{k:float(np.mean(per[k])) if per[k] else None for k in arms},'paired_dice_minus_random':{k:float(np.mean(delta[k])) if delta[k] else None for k in arms},'empty_false_alarms':empty,'outcome_sha256':digest(folder/'results.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'rows':rows,'details':details,'fold_sha256':fold_hashes,'protocol_sha256':digest(ROOT/'protocols/lean_source_utility_channel_loso_v1.json'),'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Source-utility channel gate: leave-one-sequence-out check','','Each fold trained the fixed channel gate on the other four sequences and froze its predictions before this audit joined outcomes. All sequences and outcomes had been inspected previously. Pairwise concordance is averaged within reads; 0.5 is chance for non-tied pairs. Δ Dice is the selected equal-key mask minus the mean of the five source-deletion masks.','','| Held out | Present / empty | Gate rank | Uniform rank | RGB rank | Shuffled rank | Age rank | Gate Δ Dice |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 c=r['concordance'];d=r['paired_dice_minus_random']['utility_trained'];f=lambda x:'—' if x is None else f'{x:.3f}';lines.append(f"| {r['sequence']} | {r['present_reads']} / {r['empty_reads']} | {f(c['utility_trained'])} | {f(c['unweighted'])} | {f(c['rgb'])} | {f(c['shuffled'])} | {f(c['age'])} | {'—' if d is None else f'{d:+.7f}'} |")
lines+=['','Only 16 present reads and one empty read across five sequences are available; patient independence is unknown. Source groups and paired outcomes are dependent, effects are very small, and this follow-up was chosen after prior inspection. No propagation result or inference-ready checkpoint follows. Pre-read Lean usage and distributed-policy controls were unavailable for these cached descriptors.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))

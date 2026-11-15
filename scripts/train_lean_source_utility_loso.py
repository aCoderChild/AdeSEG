#!/usr/bin/env python3
"""Fit a fixed utility-gate fold without reading held-out outcome files."""
import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('features','deletions','output'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--holdout',required=True,choices=('seq2','seq3','seq4','seq9','seq10'));a=p.parse_args();read=lambda f:json.loads(f.read_text());manifest=read(a.features/'manifest.json')
if not manifest['passed'] or manifest['runner_sha256']!=digest(ROOT/'scripts/prepare_lean_source_utility_channel_pilot.py') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_source_utility_channel_pilot_v1.json'):raise ValueError('Feature provenance changed')
features=[]
for item in manifest['samples']:
 path=a.features/item['file']
 if digest(path)!=item['file_sha256']:raise ValueError('Feature changed')
 x=torch.load(path,map_location='cpu',weights_only=True)
 if x['sequence']!=item['sequence'] or x['frame']!=item['frame'] or x['contributions'].shape!=(5,32):raise ValueError('Feature identity differs')
 features.append((item,x))
seqs=('seq2','seq3','seq4','seq9','seq10');targets={};outcome_hashes={}
for seq in seqs:
 if seq==a.holdout:continue
 folder=a.deletions/f'{seq}_audit_attempt01';vf=a.deletions/f'{seq}_score_verification_attempt01';data=read(folder/'results.json');v=read(vf/'verification.json')
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_scores.py') or v['results_sha256']!=digest(folder/'results.json'):raise ValueError('Training outcomes invalid')
 outcome_hashes[seq]=digest(folder/'results.json')
 for row in data['frames']:
  if row['present']:
   scores=row['scores'];targets[(seq,row['frame'])]=torch.tensor([scores['noop']['dice']-scores[f'group{i}']['dice'] for i in range(5)],dtype=torch.float32)
train=[(x,targets[(item['sequence'],item['frame'])]) for item,x in features if item['sequence']!=a.holdout and (item['sequence'],item['frame']) in targets]
held=[(item,x) for item,x in features if item['sequence']==a.holdout]
if len(train)!=len(targets) or not train or not held:raise ValueError('Training coverage differs')
torch.set_num_threads(2);torch.manual_seed(20261114);torch.use_deterministic_algorithms(True);model=PromptContrastChannelAttention(initialize_for_training=True);model.train();initial=parameter_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001);losses=[]
for step in range(100):
 opt.zero_grad();terms=[]
 for x,target in train:
  weights=model.channel_weights(x['foreground'],x['background']);scores=x['contributions']@weights;scores=(scores-scores.mean())/(scores.std(unbiased=False).detach()+1e-6);pairs=[]
  for i in range(5):
   for j in range(i+1,5):
    d=target[i]-target[j]
    if d!=0:pairs.append(F.softplus(-torch.sign(d)*(scores[i]-scores[j])))
  if pairs:terms.append(torch.stack(pairs).mean())
 if not terms:raise ValueError('No informative pairs')
 loss=torch.stack(terms).mean()
 if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
 loss.backward();opt.step();losses.append(float(loss.detach()))
final=parameter_hash(model)
if initial==final:raise ValueError('No parameter update')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.save({'format':'lean_source_utility_loso_research_only_v1','state_dict':model.gate.state_dict(),'holdout':a.holdout,'steps':100,'engineering_only':True,'feature_manifest_sha256':digest(a.features/'manifest.json')},out/'research_gate.pt');pred=[]
for item,x in held:
 with torch.enable_grad():score=(x['contributions']@model.channel_weights(x['foreground'],x['background'])).detach().tolist()
 pred.append({'sequence':a.holdout,'frame':item['frame'],'scores':score,'controls':x['controls']})
result={'passed':True,'holdout':a.holdout,'predictions':pred,'training_sequences':sorted(outcome_hashes),'training_present_reads':len(train),'training_outcome_sha256':outcome_hashes,'feature_manifest_sha256':digest(a.features/'manifest.json'),'initial_parameter_sha256':initial,'final_parameter_sha256':final,'initial_loss':losses[0],'final_loss':losses[-1],'checkpoint_sha256':digest(out/'research_gate.pt'),'author_sha256':author_sha256(),'protocol_sha256':digest(ROOT/'protocols/lean_source_utility_channel_loso_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Research-only fold; held-out outcome files not opened.'}
(out/'predictions.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'holdout':a.holdout,'training_reads':len(train),'initial_loss':losses[0],'final_loss':losses[-1]}))

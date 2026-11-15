#!/usr/bin/env python3
"""Train TCMN channel gate on development-only fixed-read utility ordering."""
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
a=p.parse_args();read=lambda f:json.loads(f.read_text());manifest=read(a.features/'manifest.json')
if not manifest['passed'] or manifest['runner_sha256']!=digest(ROOT/'scripts/prepare_lean_utility_channel_pilot.py') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_utility_trained_channel_pilot_v1.json'):raise ValueError('Feature provenance changed')
features=[]
for item in manifest['samples']:
 path=a.features/item['file']
 if digest(path)!=item['file_sha256']:raise ValueError('Feature changed')
 x=torch.load(path,map_location='cpu',weights_only=True)
 if x['sequence']!=item['sequence'] or x['frame']!=item['frame'] or x['contributions'].shape!=(5,32):raise ValueError('Feature identity differs')
 features.append((item,x))
if {i['sequence'] for i,_ in features if i['group']=='development'}!={'seq2','seq3','seq4'} or {i['sequence'] for i,_ in features if i['group']=='validation'}!={'seq9','seq10'}:raise ValueError('Split differs')
# Open development outcomes only; validation outcomes are deliberately not read.
targets={};outcome_hashes={}
for seq in ('seq2','seq3','seq4'):
 folder=a.deletions/f'{seq}_audit_attempt01';vf=a.deletions/f'{seq}_score_verification_attempt01';audit=read(folder/'results.json');v=read(vf/'verification.json')
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_scores.py') or v['results_sha256']!=digest(folder/'results.json'):raise ValueError('Development outcomes invalid')
 outcome_hashes[seq]=digest(folder/'results.json')
 for row in audit['frames']:
  if row['present']:
   scores=row['scores'];targets[(seq,row['frame'])]=torch.tensor([scores['noop']['dice']-scores[f'group{i}']['dice'] for i in range(5)],dtype=torch.float32)
train=[(x,targets[(item['sequence'],item['frame'])]) for item,x in features if item['group']=='development' and (item['sequence'],item['frame']) in targets]
if len(train)!=11:raise ValueError('Development present-read count changed')
torch.set_num_threads(2);torch.manual_seed(20261114);torch.use_deterministic_algorithms(True);model=PromptContrastChannelAttention(initialize_for_training=True);model.train();initial=parameter_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001);losses=[]
for step in range(100):
 opt.zero_grad();terms=[]
 for x,target in train:
  weights=model.channel_weights(x['foreground'],x['background']);scores=x['contributions']@weights;scores=(scores-scores.mean())/(scores.std(unbiased=False).detach()+1e-6);pairs=[]
  for i in range(5):
   for j in range(i+1,5):
    diff=target[i]-target[j]
    if diff!=0:pairs.append(F.softplus(-torch.sign(diff)*(scores[i]-scores[j])))
  if pairs:terms.append(torch.stack(pairs).mean())
 if not terms:raise ValueError('No non-tied development pairs')
 loss=torch.stack(terms).mean()
 if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
 loss.backward();opt.step();losses.append(float(loss.detach()))
final=parameter_hash(model)
if initial==final:raise ValueError('Training did not update weights')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
# Research-only artifact: cannot be loaded by the production inference adapter.
torch.save({'format':'lean_group_utility_research_only_v1','state_dict':model.gate.state_dict(),'steps':100,'engineering_only':True,'feature_manifest_sha256':digest(a.features/'manifest.json')},out/'research_gate.pt')
pred=[]
for item,x in features:
 with torch.enable_grad():scores=(x['contributions']@model.channel_weights(x['foreground'],x['background'])).detach().tolist()
 pred.append({'sequence':item['sequence'],'frame':item['frame'],'group':item['group'],'scores':scores,'controls':x['controls']})
result={'passed':True,'predictions':pred,'training_steps':100,'development_present_reads':len(train),'initial_parameter_sha256':initial,'final_parameter_sha256':final,'initial_loss':losses[0],'final_loss':losses[-1],'development_outcome_sha256':outcome_hashes,'feature_manifest_sha256':digest(a.features/'manifest.json'),'checkpoint_sha256':digest(out/'research_gate.pt'),'author_sha256':author_sha256(),'protocol_sha256':digest(ROOT/'protocols/lean_utility_trained_channel_pilot_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Development-trained exploratory group scorer; no validation outcomes opened and no online inference.'}
(out/'predictions.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'initial_loss':losses[0],'final_loss':losses[-1]}))

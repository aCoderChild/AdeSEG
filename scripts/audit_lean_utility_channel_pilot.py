#!/usr/bin/env python3
"""Verify trained group scorer, then evaluate frozen scores against held-out effects."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('first','second','features','deletions','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());x=read(a.first/'predictions.json');y=read(a.second/'predictions.json');manifest=read(a.features/'manifest.json')
if x!=y or not x['passed'] or x['feature_manifest_sha256']!=digest(a.features/'manifest.json') or x['author_sha256']!=author_sha256() or x['runner_sha256']!=digest(ROOT/'scripts/train_lean_utility_channel_pilot.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_utility_trained_channel_pilot_v1.json') or any(digest(folder/'research_gate.pt')!=x['checkpoint_sha256'] for folder in (a.first,a.second)):raise ValueError('Training repeat or provenance changed')
state=torch.load(a.first/'research_gate.pt',map_location='cpu',weights_only=True)
if state['format']!='lean_group_utility_research_only_v1' or state['engineering_only'] is not True or state['steps']!=100:raise ValueError('Incorrect research checkpoint')
model=PromptContrastChannelAttention(initialize_for_training=True);model.gate.load_state_dict(state['state_dict'],strict=True);model.train()
if parameter_hash(model)!=x['final_parameter_sha256']:raise ValueError('Parameter checksum differs')
entries={(i['sequence'],i['frame']):i for i in manifest['samples']};pred={(i['sequence'],i['frame']):i for i in x['predictions']}
if entries.keys()!=pred.keys():raise ValueError('Prediction coverage differs')
for key,item in entries.items():
 path=a.features/item['file']
 if digest(path)!=item['file_sha256']:raise ValueError('Features changed')
 f=torch.load(path,map_location='cpu',weights_only=True)
 with torch.enable_grad():score=(f['contributions']@model.channel_weights(f['foreground'],f['background'])).detach().numpy()
 if not np.array_equal(score,np.asarray(pred[key]['scores'],dtype=np.float32)) or f['controls']!=pred[key]['controls']:raise ValueError('Prediction does not reproduce checkpoint')
# Only now open validation outcomes, after checking both frozen predictions and trained weights.
rows=[];details=[];arms=('utility_trained','unweighted','cbam','rgb','shuffled','age')
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 audit=a.deletions/f'{seq}_audit_attempt01';ver=a.deletions/f'{seq}_score_verification_attempt01';data=read(audit/'results.json');v=read(ver/'verification.json')
 if not v['passed'] or v['results_sha256']!=digest(audit/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_scores.py'):raise ValueError('Outcome audit changed')
 if seq in x['development_outcome_sha256'] and x['development_outcome_sha256'][seq]!=digest(audit/'results.json'):raise ValueError('Training target changed')
 per={k:[] for k in arms};empty=0;ties=0
 for row in data['frames']:
  entry=pred[(seq,row['frame'])]
  if not row['present']:empty+=1;continue
  util=[row['scores']['noop']['dice']-row['scores'][f'group{i}']['dice'] for i in range(5)];pvals={'utility_trained':entry['scores'],**entry['controls']};vals={}
  for arm in arms:
   scores=pvals[arm];pair=[]
   for i in range(5):
    for j in range(i+1,5):
     du=util[i]-util[j]
     if du==0:
      if arm=='utility_trained':ties+=1
      continue
     dp=(scores[i]-scores[j])*du;pair.append(1. if dp>0 else 0. if dp<0 else .5)
   if pair:per[arm].append(float(np.mean(pair)));vals[arm]=float(np.mean(pair))
  details.append({'sequence':seq,'frame':row['frame'],'concordance':vals})
 rows.append({'sequence':seq,'split':'development' if seq in ('seq2','seq3','seq4') else 'validation','present_reads':len(per['utility_trained']),'empty_reads':empty,'tied_utility_pairs':ties,'concordance':{k:float(np.mean(v)) if v else None for k,v in per.items()},'outcome_sha256':digest(audit/'results.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'rows':rows,'read_details':details,'first_predictions_sha256':digest(a.first/'predictions.json'),'second_predictions_sha256':digest(a.second/'predictions.json'),'checkpoint_sha256':x['checkpoint_sha256'],'feature_manifest_sha256':x['feature_manifest_sha256'],'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Utility-trained channel gate: fixed-read feasibility','','The TCMN-author channel gate was trained on 11 development reads to rank the five frozen group-deletion effects. Predictions were generated twice and independently reproduced from the trained research-only checkpoint before this audit joined them to validation outcomes. Those outcomes had been inspected in earlier work. Values are mean within-read pairwise concordance; 0.5 is chance for non-tied pairs.','','| Sequence | Split | Present reads | Utility-trained | Uniform | Occupancy-trained | RGB | Shuffled | Age |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 c=r['concordance'];fmt=lambda q:'—' if q is None else f'{q:.3f}';lines.append(f"| {r['sequence']} | {r['split']} | {r['present_reads']} | {fmt(c['utility_trained'])} | {fmt(c['unweighted'])} | {fmt(c['cbam'])} | {fmt(c['rgb'])} | {fmt(c['shuffled'])} | {fmt(c['age'])} |")
lines+=['','The training targets are conditional deletion effects whose absolute Dice magnitude was very small. Groups share each read, and only five validation present reads from two previously inspected sequences are available. Patient independence is unknown, and one eligible empty read is not a useful absence test. The fitted artifact is explicitly research-only, with no live Lean integration or propagation results. Historical pre-read Lean usage scores were not captured; earlier priority-read studies provide separate usage/distributed controls. Results do not establish individual-token usefulness, generalization or a full-video gain.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))

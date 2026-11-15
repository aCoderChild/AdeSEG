#!/usr/bin/env python3
"""Freeze automatic-prompt feature scores for existing deleted token groups."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast
from modeling.lean_rgb_contrast_control import rgb_reference,rgb_contrast
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('capture','replay','replay-verification','training','training-verification','dataset','preflight','merged','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());cap=read(a.capture/'source_capture_receipt.json');u=read(a.capture/'utility_capture_receipt.json');r=read(a.replay/'completion.json');v=read(a.replay_verification/'verification.json');tr=read(a.training/'training_receipt.json');tv=read(a.training_verification/'verification.json');ds=read(a.dataset/'dataset.json');pre=read(a.preflight/'manifest.json');seq=u['sequence']
if seq not in ('seq2','seq3','seq4','seq9','seq10') or not cap['passed'] or not r['passed'] or not v['passed'] or r['sequence']!=seq or r['capture_sha256']!=digest(a.capture/'source_capture_receipt.json') or v['first_sha256']!=digest(a.replay/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_deletion.py') or not tr['passed'] or not tv['passed'] or tv['first_receipt_sha256']!=digest(a.training/'training_receipt.json') or tv['dataset_sha256']!=digest(a.dataset/'dataset.json') or tv['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_repeat.py') or ds['backbone_sha256']!=digest(a.checkpoint) or ds['preflight_sha256']!=digest(a.preflight/'manifest.json'):
 raise ValueError('Feature, training or replay provenance differs')
ckpt=a.training/'prompt_conditioned.pt'
if digest(ckpt)!=tr['arms']['prompt_conditioned']['checkpoint_sha256']:raise ValueError('Trained gate checkpoint changed')
for group in ('source_features','source_snapshots'):
 if not cap[group] or any(digest(a.capture/group/n)!=h for n,h in cap[group].items()):raise ValueError('Source capture changed')
source=Path(read(a.merged/'locations.json')[seq]);record=read(source/'records.json')[0];done=read(source/'completion.json');rows=read(source/seq/'frames.json')
if digest(source/'records.json')!=done['records_sha256'] or digest(source/seq/'frames.json')!=record['frames_sha256']:raise ValueError('Original input metadata changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[seq];prompt=read(source/seq/'prompt.json');prompt_path=source/seq/'prompt_refined.png'
if digest(prompt_path)!=prompt['refined_mask_sha256']:raise ValueError('Prompt changed')
image_hashes={}
def image(i):
 path=frames[i].image_path;sha=digest(path)
 if sha!=rows[i]['image_sha256']:raise ValueError('Input image differs')
 image_hashes[str(i)]=sha
 with Image.open(path) as im:return np.array(im.convert('RGB'),copy=True)
ref=rgb_reference(image(prompt['first_detection_index']),np.asarray(Image.open(prompt_path))>0)
torch.set_num_threads(2);gate=PromptContrastChannelAttention(checkpoint=ckpt,expected_backbone_sha256=ds['backbone_sha256'],expected_feature_pipeline_sha256=ds['feature_pipeline_sha256']).eval().requires_grad_(False)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);events=[];cache={}
with torch.inference_mode():
 for e in r['events']:
  if 'groups' not in e:continue
  frame=e['frame'];snap=torch.load(a.capture/'source_snapshots'/f'{frame:06d}.pt',map_location='cpu',weights_only=True);current=torch.load(a.capture/'features'/f'{frame:06d}.pt',map_location='cpu',weights_only=True);fg,bg=current['foreground'],current['background'];maps={k:{} for k in ('cbam','unweighted','rgb','shuffled','age')}
  for s in snap['sources']:
   if s['frame']>=frame-1:continue
   i=s['frame'];fpath=a.capture/'source_features'/f'{i:06d}.pt';entry=torch.load(fpath,map_location='cpu',weights_only=True)
   if entry['frame']!=i:raise ValueError('Source feature identity differs')
   feature=entry['feature'];gate_score,_=gate(feature,fg,bg);plain=blend_anchor_with_early_contrast(torch.zeros(4096),feature,fg,bg,(64,64),1.);rgb=rgb_contrast(image(i),ref)
   seed=int.from_bytes(hashlib.sha256(f'lean_group_utility_score_diagnostic_v1:{seq}:{frame}:{i}'.encode()).digest()[:16],'big');permutation=np.random.Generator(np.random.PCG64(seed)).permutation(4096)
   ids=s['token_ids'].long();start=s['start'];indices=range(start,start+len(ids))
   for name,vals in (('cbam',gate_score[ids]),('unweighted',plain[ids]),('rgb',rgb[ids]),('shuffled',plain[torch.from_numpy(permutation)[ids]]),('age',torch.full((len(ids),),float(frame-i)))):
    maps[name].update(zip(indices,vals.tolist(),strict=True))
  scores={name:[float(np.mean([mapping[j] for j in group])) for group in e['groups']] for name,mapping in maps.items()}
  if any(not np.isfinite(vals).all() for vals in scores.values()):raise ValueError('Nonfinite group scores')
  events.append({'frame':frame,'scores':scores})
result={'passed':True,'sequence':seq,'events':events,'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.replay_verification/'verification.json'),'capture_sha256':digest(a.capture/'source_capture_receipt.json'),'training_sha256':digest(a.training/'training_receipt.json'),'gate_sha256':digest(ckpt),'prompt_sha256':digest(prompt_path),'image_sha256':image_hashes,'protocol_sha256':digest(ROOT/'protocols/lean_group_utility_score_diagnostic_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Scores only; no deletion outcome or target mask opened.'}
(out/'scores.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'reads':len(events)}))

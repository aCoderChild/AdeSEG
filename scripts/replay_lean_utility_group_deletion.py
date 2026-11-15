#!/usr/bin/env python3
"""Frozen-read matched group deletions; no target labels are opened."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_memory_capture import clone_data
from modeling.lean_utility_group_deletion import partition_history,delete_group
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED
from scripts.run_lean_offline_memory_ablation import equal
p=argparse.ArgumentParser()
for n in ('capture','source-verification','model-verification','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());u=read(a.capture/'utility_capture_receipt.json');r=read(a.capture/'source_capture_receipt.json');v=read(a.source_verification/'verification.json');m=read(a.model_verification/'verification.json')
if not u['passed'] or not r['passed'] or not v['passed'] or r['nested_receipt_sha256']!=digest(a.capture/'utility_capture_receipt.json') or r['observer_sha256']!=digest(ROOT/'modeling/lean_utility_source_capture.py') or not m['passed'] or m['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or m['identity']['checkpoint_sha256']!=digest(a.checkpoint) or any(m['identity']['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Invalid capture or model provenance')
if u['sequence']=='seq3':
 if v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_source_gates.py') or digest(a.capture/'source_capture_receipt.json') not in (v['first_sha256'],v['second_sha256']):raise ValueError('Engineering source gate invalid')
elif u['sequence'] in ('seq2','seq4','seq9','seq10'):
 if v['runner_sha256']!=digest(ROOT/'scripts/verify_lean_utility_source_candidate.py') or v['sequence']!=u['sequence'] or v['source_receipt_sha256']!=digest(a.capture/'source_capture_receipt.json'):raise ValueError('Candidate source audit invalid')
else:raise ValueError('Sequence outside protocol split')
for group in ('source_features','source_snapshots'):
 if not r[group] or any(digest(a.capture/group/n)!=h for n,h in r[group].items()):raise ValueError('Captured source data changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);events=[]
shots={z['frame']:z['sha256'] for z in u['snapshots']}
with torch.inference_mode(),indexed_spatial_rope(model,[]):
 for event in r['observations']:
  frame=event['frame'];path=a.capture/'snapshots'/f'{frame:06d}.pt'
  if digest(path)!=shots[frame]:raise ValueError('Read snapshot changed')
  data=torch.load(path,map_location='cpu',weights_only=True);src=torch.load(a.capture/'source_snapshots'/path.name,map_location='cpu',weights_only=True)
  if src['frame']!=frame:raise ValueError('Source frame mismatch')
  groups=partition_history(src['sources'],frame,u['sequence'])
  if groups is None:
   events.append({'frame':frame,'excluded':'eligible historical key count differs from 5120'});continue
  args=data['attention_inputs'];ptr=int(args['num_obj_ptr_tokens'])
  if ptr!=src['pointer_count'] or not equal(model.memory_attention(**args),data['attention_outputs']) or not equal(model._forward_sam_heads(**data['decoder_inputs']),data['decoder_outputs']):raise ValueError('Frozen state differs')
  masks={};key_counts={}
  for arm in range(-1,5):
   call=clone_data(args);name='noop' if arm==-1 else f'group{arm}'
   if arm>=0:call['memory'],call['memory_pos']=delete_group(call['memory'],call['memory_pos'],ptr,groups[arm])
   key_counts[name]=len(call['memory']);features=model.memory_attention(**call)[0];head=clone_data(data['decoder_inputs']);head['backbone_features']=features.permute(1,2,0).view(1,features.shape[2],64,64);result=model._forward_sam_heads(**head)
   if arm==-1 and not equal(result,data['decoder_outputs']):raise ValueError('No-op decoder changed')
   _,video=model._get_orig_video_res_output({'device':torch.device('cpu'),'video_height':data['video_height'],'video_width':data['video_width']},result[3]);mask=(video[0,0].numpy()>0).astype(np.uint8)*255
   if arm==-1 and not np.array_equal(mask>0,np.asarray(Image.open(a.capture/f'{frame:06d}.png'))>0):raise ValueError('No-op mask changed')
   path=out/f'{frame:06d}_{name}.png';Image.fromarray(mask).save(path);masks[name]=digest(path)
  if len({key_counts[f'group{i}'] for i in range(5)})!=1 or key_counts['noop']-key_counts['group0']!=1024:raise ValueError('Deletion budgets differ')
  events.append({'frame':frame,'groups':groups,'key_counts':key_counts,'masks':masks})
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'sequence':u['sequence'],'events':events,'capture_sha256':digest(a.capture/'source_capture_receipt.json'),'source_verification_sha256':digest(a.source_verification/'verification.json'),'model_verification_sha256':digest(a.model_verification/'verification.json'),'protocol_sha256':digest(ROOT/'protocols/lean_utility_group_deletion_v1.json'),'partition_sha256':digest(ROOT/'modeling/lean_utility_group_deletion.py'),'runner_sha256':digest(Path(__file__)),'meaning':'Frozen local deletions only; no labels were opened.'}
(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':u['sequence'],'eligible_frames':[e['frame'] for e in events if 'groups' in e]}))

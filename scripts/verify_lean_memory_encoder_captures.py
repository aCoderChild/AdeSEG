#!/usr/bin/env python3
"""Verify cropped encoder captures and their stored next-read memory blocks."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK,PATCHED
p=argparse.ArgumentParser()
for k in ('recent-encoder','spread-encoder','recent-read','spread-read','recent-reference','spread-reference','capture-verification','model-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());gate=read(a.capture_verification/'verification.json');mv=read(a.model_verification/'verification.json');ident=mv['identity']
if not gate['passed'] or gate['runner_sha256']!=digest(ROOT/'scripts/capture_lean_memory_encoder_inputs.py') or gate['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or gate['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_memory_encoder_capture_gates.py') or not mv['passed'] or mv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Engineering or model verification changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);result={}
with torch.inference_mode():
 for policy,enc,reads,reference in (('recent',a.recent_encoder,a.recent_read,a.recent_reference),('spread',a.spread_encoder,a.spread_read,a.spread_reference)):
  c=read(enc/'encoder_capture_receipt.json');rf='recent_capture_receipt.json' if policy=='recent' else 'spread_capture_receipt.json';rc=read(reads/rf);orig=read(reference/'token_window_observation.json')
  if not all(x['passed'] for x in (c,rc,orig)) or c['policy']!=policy or c['entry']!='crop' or c['targets']!=[47,50] or c['runner_sha256']!=gate['runner_sha256'] or c['module_sha256']!=gate['module_sha256'] or c['protocol_sha256']!=gate['protocol_sha256'] or c['reference_observation_sha256']!=digest(reference/'token_window_observation.json') or c['output_observation_sha256']!=digest(enc/'token_window_observation.json'):raise ValueError('Captured source identity differs')
  for name in ('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json'):
   if digest(enc/name)!=digest(reference/name):raise ValueError('Captured trajectory trace differs')
  if any(digest(enc/f'{i:06d}.png')!=digest(reference/f'{i:06d}.png') for i in range(56)):raise ValueError('Captured trajectory mask differs')
  for source in (47,50):
   ep=enc/'encoder_snapshots'/f'{source:06d}.pt';rp=reads/'snapshots'/f'{source+1:06d}.pt';eh=next(z['read_sha256'] for z in c['snapshots'] if z['frame']==source);rh=next(z['read_sha256'] for z in rc['snapshots'] if z['frame']==source+1)
   if digest(ep)!=eh or digest(rp)!=rh:raise ValueError('Captured snapshot differs')
   es=torch.load(ep,map_location='cpu',weights_only=True);rs=torch.load(rp,map_location='cpu',weights_only=True);out0=model._encode_new_memory(**es['inputs']);inp=rs['attention_inputs'];end=len(inp['memory'])-int(inp['num_obj_ptr_tokens']);stored=out0[0].to(torch.bfloat16).flatten(2).permute(2,0,1).to(inp['memory'].dtype);position=out0[1][-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0]
   if not equal(out0,es['outputs']) or end!=16384 or stored.shape!=(4096,1,model.mem_dim) or not torch.equal(stored,inp['memory'][end-4096:end]) or not torch.equal(position,inp['memory_pos'][end-4096:end]):raise ValueError('Exact next-read bfloat16 block differs')
  result[policy]={'encoder_receipt_sha256':digest(enc/'encoder_capture_receipt.json'),'read_receipt_sha256':digest(reads/rf),'reference_observation_sha256':digest(reference/'token_window_observation.json')}
assert_parameter_integrity(model,before,out,'after')
(out/'verification.json').write_text(json.dumps({'passed':True,'policies':result,'capture_gate_sha256':digest(a.capture_verification/'verification.json'),'checkpoint_sha256':digest(a.checkpoint),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Both full trajectories and exact source-to-read bfloat16 blocks independently verified.')

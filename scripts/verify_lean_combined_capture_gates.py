#!/usr/bin/env python3
"""Verify nested read and encoder observers against separately audited captures."""
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
for k in ('recent-first','recent-second','recent-encoder-reference','recent-read-reference','spread-first','spread-second','spread-encoder-reference','spread-read-reference','model-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());mv=read(a.model_verification/'verification.json');ident=mv['identity']
if not mv['passed'] or mv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Model identity changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);summaries={}
with torch.inference_mode():
 for policy,first,second,enc_ref,read_ref in (('recent',a.recent_first,a.recent_second,a.recent_encoder_reference,a.recent_read_reference),('spread',a.spread_first,a.spread_second,a.spread_encoder_reference,a.spread_read_reference)):
  rf='recent_capture_receipt.json' if policy=='recent' else 'spread_capture_receipt.json';ec=read(first/'encoder_capture_receipt.json');ec2=read(second/'encoder_capture_receipt.json');rc=read(first/rf);rc2=read(second/rf);oldenc=read(enc_ref/'encoder_capture_receipt.json');oldread=read(read_ref/rf)
  if ec!=ec2 or rc!=rc2 or not all(z['passed'] for z in (ec,rc,oldenc,oldread)) or ec['policy']!=policy or ec['entry']!='gate' or ec['targets']!=[7,11] or rc['targets']!=[8,12] or ec['runner_sha256']!=digest(ROOT/'scripts/capture_lean_encoder_and_read.py') or ec['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or rc['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or ec['protocol_sha256']!=digest(ROOT/'protocols/lean_encoder_empty_controls_v1.json'):raise ValueError('Combined capture receipt differs')
  for source in (7,11):
   target=source+1;ep=first/'encoder_snapshots'/f'{source:06d}.pt';ep2=second/'encoder_snapshots'/f'{source:06d}.pt';rp=first/'snapshots'/f'{target:06d}.pt';rp2=second/'snapshots'/f'{target:06d}.pt'
   for path,receipt,frame in ((ep,ec,source),(ep2,ec,source),(rp,rc,target),(rp2,rc,target)):
    if digest(path)!=next(z['read_sha256'] for z in receipt['snapshots'] if z['frame']==frame):raise ValueError('Snapshot hash differs')
   es=torch.load(ep,map_location='cpu',weights_only=True);rs=torch.load(rp,map_location='cpu',weights_only=True)
   if not equal(es,torch.load(ep2,map_location='cpu',weights_only=True)) or not equal(rs,torch.load(rp2,map_location='cpu',weights_only=True)) or not equal(es,torch.load(enc_ref/'encoder_snapshots'/f'{source:06d}.pt',map_location='cpu',weights_only=True)) or not equal(rs,torch.load(read_ref/'snapshots'/f'{target:06d}.pt',map_location='cpu',weights_only=True)):raise ValueError('Nested observer changed the captured tensors')
   out0=model._encode_new_memory(**es['inputs']);inp=rs['attention_inputs'];end=len(inp['memory'])-int(inp['num_obj_ptr_tokens']);stored=out0[0].to(torch.bfloat16).flatten(2).permute(2,0,1).to(inp['memory'].dtype);pos=out0[1][-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0]
   if not equal(out0,es['outputs']) or end!=16384 or not torch.equal(stored,inp['memory'][end-4096:end]) or not torch.equal(pos,inp['memory_pos'][end-4096:end]):raise ValueError('Stored bfloat16 next-read block differs')
  summaries[policy]={'encoder_receipt_sha256':digest(first/'encoder_capture_receipt.json'),'read_receipt_sha256':digest(first/rf)}
assert_parameter_integrity(model,before,out,'after')
(out/'verification.json').write_text(json.dumps({'passed':True,'policies':summaries,'runner_sha256':digest(ROOT/'scripts/capture_lean_encoder_and_read.py'),'encoder_capture_sha256':digest(ROOT/'modeling/lean_memory_encoder_capture.py'),'read_capture_sha256':digest(ROOT/'modeling/lean_memory_capture.py'),'protocol_sha256':digest(ROOT/'protocols/lean_encoder_empty_controls_v1.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Nested observers and exact bfloat16 source-to-read reconstruction verified.')

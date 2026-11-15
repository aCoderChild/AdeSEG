#!/usr/bin/env python3
"""Verify repeated encoder captures and exact bfloat16 newest-key storage."""
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
for k in ('recent-first','recent-second','recent-read','spread-first','spread-second','spread-read','model-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());v=read(a.model_verification/'verification.json');ident=v['identity']
if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Model identity changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);result={}
with torch.inference_mode():
 for policy,first,second,read_capture in (('recent',a.recent_first,a.recent_second,a.recent_read),('spread',a.spread_first,a.spread_second,a.spread_read)):
  x=read(first/'encoder_capture_receipt.json');y=read(second/'encoder_capture_receipt.json');readfile='recent_capture_receipt.json' if policy=='recent' else 'spread_capture_receipt.json';rc=read(read_capture/readfile)
  if x!=y or not x['passed'] or not rc['passed'] or x['entry']!='gate' or x['policy']!=policy or x['targets']!=[7,11] or rc['targets']!=[8,12] or x['runner_sha256']!=digest(ROOT/'scripts/capture_lean_memory_encoder_inputs.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json'):raise ValueError('Encoder capture provenance differs')
  hashes={}
  for item in x['snapshots']:
   source=item['frame'];p0=first/'encoder_snapshots'/f'{source:06d}.pt';p1=second/'encoder_snapshots'/f'{source:06d}.pt';target=read_capture/'snapshots'/f'{source+1:06d}.pt';targetsha=next(z['read_sha256'] for z in rc['snapshots'] if z['frame']==source+1)
   if digest(p0)!=item['read_sha256'] or digest(p1)!=item['read_sha256'] or digest(target)!=targetsha:raise ValueError('Saved encoder or read snapshot changed')
   snap=torch.load(p0,map_location='cpu',weights_only=True)
   if not equal(snap,torch.load(p1,map_location='cpu',weights_only=True)) or snap['frame']!=source:raise ValueError('Repeated encoder inputs differ')
   computed=model._encode_new_memory(**snap['inputs'])
   if not equal(computed,snap['outputs']):raise ValueError('Author encoder replay differs')
   readshot=torch.load(target,map_location='cpu',weights_only=True);inp=readshot['attention_inputs'];ptr=int(inp['num_obj_ptr_tokens']);end=len(inp['memory'])-ptr
   expected=computed[0].to(torch.bfloat16).flatten(2).permute(2,0,1).to(inp['memory'].dtype)
   position=computed[1][-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0]
   if end!=16384 or expected.shape!=(4096,1,model.mem_dim) or not torch.equal(inp['memory'][end-4096:end],expected) or not torch.equal(inp['memory_pos'][end-4096:end],position):raise ValueError('Encoder output does not reconstruct saved newest read')
   hashes[str(source)]=item['read_sha256']
  result[policy]={'snapshots':hashes,'first_receipt_sha256':digest(first/'encoder_capture_receipt.json'),'second_receipt_sha256':digest(second/'encoder_capture_receipt.json'),'read_receipt_sha256':digest(read_capture/readfile)}
assert_parameter_integrity(model,before,out,'after')
(out/'verification.json').write_text(json.dumps({'passed':True,'policies':result,'runner_sha256':digest(ROOT/'scripts/capture_lean_memory_encoder_inputs.py'),'module_sha256':digest(ROOT/'modeling/lean_memory_encoder_capture.py'),'protocol_sha256':digest(ROOT/'protocols/lean_newest_memory_component_v1.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All encoder snapshots and exact bfloat16 newest-memory reconstructions verified.')

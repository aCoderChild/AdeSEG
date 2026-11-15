#!/usr/bin/env python3
"""Replay frozen spread reads with controlled learned-temporal-slot substitutions."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_temporal_rank_positions import remap_temporal_positions
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_temporal_spread_window import FORK,PATCHED

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
for k in ('capture','capture-verification','model-verification','replay-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();c=read(a.capture/'spread_capture_receipt.json');cv=read(a.capture_verification/'verification.json');mv=read(a.model_verification/'verification.json');expected=[8,12] if a.mode=='gate' else [8,20,35,48,51];entry='gate' if a.mode=='gate' else 'crop'
if not all(x['passed'] for x in (c,cv,mv)) or c['entry']!=entry or c['targets']!=expected or c['runner_sha256']!=digest(ROOT/'scripts/capture_lean_temporal_spread_reads.py') or c['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or c['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json') or cv['runner_sha256']!=c['runner_sha256'] or cv['capture_sha256']!=c['capture_sha256'] or cv['protocol_sha256']!=c['protocol_sha256'] or cv['verifier_sha256']!=digest(ROOT/'scripts/verify_temporal_spread_capture_gates.py'):raise ValueError('Capture or engineering scope changed')
ident=mv['identity']
if mv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_temporal_spread_window.py') or any(digest(FORK/k)!=ident['patch_sha256'][k] for k in PATCHED):raise ValueError('Model or runtime changed')
if a.mode=='candidate':
 rv=read(a.replay_verification/'verification.json')
 if not rv['passed'] or rv['runner_sha256']!=digest(Path(__file__)) or rv['module_sha256']!=digest(ROOT/'modeling/lean_temporal_rank_positions.py') or rv['protocol_sha256']!=c['protocol_sha256'] or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_slot_rank_gates.py'):raise ValueError('Rank-replay engineering gates changed')
events=read(a.capture/'token_window_events.json');ob=read(a.capture/'token_window_observation.json')
if ob['events_sha256']!=digest(a.capture/'token_window_events.json') or c['output_observation_sha256']!=digest(a.capture/'token_window_observation.json'):raise ValueError('Captured trace changed')
plans={x['frame']:x for x in events if x['stage']=='admission_plan'};reads={x['frame']:x for x in events if x['stage']=='after_read'};shots={x['frame']:x['read_sha256'] for x in c['snapshots']}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
import sam2
if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Unexpected model worktree')
before=parameter_fingerprints(model);rope=[];result_rows=[]
with torch.inference_mode():
 grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous();offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();lookup=MemoryPositionLookup(grid,offsets)
 with indexed_spatial_rope(model,rope):
  for frame in expected:
   path=a.capture/'snapshots'/f'{frame:06d}.pt'
   if digest(path)!=shots[frame]:raise ValueError('Snapshot changed')
   snapshot=torch.load(path,map_location='cpu',weights_only=True);original=snapshot['attention_inputs'];ptr=int(original['num_obj_ptr_tokens']);pos=original['memory_pos'];indices=lookup.decode(pos,ptr);plan=plans[frame-1];countmap={z['frame']:z['tokens'] for z in plan['candidate_entries']};history=[(i,countmap[i]) for i in plan['selected_frames']];insurance=reads[frame]['insurance_tokens']
   if frame!=snapshot['frame'] or not equal(model.memory_attention(**original),snapshot['attention_outputs']) or not equal(model._forward_sam_heads(**snapshot['decoder_inputs']),snapshot['decoder_outputs']):raise ValueError('Snapshot attention or decoder differs')
   masks={};rotary={};segments={};position_sha={}
   for mode in ('noop','rank','permuted'):
    newpos,detail=remap_temporal_positions(pos,indices,grid,offsets,frame,history,insurance,ptr,mode);args=dict(original);args['memory_pos']=newpos
    if not torch.equal(args['memory'],original['memory']) or len(args['memory'])!=len(original['memory']) or args['num_obj_ptr_tokens']!=ptr:raise ValueError('Memory values or count changed')
    position_sha[mode]=hashlib.sha256(newpos.contiguous().numpy().tobytes()).hexdigest()
    attention=model.memory_attention(**args);rotary[mode]=rope[-1];head=clone_data(snapshot['decoder_inputs']);features=attention[0];head['backbone_features']=features.permute(1,2,0).view(1,features.shape[2],64,64);pred=model._forward_sam_heads(**head)
    if mode=='noop' and (not equal(attention,snapshot['attention_outputs']) or not equal(pred,snapshot['decoder_outputs'])):raise ValueError('No-op replay changed')
    state={'device':torch.device('cpu'),'video_height':snapshot['video_height'],'video_width':snapshot['video_width']};_,video=model._get_orig_video_res_output(state,pred[3]);mask=video[0,0].cpu().numpy()>0
    if entry=='crop':mask=restore_mask(mask)
    dest=out/f'{frame:06d}_{mode}.png';Image.fromarray(mask.astype(np.uint8)*255).save(dest);masks[mode]=digest(dest);segments[mode]=detail
    if mode=='noop' and not np.array_equal(mask,np.asarray(Image.open(a.capture/f'{frame:06d}.png'))>0):raise ValueError('No-op saved mask differs')
   if any(rotary[x]['indices_sha256']!=rotary['noop']['indices_sha256'] for x in ('rank','permuted')):raise ValueError('Spatial RoPE indices changed')
   ranked=[x['assigned_slot'] for x in segments['rank']];permuted=[x['assigned_slot'] for x in segments['permuted']]
   if sorted(ranked)!=sorted(permuted) or ranked[-1]!=0 or permuted[-1]!=0:raise ValueError('Rank control slot multiset differs')
   result_rows.append({'frame':frame,'total_keys':len(original['memory']),'pointer_keys':ptr,'insurance_tokens':insurance,'history':history,'masks':masks,'rotary':rotary,'position_sha256':position_sha,'segments':segments})
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'mode':a.mode,'events':result_rows,'capture_receipt_sha256':digest(a.capture/'spread_capture_receipt.json'),'model_verification_sha256':digest(a.model_verification/'verification.json'),'checkpoint_sha256':digest(a.checkpoint),'module_sha256':digest(ROOT/'modeling/lean_temporal_rank_positions.py'),'protocol_sha256':digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Frozen-read learned-slot substitution only; no labels or feedback.'}
(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'frames':expected}))

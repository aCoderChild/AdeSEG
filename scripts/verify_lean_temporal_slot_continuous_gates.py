#!/usr/bin/env python3
"""Independently replay and verify raw and post-presence decoder tensors."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_temporal_spread_window import FORK
p=argparse.ArgumentParser()
for k in ('first','second','capture','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());x=read(a.first/'completion.json');y=read(a.second/'completion.json');c=read(a.capture/'spread_capture_receipt.json');mode=x['mode'];expected=[8,12] if mode=='gate' else [8,20,35,48,51]
if x!=y or not all(z['passed'] for z in (x,c)) or mode not in ('gate','candidate') or c['entry']!=('gate' if mode=='gate' else 'crop') or x['capture_receipt_sha256']!=digest(a.capture/'spread_capture_receipt.json') or x['checkpoint_sha256']!=digest(a.checkpoint) or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_temporal_slot_continuous.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_temporal_rank_positions.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_continuous_diagnostic_v1.json') or [e['frame'] for e in x['events']]!=expected:raise ValueError('Replay provenance or repetition differs')
torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);observations=[]
with torch.inference_mode():
 grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous();offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();lookup=MemoryPositionLookup(grid,offsets)
 with indexed_spatial_rope(model,observations):
  for e in x['events']:
   frame=e['frame'];snapfile=a.capture/'snapshots'/f'{frame:06d}.pt';h=next(z['read_sha256'] for z in c['snapshots'] if z['frame']==frame)
   if digest(snapfile)!=h:raise ValueError('Snapshot changed')
   snap=torch.load(snapfile,map_location='cpu',weights_only=True);args=snap['attention_inputs'];pos=args['memory_pos'];ptr=int(args['num_obj_ptr_tokens']);indices=lookup.decode(pos,ptr);history=e['history'];start=4096+e['insurance_tokens'];end=len(pos)-ptr
   if len(pos)!=e['total_keys'] or ptr!=e['pointer_keys'] or start+sum(z[1] for z in history)!=end:raise ValueError('Key geometry differs')
   for arm in ('noop','rank','permuted'):
    newpos=pos.clone();cursor=start
    for j,(source,n) in enumerate(history):
     actual=min(frame-source,6)-1;assigned=actual if arm=='noop' else (len(history)-1-j if arm=='rank' else (j+1 if j<len(history)-1 else 0));spatial=grid[indices[cursor:cursor+n]]
     if not torch.equal(pos[cursor:cursor+n],(spatial+offsets[actual]).unsqueeze(1)):raise ValueError('Original source positions differ')
     newpos[cursor:cursor+n]=(spatial+offsets[assigned]).unsqueeze(1);cursor+=n
    if hashlib.sha256(newpos.contiguous().numpy().tobytes()).hexdigest()!=e['position_sha256'][arm]:raise ValueError('Position tensor differs')
    call=dict(args);call['memory_pos']=newpos;attention=model.memory_attention(**call);rot=observations[-1]
    if rot!=e['rotary'][arm] or rot['indices_sha256']!=e['rotary']['noop']['indices_sha256']:raise ValueError('Spatial rotary indices differ')
    head=clone_data(snap['decoder_inputs']);z=attention[0];head['backbone_features']=z.permute(1,2,0).view(1,z.shape[2],64,64);raw=[];hook=model.sam_mask_decoder.register_forward_hook(lambda _m,_i,value:raw.append(clone_data(value)))
    try:pred=model._forward_sam_heads(**head)
    finally:hook.remove()
    if len(raw)!=1:raise ValueError('Raw decoder not captured exactly once')
    computed={'attention':clone_data(attention[0]),'raw_masks':raw[0][0],'object_logits':clone_data(pred[6]),'post_masks':clone_data(pred[3])}
    if not torch.equal(raw[0][3],pred[6]):raise ValueError('Object logits differ')
    if arm=='noop' and (not equal(attention,snap['attention_outputs']) or not equal(pred,snap['decoder_outputs'])):raise ValueError('No-op differs from captured original')
    for folder in (a.first,a.second):
     tensor=folder/f'{frame:06d}_{arm}.pt';mask=folder/f'{frame:06d}_{arm}.png'
     if digest(tensor)!=e['tensors'][arm] or digest(mask)!=e['masks'][arm] or not equal(torch.load(tensor,map_location='cpu',weights_only=True),computed):raise ValueError('Saved raw or post-suppression tensors differ')
   if e['masks']['noop']!=digest(a.capture/f'{frame:06d}.png'):raise ValueError('No-op mask differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);assert_parameter_integrity(model,before,out,'after')
(out/'verification.json').write_text(json.dumps({'passed':True,'mode':mode,'frames':expected,'runner_sha256':x['runner_sha256'],'protocol_sha256':x['protocol_sha256'],'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All positions, raw decoder tensors, native object logits and post-suppression tensors independently replayed.')

#!/usr/bin/env python3
"""Exchange matched memory segments between frozen recent and spread reads."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_recent_spread_exchange import exchange_memory
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK,PATCHED

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
for k in ('recent-capture','spread-capture','recent-capture-verification','spread-capture-verification','model-verification','replay-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();recent=read(a.recent_capture/'recent_capture_receipt.json');spread=read(a.spread_capture/'spread_capture_receipt.json');rv=read(a.recent_capture_verification/'verification.json');sv=read(a.spread_capture_verification/'verification.json');mv=read(a.model_verification/'verification.json');expected=[8,12] if a.mode=='gate' else [8,20,35,48,51];entry='gate' if a.mode=='gate' else 'crop'
if not all(x['passed'] for x in (recent,spread,rv,sv,mv)) or recent['entry']!=entry or spread['entry']!=entry or recent['targets']!=expected or spread['targets']!=expected or recent['runner_sha256']!=digest(ROOT/'scripts/capture_lean_recent_window_reads.py') or recent['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or recent['protocol_sha256']!=digest(ROOT/'protocols/lean_recent_spread_memory_exchange_v1.json') or rv['runner_sha256']!=recent['runner_sha256'] or rv['capture_sha256']!=recent['capture_sha256'] or rv['protocol_sha256']!=recent['protocol_sha256'] or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_recent_window_capture_gates.py'):raise ValueError('Recent capture or gates changed')
if spread['runner_sha256']!=digest(ROOT/'scripts/capture_lean_temporal_spread_reads.py') or spread['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or spread['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json') or sv['runner_sha256']!=spread['runner_sha256'] or sv['capture_sha256']!=spread['capture_sha256'] or sv['protocol_sha256']!=spread['protocol_sha256'] or sv['verifier_sha256']!=digest(ROOT/'scripts/verify_temporal_spread_capture_gates.py'):raise ValueError('Spread capture or gates changed')
ident=mv['identity']
if mv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(digest(FORK/k)!=ident['patch_sha256'][k] for k in PATCHED):raise ValueError('Model worktree changed')
if a.mode=='candidate':
 v=read(a.replay_verification/'verification.json')
 if not v['passed'] or v['runner_sha256']!=digest(Path(__file__)) or v['module_sha256']!=digest(ROOT/'modeling/lean_recent_spread_exchange.py') or v['protocol_sha256']!=recent['protocol_sha256'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_recent_spread_exchange_gates.py'):raise ValueError('Exchange engineering gates changed')
 if digest(a.recent_capture/'prompt.json')!=digest(a.spread_capture/'prompt.json') or digest(a.recent_capture/'prompt.png')!=digest(a.spread_capture/'prompt.png'):raise ValueError('Initial crop prompt differs')
traces={}
for name,folder,c in (('recent',a.recent_capture,recent),('spread',a.spread_capture,spread)):
 o=read(folder/'token_window_observation.json');events=read(folder/'token_window_events.json')
 if digest(folder/'token_window_events.json')!=o['events_sha256'] or digest(folder/'token_window_observation.json')!=c['output_observation_sha256']:raise ValueError('Captured events changed')
 traces[name]={'plans':{x['frame']:x for x in events if x['stage']=='admission_plan'},'reads':{x['frame']:x for x in events if x['stage']=='after_read'},'shots':{x['frame']:x['read_sha256'] for x in c['snapshots']}}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
import sam2
if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Unexpected Lean worktree')
before=parameter_fingerprints(model);rope=[];rows=[]
with torch.inference_mode():
 grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous();offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();lookup=MemoryPositionLookup(grid,offsets)
 with indexed_spatial_rope(model,rope):
  for frame in expected:
   data={};histories={}
   for name,folder in (('recent',a.recent_capture),('spread',a.spread_capture)):
    path=folder/'snapshots'/f'{frame:06d}.pt'
    if digest(path)!=traces[name]['shots'][frame]:raise ValueError('Snapshot changed')
    snap=torch.load(path,map_location='cpu',weights_only=True);inp=snap['attention_inputs'];ptr=int(inp['num_obj_ptr_tokens']);end=len(inp['memory'])-ptr;indices=lookup.decode(inp['memory_pos'],ptr);t=traces[name];plan=t['plans'][frame-1];ids=plan['selected_frames'];sizes={z['frame']:z['tokens'] for z in plan['candidate_entries']} if name=='spread' else {i:(plan['new_tokens'] if i==frame-1 else 1024) for i in ids};insurance=t['reads'][frame]['insurance_tokens'];cursor=4096+insurance;history=[]
    if snap['frame']!=frame or end!=16384 or insurance!=3072 or len(ids)!=6 or ids[-1]!=frame-1 or sum(sizes[i] for i in ids)!=end-cursor:raise ValueError('History or insurance layout differs')
    for i in ids:
     n=sizes[i];slot=min(frame-i,6)-1;expected_pos=(grid[indices[cursor:cursor+n]]+offsets[slot]).unsqueeze(1)
     if not torch.equal(inp['memory_pos'][cursor:cursor+n],expected_pos):raise ValueError('Source age/position reconstruction differs')
     history.append({'frame':i,'tokens':n,'start':cursor,'end':cursor+n,'slot':slot});cursor+=n
    if history[-1]['tokens']!=4096 or history[-1]['start']!=end-4096:raise ValueError('Newest entry not dense final block')
    if not equal(model.memory_attention(**inp),snap['attention_outputs']) or not equal(model._forward_sam_heads(**snap['decoder_inputs']),snap['decoder_outputs']):raise ValueError('No-op attention or decoder differs')
    data[name]=snap;histories[name]=history
   aa=data['recent']['attention_inputs'];bb=data['spread']['attention_inputs'];da=data['recent']['decoder_inputs'];db=data['spread']['decoder_inputs']
   if set(da)!=set(db):raise ValueError('Decoder input schemas differ')
   same_query=equal(aa['curr'],bb['curr']) and equal(aa['curr_pos'],bb['curr_pos'])
   same_decoder_inputs=all(equal(da[k],db[k]) for k in da if k!='backbone_features')
   # Each arm preserves its recipient query and decoder inputs; only memory is exchanged.
   calls={'recent_noop':aa,'spread_noop':bb}
   for receiver,donor in (('recent','spread'),('spread','recent')):
    for part in ('spatial','newest','pointers'):
     calls[f'{receiver}_with_{donor}_{part}']=exchange_memory(data[receiver]['attention_inputs'],data[donor]['attention_inputs'],part)
   masks={};objects={};rotary={};inputs={}
   for arm,args in calls.items():
    receiver='recent' if arm.startswith('recent_') else 'spread';snap=data[receiver];inp=snap['attention_inputs'];attention=model.memory_attention(**args);rotary[arm]=rope[-1];head=clone_data(snap['decoder_inputs']);feat=attention[0];head['backbone_features']=feat.permute(1,2,0).view(1,feat.shape[2],64,64);prediction=model._forward_sam_heads(**head)
    if arm.endswith('_noop') and (not equal(attention,snap['attention_outputs']) or not equal(prediction,snap['decoder_outputs'])):raise ValueError('No-op differs')
    objects[arm]=float(prediction[6].item());inputs[arm]={k:hashlib.sha256(args[k].contiguous().numpy().tobytes()).hexdigest() for k in ('memory','memory_pos')}
    state={'device':torch.device('cpu'),'video_height':snap['video_height'],'video_width':snap['video_width']};_,video=model._get_orig_video_res_output(state,prediction[3]);mask=video[0,0].cpu().numpy()>0
    if entry=='crop':mask=restore_mask(mask)
    dest=out/f'{frame:06d}_{arm}.png';Image.fromarray(mask.astype(np.uint8)*255).save(dest);masks[arm]=digest(dest)
    folder=a.recent_capture if receiver=='recent' else a.spread_capture
    if arm.endswith('_noop') and not np.array_equal(mask,np.asarray(Image.open(folder/f'{frame:06d}.png'))>0):raise ValueError('No-op mask differs')
   rows.append({'frame':frame,'total_keys':len(aa['memory']),'pointer_keys':int(aa['num_obj_ptr_tokens']),'history':histories,'same_current_query':same_query,'same_decoder_image_inputs':same_decoder_inputs,'masks':masks,'object_logits':objects,'rotary':rotary,'input_sha256':inputs})
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'mode':a.mode,'events':rows,'recent_capture_sha256':digest(a.recent_capture/'recent_capture_receipt.json'),'spread_capture_sha256':digest(a.spread_capture/'spread_capture_receipt.json'),'checkpoint_sha256':digest(a.checkpoint),'module_sha256':digest(ROOT/'modeling/lean_recent_spread_exchange.py'),'protocol_sha256':digest(ROOT/'protocols/lean_recent_spread_memory_exchange_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Matched-key local exchange across trajectories; no labels or feedback.'}
(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'frames':expected}))

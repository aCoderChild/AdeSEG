#!/usr/bin/env python3
"""Independently reconstruct memory exchanges and replay attention/decoder."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK
p=argparse.ArgumentParser()
for k in ('first','second','recent-capture','spread-capture','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());x=read(a.first/'completion.json');y=read(a.second/'completion.json');rc=read(a.recent_capture/'recent_capture_receipt.json');sc=read(a.spread_capture/'spread_capture_receipt.json');mode=x['mode'];frames=[8,12] if mode=='gate' else [8,20,35,48,51]
if x!=y or not all(z['passed'] for z in (x,rc,sc)) or mode not in ('gate','candidate') or x['recent_capture_sha256']!=digest(a.recent_capture/'recent_capture_receipt.json') or x['spread_capture_sha256']!=digest(a.spread_capture/'spread_capture_receipt.json') or x['checkpoint_sha256']!=digest(a.checkpoint) or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_recent_spread_exchange.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_recent_spread_exchange.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_recent_spread_memory_exchange_v1.json') or [e['frame'] for e in x['events']]!=frames:raise ValueError('Replay provenance or repetitions differ')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);observations=[]
with torch.inference_mode():
 grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous();offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();lookup=MemoryPositionLookup(grid,offsets)
 with indexed_spatial_rope(model,observations):
  for e in x['events']:
   frame=e['frame'];data={}
   for name,folder,receipt in (('recent',a.recent_capture,rc),('spread',a.spread_capture,sc)):
    path=folder/'snapshots'/f'{frame:06d}.pt';sha=next(z['read_sha256'] for z in receipt['snapshots'] if z['frame']==frame)
    if digest(path)!=sha:raise ValueError('Snapshot changed')
    data[name]=torch.load(path,map_location='cpu',weights_only=True);inp=data[name]['attention_inputs'];ptr=int(inp['num_obj_ptr_tokens']);end=len(inp['memory'])-ptr;idx=lookup.decode(inp['memory_pos'],ptr)
    if end!=16384 or ptr!=e['pointer_keys'] or len(inp['memory'])!=e['total_keys']:raise ValueError('Attention geometry differs')
    cursor=7168
    for h in e['history'][name]:
     if h['start']!=cursor or h['end']!=cursor+h['tokens'] or h['slot']!=min(frame-h['frame'],6)-1:raise ValueError('History segment differs')
     expected=(grid[idx[cursor:h['end']]]+offsets[h['slot']]).unsqueeze(1)
     if not torch.equal(inp['memory_pos'][cursor:h['end']],expected):raise ValueError('Original history position differs')
     cursor=h['end']
    if cursor!=end or e['history'][name][-1]['start']!=end-4096 or e['history'][name][-1]['frame']!=frame-1:raise ValueError('Newest memory differs')
   aa=data['recent']['attention_inputs'];bb=data['spread']['attention_inputs'];da=data['recent']['decoder_inputs'];db=data['spread']['decoder_inputs']
   query_same=equal(aa['curr'],bb['curr']) and equal(aa['curr_pos'],bb['curr_pos']);decoder_same=set(da)==set(db) and all(equal(da[k],db[k]) for k in da if k!='backbone_features')
   if query_same!=e['same_current_query'] or decoder_same!=e['same_decoder_image_inputs'] or not torch.equal(aa['memory'][:4096],bb['memory'][:4096]) or not torch.equal(aa['memory_pos'][:4096],bb['memory_pos'][:4096]):raise ValueError('Query or condition observation differs')
   for arm,sha in e['masks'].items():
    receiver='recent' if arm.startswith('recent_') else 'spread';donor='spread' if receiver=='recent' else 'recent';snap=data[receiver];args=dict(snap['attention_inputs']);source=data[donor]['attention_inputs'];ptr=int(args['num_obj_ptr_tokens']);end=len(args['memory'])-ptr
    if arm.endswith('_noop'):start=stop=0
    elif arm.endswith('_spatial'):start,stop=4096,end
    elif arm.endswith('_newest'):start,stop=end-4096,end
    elif arm.endswith('_pointers'):start,stop=end,len(args['memory'])
    else:raise ValueError('Unknown arm')
    for key in ('memory','memory_pos'):
     orig=args[key];other=source[key];args[key]=torch.cat((orig[:start],other[start:stop],orig[stop:]),dim=0)
     if hashlib.sha256(args[key].contiguous().numpy().tobytes()).hexdigest()!=e['input_sha256'][arm][key]:raise ValueError('Exchanged input tensor differs')
    attention=model.memory_attention(**args);rot=observations[-1]
    if rot!=e['rotary'][arm]:raise ValueError('Rotary geometry differs')
    head=clone_data(snap['decoder_inputs']);feat=attention[0];head['backbone_features']=feat.permute(1,2,0).view(1,feat.shape[2],64,64);pred=model._forward_sam_heads(**head)
    if abs(float(pred[6].item())-e['object_logits'][arm])>1e-12:raise ValueError('Native object logit differs')
    if arm.endswith('_noop') and (not equal(attention,snap['attention_outputs']) or not equal(pred,snap['decoder_outputs'])):raise ValueError('No-op model replay differs')
    state={'device':torch.device('cpu'),'video_height':snap['video_height'],'video_width':snap['video_width']};_,video=model._get_orig_video_res_output(state,pred[3]);mask=video[0,0].cpu().numpy()>0
    if mode=='candidate':mask=restore_mask(mask)
    for folder in (a.first,a.second):
     path=folder/f'{frame:06d}_{arm}.png'
     if digest(path)!=sha or not np.array_equal(np.asarray(Image.open(path))>0,mask):raise ValueError('Saved mask differs')
   for name,folder in (('recent',a.recent_capture),('spread',a.spread_capture)):
    if e['masks'][name+'_noop']!=digest(folder/f'{frame:06d}.png'):raise ValueError('No-op saved mask differs')
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'mode':mode,'frames':frames,'runner_sha256':x['runner_sha256'],'module_sha256':x['module_sha256'],'protocol_sha256':x['protocol_sha256'],'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('Exchanged tensors, query identity, source positions, object scores and masks independently reproduced.')

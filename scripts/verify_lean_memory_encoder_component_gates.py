#!/usr/bin/env python3
"""Independently recompute all encoder mixtures and their target-read outputs."""
import argparse,hashlib,json,sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK
p=argparse.ArgumentParser()
for k in ('first','second','recent-encoder','spread-encoder','recent-read','spread-read','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());x=read(a.first/'completion.json');y=read(a.second/'completion.json');mode=x['mode'];sources=[7,11] if mode=='gate' else [47,50];targets=[i+1 for i in sources]
if x!=y or not x['passed'] or mode not in ('gate','candidate') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_memory_encoder_components.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_components.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or x['checkpoint_sha256']!=digest(a.checkpoint) or [e['target_frame'] for e in x['events']]!=targets:raise ValueError('Replay provenance or repeats differ')
folders={'recent':(a.recent_encoder,a.recent_read),'spread':(a.spread_encoder,a.spread_read)};receipts={};read_receipts={}
for policy,(enc,reads) in folders.items():
 rf='recent_capture_receipt.json' if policy=='recent' else 'spread_capture_receipt.json';receipts[policy]=read(enc/'encoder_capture_receipt.json');read_receipts[policy]=read(reads/rf)
 if x[policy+'_encoder_sha256']!=digest(enc/'encoder_capture_receipt.json') or x[policy+'_read_sha256']!=digest(reads/rf) or not receipts[policy]['passed'] or not read_receipts[policy]['passed']:raise ValueError('Source capture changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);rope=[]
with torch.inference_mode(),indexed_spatial_rope(model,rope):
 for source,target,e in zip(sources,targets,x['events'],strict=True):
  data={};reads={};native={}
  for policy,(enc,folder) in folders.items():
   ep=enc/'encoder_snapshots'/f'{source:06d}.pt';rp=folder/'snapshots'/f'{target:06d}.pt';eh=next(z['read_sha256'] for z in receipts[policy]['snapshots'] if z['frame']==source);rh=next(z['read_sha256'] for z in read_receipts[policy]['snapshots'] if z['frame']==target)
   if digest(ep)!=eh or digest(rp)!=rh:raise ValueError('Encoder or read snapshot changed')
   data[policy]=torch.load(ep,map_location='cpu',weights_only=True);reads[policy]=torch.load(rp,map_location='cpu',weights_only=True);orig=reads[policy]['attention_inputs'];end=len(orig['memory'])-int(orig['num_obj_ptr_tokens']);value,pos=model._encode_new_memory(**data[policy]['inputs']);native[policy]=value.to(torch.bfloat16).flatten(2).permute(2,0,1).to(orig['memory'].dtype)
   if not equal((value,pos),data[policy]['outputs']) or not torch.equal(native[policy],orig['memory'][end-4096:end]) or not torch.equal(pos[-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0],orig['memory_pos'][end-4096:end]):raise ValueError('Native encoder or storage differs')
   if float(data[policy]['inputs']['object_score_logits'].item())!=e['source_object_logits'][policy]:raise ValueError('Source object logit differs')
  if e['same_source_vision']!=equal(data['recent']['inputs']['current_vision_feats'],data['spread']['inputs']['current_vision_feats']) or e['same_source_mask']!=equal(data['recent']['inputs']['pred_masks_high_res'],data['spread']['inputs']['pred_masks_high_res']):raise ValueError('Source identity summary differs')
  for recipient,donor in (('recent','spread'),('spread','recent')):
   base=data[recipient]['inputs'];other=data[donor]['inputs'];snap=reads[recipient];orig=snap['attention_inputs'];ptr=int(orig['num_obj_ptr_tokens']);end=len(orig['memory'])-ptr
   if base['feat_sizes']!=other['feat_sizes'] or base['is_mask_from_pts']!=other['is_mask_from_pts']:raise ValueError('Input geometry or prompt mode differs')
   for bits in product('rd',repeat=3):
    arm=''.join(bits);name=f'{recipient}_{arm}';v=other if bits[0]=='d' else base;m=other if bits[1]=='d' else base;o=other if bits[2]=='d' else base
    mixed={'current_vision_feats':v['current_vision_feats'],'feat_sizes':v['feat_sizes'],'pred_masks_high_res':m['pred_masks_high_res'],'object_score_logits':o['object_score_logits'],'is_mask_from_pts':base['is_mask_from_pts']};value,pos=model._encode_new_memory(**mixed);stored=value.to(torch.bfloat16).flatten(2).permute(2,0,1).to(orig['memory'].dtype)
    if hashlib.sha256(stored.contiguous().numpy().tobytes()).hexdigest()!=e['newest_feature_sha256'][name]:raise ValueError('Mixed encoded memory differs')
    if arm=='rrr' and not torch.equal(stored,native[recipient]) or arm=='ddd' and not torch.equal(stored,native[donor]):raise ValueError('Boundary memory differs')
    if not torch.equal(pos[-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0],orig['memory_pos'][end-4096:end]):raise ValueError('Mixed source position differs')
    args=dict(orig);args['memory']=torch.cat((orig['memory'][:end-4096],stored,orig['memory'][end:]),dim=0);att=model.memory_attention(**args)
    if rope[-1]!=e['rotary'][name]:raise ValueError('Rotary positions differ')
    head=clone_data(snap['decoder_inputs']);z=att[0];head['backbone_features']=z.permute(1,2,0).view(1,z.shape[2],64,64);pred=model._forward_sam_heads(**head)
    if abs(float(pred[6].item())-e['object_logits'][name])>1e-12:raise ValueError('Native object logit differs')
    if arm=='rrr' and (not equal(att,snap['attention_outputs']) or not equal(pred,snap['decoder_outputs'])):raise ValueError('No-op differs')
    state={'device':torch.device('cpu'),'video_height':snap['video_height'],'video_width':snap['video_width']};_,video=model._get_orig_video_res_output(state,pred[3]);mask=video[0,0].cpu().numpy()>0
    if mode=='candidate':mask=restore_mask(mask)
    for folder in (a.first,a.second):
     path=folder/f'{target:06d}_{name}.png'
     if digest(path)!=e['masks'][name] or not np.array_equal(np.asarray(Image.open(path))>0,mask):raise ValueError('Saved mask differs')
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'mode':mode,'targets':targets,'runner_sha256':x['runner_sha256'],'module_sha256':x['module_sha256'],'protocol_sha256':x['protocol_sha256'],'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('Every encoder mix, storage conversion, target output and saved mask independently reproduced.')

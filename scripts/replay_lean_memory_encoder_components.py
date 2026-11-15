#!/usr/bin/env python3
"""Replay factorial mixes of frozen encoder inputs into a fixed target read."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_encoder_components import ARMS,mix_encoder_inputs
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK,PATCHED

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
for k in ('recent-encoder','spread-encoder','recent-read','spread-read','capture-verification','exchange-reference','exchange-verification','model-verification','replay-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();sources=[7,11] if a.mode=='gate' else [47,50];targets=[i+1 for i in sources];entry='gate' if a.mode=='gate' else 'crop';cv=read(a.capture_verification/'verification.json');ev=read(a.exchange_verification/'verification.json');er=read(a.exchange_reference/'completion.json');mv=read(a.model_verification/'verification.json')
if not all(x['passed'] for x in (cv,ev,er,mv)) or cv['runner_sha256']!=digest(ROOT/'scripts/capture_lean_memory_encoder_inputs.py') or cv['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or cv['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or cv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_memory_encoder_capture_gates.py') or ev['mode']!=a.mode or ev['first_sha256']!=digest(a.exchange_reference/'completion.json') or ev['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_recent_spread_exchange_gates.py'):raise ValueError('Capture or prior exchange verification changed')
ident=mv['identity']
if mv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(digest(FORK/k)!=ident['patch_sha256'][k] for k in PATCHED):raise ValueError('Model worktree changed')
if a.mode=='candidate':
 rg=read(a.replay_verification/'verification.json')
 if not rg['passed'] or rg['runner_sha256']!=digest(Path(__file__)) or rg['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_components.py') or rg['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or rg['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_memory_encoder_component_gates.py'):raise ValueError('Factorial replay gates changed')
receipts={};read_receipts={};folders={'recent':(a.recent_encoder,a.recent_read),'spread':(a.spread_encoder,a.spread_read)}
for policy,(enc,reads) in folders.items():
 c=read(enc/'encoder_capture_receipt.json');rf='recent_capture_receipt.json' if policy=='recent' else 'spread_capture_receipt.json';rc=read(reads/rf)
 if not c['passed'] or not rc['passed'] or c['policy']!=policy or c['entry']!=entry or c['targets']!=sources or rc['entry']!=entry or c['runner_sha256']!=digest(ROOT/'scripts/capture_lean_memory_encoder_inputs.py') or c['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or c['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or digest(enc/'token_window_observation.json')!=c['output_observation_sha256']:raise ValueError('Frozen source capture changed')
 if a.mode=='gate' and cv['policies'][policy]['first_receipt_sha256']!=digest(enc/'encoder_capture_receipt.json'):raise ValueError('Gate source capture differs')
 receipts[policy]=c;read_receipts[policy]=rc
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
import sam2
if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Unexpected Lean worktree')
before=parameter_fingerprints(model);rope=[];rows=[];exchange_events={x['frame']:x for x in er['events']}
with torch.inference_mode(),indexed_spatial_rope(model,rope):
 for source,target in zip(sources,targets,strict=True):
  encoded={};target_read={};base_values={};position_blocks={};baseline=exchange_events[target]
  for policy,(enc,reads) in folders.items():
   ep=enc/'encoder_snapshots'/f'{source:06d}.pt';rp=reads/'snapshots'/f'{target:06d}.pt';eh=next(x['read_sha256'] for x in receipts[policy]['snapshots'] if x['frame']==source);rh=next(x['read_sha256'] for x in read_receipts[policy]['snapshots'] if x['frame']==target)
   if digest(ep)!=eh or digest(rp)!=rh:raise ValueError('Source or target snapshot changed')
   es=torch.load(ep,map_location='cpu',weights_only=True);ts=torch.load(rp,map_location='cpu',weights_only=True);inp=ts['attention_inputs'];ptr=int(inp['num_obj_ptr_tokens']);end=len(inp['memory'])-ptr;reencoded=model._encode_new_memory(**es['inputs']);stored=reencoded[0].to(torch.bfloat16).flatten(2).permute(2,0,1).to(inp['memory'].dtype);pos=reencoded[1][-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0]
   if es['frame']!=source or ts['frame']!=target or not equal(reencoded,es['outputs']) or end!=16384 or stored.shape!=(4096,1,model.mem_dim) or not torch.equal(stored,inp['memory'][end-4096:end]) or not torch.equal(pos,inp['memory_pos'][end-4096:end]):raise ValueError('Native source encoder or bfloat16 newest block differs')
   if not equal(model.memory_attention(**inp),ts['attention_outputs']) or not equal(model._forward_sam_heads(**ts['decoder_inputs']),ts['decoder_outputs']):raise ValueError('Target no-op attention or decoder differs')
   encoded[policy]=es;target_read[policy]=ts;base_values[policy]=stored;position_blocks[policy]=pos
  if not torch.equal(position_blocks['recent'],position_blocks['spread']):raise ValueError('Newest spatial positions differ')
  rec={};arm_masks={};arm_logits={};arm_features={};rotary={}
  for recipient,donor in (('recent','spread'),('spread','recent')):
   target_snap=target_read[recipient];original=target_snap['attention_inputs'];ptr=int(original['num_obj_ptr_tokens']);end=len(original['memory'])-ptr;rec[recipient]={}
   for arm in ARMS:
    mixed=mix_encoder_inputs(encoded[recipient]['inputs'],encoded[donor]['inputs'],arm);value,position=model._encode_new_memory(**mixed);stored=value.to(torch.bfloat16).flatten(2).permute(2,0,1).to(original['memory'].dtype);spatial=position[-1].flatten(2).permute(2,0,1)+model.maskmem_tpos_enc[0]
    if not torch.equal(spatial,position_blocks[recipient]):raise ValueError('Mixed encoder position changed')
    if arm=='rrr' and not torch.equal(stored,base_values[recipient]) or arm=='ddd' and not torch.equal(stored,base_values[donor]):raise ValueError('Factorial boundary does not reproduce real source')
    args=dict(original);args['memory']=torch.cat((original['memory'][:end-4096],stored,original['memory'][end:]),dim=0)
    if not torch.equal(args['memory'][:end-4096],original['memory'][:end-4096]) or not torch.equal(args['memory'][end:],original['memory'][end:]):raise ValueError('Unchanged memory modified')
    attention=model.memory_attention(**args);rotary[f'{recipient}_{arm}']=rope[-1];head=clone_data(target_snap['decoder_inputs']);feat=attention[0];head['backbone_features']=feat.permute(1,2,0).view(1,feat.shape[2],64,64);pred=model._forward_sam_heads(**head)
    if arm=='rrr' and (not equal(attention,target_snap['attention_outputs']) or not equal(pred,target_snap['decoder_outputs'])):raise ValueError('Factorial no-op differs')
    state={'device':torch.device('cpu'),'video_height':target_snap['video_height'],'video_width':target_snap['video_width']};_,video=model._get_orig_video_res_output(state,pred[3]);mask=video[0,0].cpu().numpy()>0
    if entry=='crop':mask=restore_mask(mask)
    dest=out/f'{target:06d}_{recipient}_{arm}.png';Image.fromarray(mask.astype(np.uint8)*255).save(dest);name=f'{recipient}_{arm}';arm_masks[name]=digest(dest);arm_logits[name]=float(pred[6].item());arm_features[name]=hashlib.sha256(stored.contiguous().numpy().tobytes()).hexdigest()
    previous=recipient+'_noop' if arm=='rrr' else (f'{recipient}_with_{donor}_newest' if arm=='ddd' else None)
    if previous and (arm_masks[name]!=baseline['masks'][previous] or abs(arm_logits[name]-baseline['object_logits'][previous])>1e-12):raise ValueError('Boundary control differs from independently verified prior exchange')
  rows.append({'source_frame':source,'target_frame':target,'source_object_logits':{k:float(encoded[k]['inputs']['object_score_logits'].item()) for k in folders},'same_source_vision':equal(encoded['recent']['inputs']['current_vision_feats'],encoded['spread']['inputs']['current_vision_feats']),'same_source_mask':equal(encoded['recent']['inputs']['pred_masks_high_res'],encoded['spread']['inputs']['pred_masks_high_res']),'masks':arm_masks,'object_logits':arm_logits,'newest_feature_sha256':arm_features,'rotary':rotary})
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'mode':a.mode,'events':rows,'recent_encoder_sha256':digest(a.recent_encoder/'encoder_capture_receipt.json'),'spread_encoder_sha256':digest(a.spread_encoder/'encoder_capture_receipt.json'),'recent_read_sha256':digest(a.recent_read/'recent_capture_receipt.json'),'spread_read_sha256':digest(a.spread_read/'spread_capture_receipt.json'),'exchange_reference_sha256':digest(a.exchange_reference/'completion.json'),'checkpoint_sha256':digest(a.checkpoint),'module_sha256':digest(ROOT/'modeling/lean_memory_encoder_components.py'),'protocol_sha256':digest(ROOT/'protocols/lean_newest_memory_component_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Frozen-read factorial encoder-input combinations; no labels or propagation.'}
(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'targets':targets}))

#!/usr/bin/env python3
"""Label-free fixed-read interventions on automatic conditioning memory."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_condition_region_ablation import prompt_region_scores,condition_ablation_indices,delete_condition_tokens
from modeling.lean_memory_capture import clone_data
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import VIEWPORT,restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED
from scripts.run_lean_offline_memory_ablation import equal

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('checkpoint','capture','pool-verification','merged','gate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();verified=read(a.pool_verification/'verification.json');ident=verified['identity']
 if not verified['passed'] or verified['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_token_pool_gates.py') or digest(a.checkpoint)!=ident['checkpoint_sha256'] or any(digest(FORK/x)!=ident['patch_sha256'][x] for x in PATCHED):raise ValueError('Verified author model changed')
 if a.mode=='gate':
  cap=read(a.capture/'region_capture_receipt.json');pool=read(a.capture/'pool_observation.json');seq='seq3';viewport=False
  if not cap['passed'] or not pool['passed'] or digest(a.capture/'pool_observation.json')!=cap['pool_receipt_sha256'] or pool['identity']!=ident:raise ValueError('Engineering capture changed')
  receipt_path=a.capture/'region_capture_receipt.json'
 else:
  cap=read(a.capture/'priority_capture_receipt.json');seq=cap['sequence'];viewport=cap['viewport'];receipt_path=a.capture/'priority_capture_receipt.json'
  if not cap['passed'] or seq not in ('seq8','seq11','seq20','seq23') or viewport!=(seq=='seq23'):raise ValueError('Priority capture changed')
  script='capture_lean_channel_priority_remaining.py' if seq in ('seq8','seq20') else 'capture_lean_region_priority_reads.py'
  if cap['priority_runner_sha256']!=digest(ROOT/'scripts'/script) or cap['capture_sha256']!=digest(ROOT/'modeling/lean_region_read_capture.py') or cap['selector_sha256']!=digest(ROOT/'modeling/lean_region_read_intervention.py') or digest(a.capture/'token_window_observation.json')!=cap['observation_sha256'] or digest(Path(cap['reference'])/'token_window_observation.json')!=cap['reference_observation_sha256']:raise ValueError('Capture provenance changed')
  g=read(a.gate_verification/'verification.json')
  if not g['passed'] or g['checkpoint_sha256']!=digest(a.checkpoint) or g['runner_sha256']!=digest(Path(__file__)) or g['module_sha256']!=digest(ROOT/'modeling/lean_condition_region_ablation.py') or g['protocol_sha256']!=digest(ROOT/'protocols/lean_condition_region_ablation_v1.json') or g['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_region_gate.py'):raise ValueError('Engineering gate changed')
 if viewport:
  prompt_path=a.capture/'prompt.png';prompt=read(a.capture/'prompt.json')
  if digest(prompt_path)!=digest(Path(cap['reference'])/'prompt.png'):raise ValueError('Inset-free prompt changed')
 else:
  folder=Path(read(a.merged/'locations.json')[seq])/seq;prompt_path=folder/'prompt_refined.png';prompt=read(folder/'prompt.json')
  if digest(prompt_path)!=prompt['refined_mask_sha256']:raise ValueError('Automatic prompt changed')
 mask=np.asarray(Image.open(prompt_path))>0
 if viewport:x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
 occupancy=prompt_region_scores(mask);out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK))
 model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Incorrect author worktree')
 before=parameter_fingerprints(model);grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1).contiguous();condition_pos=grid+model.maskmem_tpos_enc[model.num_maskmem-1].detach();events=[];rope=[]
 with torch.inference_mode(),indexed_spatial_rope(model,rope):
  for item in cap['snapshots']:
   i=item['frame'];path=a.capture/'snapshots'/f'{i:06d}.pt'
   if digest(path)!=item['read_sha256']:raise ValueError('Snapshot changed')
   data=torch.load(path,map_location='cpu',weights_only=True);args=data['attention_inputs'];ptr=int(args['num_obj_ptr_tokens']);total=len(args['memory'])
   if not torch.equal(args['memory_pos'][:4096],condition_pos):raise ValueError('First memory is not the original conditioning grid')
   original=model.memory_attention(**args)
   if not equal(original,data['attention_outputs']) or not equal(model._forward_sam_heads(**data['decoder_inputs']),data['decoder_outputs']):raise ValueError('Saved full read differs')
   usage=original[1].detach().double().cpu().numpy();den=usage.sum()
   if len(usage)!=total or not np.isfinite(usage).all() or (usage<0).any() or den<=0:raise ValueError('Invalid author attention usage')
   masks={};selected={};mass={};mask_roles={};keycounts={};rotary={}
   for method in ('noop','foreground','background','boundary','uniform','shuffled','attention_oracle'):
    call=clone_data(args)
    if method!='noop':
     indices=condition_ablation_indices(occupancy,usage[:4096],method);selected[method]=indices.tolist()
     call['memory'],call['memory_pos']=delete_condition_tokens(call['memory'],call['memory_pos'],ptr,indices)
     mass[method]=float(usage[indices].sum()/den);mask_roles[method]=float(occupancy[indices].mean())
     if len(indices)!=1024 or len(call['memory'])!=total-1024:raise ValueError('Matched count changed')
    keycounts[method]=len(call['memory']);features=model.memory_attention(**call)[0];rotary[method]=rope[-1]
    head=clone_data(data['decoder_inputs']);head['backbone_features']=features.permute(1,2,0).view(1,features.shape[2],64,64);result=model._forward_sam_heads(**head)
    if method=='noop' and not equal(result,data['decoder_outputs']):raise ValueError('No-op decoder differs')
    _,video=model._get_orig_video_res_output({'device':torch.device('cpu'),'video_height':data['video_height'],'video_width':data['video_width']},result[3]);image=video[0,0].cpu().numpy()>0
    if viewport:image=restore_mask(image)
    path=out/f'{i:06d}_{method}.png';Image.fromarray(image.astype(np.uint8)*255).save(path);masks[method]=digest(path)
    if method=='noop' and not np.array_equal(image,np.asarray(Image.open(a.capture/f'{i:06d}.png'))>0):raise ValueError('No-op mask differs')
   events.append({'frame':i,'masks':masks,'selected':selected,'removed_attention_mass':mass,'mean_prompt_occupancy':mask_roles,'key_counts':keycounts,'rotary':rotary})
 assert_parameter_integrity(model,before,out,'after')
 result={'passed':True,'mode':a.mode,'sequence':seq,'capture_sha256':digest(receipt_path),'prompt_sha256':digest(prompt_path),'checkpoint_sha256':digest(a.checkpoint),'module_sha256':digest(ROOT/'modeling/lean_condition_region_ablation.py'),'protocol_sha256':digest(ROOT/'protocols/lean_condition_region_ablation_v1.json'),'runner_sha256':digest(Path(__file__)),'events':events,'scope':'Label-free same-read conditional deletion, no propagation.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'sequence':seq,'frames':len(events)}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Swap nonconditioning spatial memories between two frozen trajectories."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_cross_trajectory_blocks import BLOCKS, swap_block
from modeling.lean_memory_capture import clone_data
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.seq23_viewport import restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_raw_two_control import FORK
from scripts.run_lean_token_window import PATCHED

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('checkpoint','raw-capture','fused-capture','raw-model-verification','raw-capture-verification','fused-capture-verification','swap-verification','block-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();raw=read(a.raw_capture/'raw_two_capture_receipt.json');fused=read(a.fused_capture/'fusion_capture_receipt.json');rv=read(a.raw_capture_verification/'verification.json');fv=read(a.fused_capture_verification/'verification.json');model_gate=read(a.raw_model_verification/'verification.json')
 expected=[8,12] if a.mode=='gate' else [8,20,35,48,51];entry='token' if a.mode=='gate' else 'crop'
 if not all(x['passed'] for x in (raw,fused,rv,fv,model_gate)) or raw['entry']!=entry or fused['entry']!=entry or raw['mode']!=a.mode or fused['mode']!=a.mode or raw['targets']!=expected or fused['targets']!=expected or raw['viewport']!=(entry=='crop') or fused['viewport']!=raw['viewport']:raise ValueError('Captured scope differs')
 if raw['runner_sha256']!=digest(ROOT/'scripts/capture_lean_raw_two_reads.py') or raw['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or raw['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_cross_trajectory_memory_v1.json') or rv['runner_sha256']!=raw['runner_sha256'] or rv['capture_sha256']!=raw['capture_sha256'] or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_raw_two_capture_gates.py'):raise ValueError('Raw capture gate changed')
 if fused['runner_sha256']!=digest(ROOT/'scripts/capture_lean_same_read_fusion.py') or fused['capture_sha256']!=digest(ROOT/'modeling/lean_same_read_fusion.py') or fused['protocol_sha256']!=digest(ROOT/'protocols/lean_same_read_fusion_v1.json') or fv['capture_sha256']!=fused['capture_sha256'] or fv['capture_runner_sha256']!=fused['runner_sha256'] or fv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_same_read_fusion_gates.py'):raise ValueError('Fused capture gate changed')
 identity=model_gate['identity']
 if model_gate['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or identity['checkpoint_sha256']!=digest(a.checkpoint) or any(digest(FORK/k)!=identity['patch_sha256'][k] for k in PATCHED):raise ValueError('Author checkpoint or fork changed')
 if a.mode=='candidate':
  bv=read(a.block_verification/'verification.json')
  if not bv['passed'] or bv['runner_sha256']!=digest(Path(__file__)) or bv['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_blocks.py') or bv['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_memory_block_swap_v1.json') or bv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_cross_trajectory_blocks.py'):raise ValueError('Block swap gates changed')
  sv=read(a.swap_verification/'verification.json')
  if not sv['passed'] or sv['runner_sha256']!=digest(ROOT/'scripts/replay_lean_cross_trajectory_memory.py') or sv['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_swap.py') or sv['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_cross_trajectory_memory_v1.json') or sv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_cross_trajectory_gates.py'):raise ValueError('Swap engineering gates changed')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Unexpected author worktree')
 before=parameter_fingerprints(model);events=[];rope=[];rs={x['frame']:x for x in raw['snapshots']};fs={x['frame']:x for x in fused['snapshots']}
 with torch.inference_mode(),indexed_spatial_rope(model,rope):
  for i in expected:
   data={}
   for name,folder,items in (('raw',a.raw_capture,rs),('fused',a.fused_capture,fs)):
    path=folder/'snapshots'/f'{i:06d}.pt'
    if digest(path)!=items[i]['read_sha256']:raise ValueError('Captured snapshot changed')
    data[name]=torch.load(path,map_location='cpu',weights_only=True)
    if data[name]['frame']!=i:raise ValueError('Frame identity differs')
    if not equal(model.memory_attention(**data[name]['attention_inputs']),data[name]['attention_outputs']) or not equal(model._forward_sam_heads(**data[name]['decoder_inputs']),data[name]['decoder_outputs']):raise ValueError('Frozen attention or decoder differs')
   da=data['raw']['decoder_inputs'];db=data['fused']['decoder_inputs']
   if set(da)!=set(db) or any(not equal(da[k],db[k]) for k in da if k!='backbone_features'):raise ValueError('Decoder image features or prompt inputs differ')
   counts={k:len(data[k]['attention_inputs']['memory']) for k in data};pointers={k:int(data[k]['attention_inputs']['num_obj_ptr_tokens']) for k in data}
   if counts['raw']!=counts['fused'] or pointers['raw']!=pointers['fused']:raise ValueError('Attention key counts not matched')
   source=torch.load(a.fused_capture/'fusion_sources'/f'{i:06d}.pt',map_location='cpu',weights_only=True)
   if source['start']!=7168 or len(source['fused'])!=1024 or not torch.equal(source['fused'],data['fused']['attention_inputs']['memory'][7168:8192]):raise ValueError('Captured fused skeleton does not occupy middle block')
   calls={'raw_noop':data['raw']['attention_inputs'],'fused_noop':data['fused']['attention_inputs']}
   for block in BLOCKS:
    for recipient,donor in (('raw','fused'),('fused','raw')):
     calls[f'{recipient}_with_{donor}_{block}']=swap_block(data[recipient]['attention_inputs'],data[donor]['attention_inputs'],block)
   masks={};rotary={}
   for arm,args in calls.items():
    recipient='raw' if arm.startswith('raw_') else 'fused';snapshot=data[recipient]
    if len(args['memory'])!=counts[recipient] or int(args['num_obj_ptr_tokens'])!=pointers[recipient]:raise ValueError('Swap key count changed')
    attention=model.memory_attention(**args);rotary[arm]=rope[-1];features=attention[0];head=clone_data(snapshot['decoder_inputs']);head['backbone_features']=features.permute(1,2,0).view(1,features.shape[2],64,64);result=model._forward_sam_heads(**head)
    if arm.endswith('_noop') and (not equal(attention,snapshot['attention_outputs']) or not equal(result,snapshot['decoder_outputs'])):raise ValueError('No-op replay differs')
    state={'device':torch.device('cpu'),'video_height':snapshot['video_height'],'video_width':snapshot['video_width']};_,video=model._get_orig_video_res_output(state,result[3]);mask=video[0,0].cpu().numpy()>0
    if raw['viewport']:mask=restore_mask(mask)
    path=out/f'{i:06d}_{arm}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path);masks[arm]=digest(path)
    folder=a.raw_capture if recipient=='raw' else a.fused_capture
    if arm.endswith('_noop') and not np.array_equal(mask,np.asarray(Image.open(folder/f'{i:06d}.png'))>0):raise ValueError('No-op mask differs')
   events.append({'frame':i,'total_keys':counts['raw'],'pointer_keys':pointers['raw'],'masks':masks,'rotary':rotary})
 assert_parameter_integrity(model,before,out,'after')
 result={'passed':True,'mode':a.mode,'events':events,'raw_capture_sha256':digest(a.raw_capture/'raw_two_capture_receipt.json'),'fused_capture_sha256':digest(a.fused_capture/'fusion_capture_receipt.json'),'checkpoint_sha256':digest(a.checkpoint),'swap_sha256':digest(ROOT/'modeling/lean_cross_trajectory_blocks.py'),'protocol_sha256':digest(ROOT/'protocols/lean_seq23_memory_block_swap_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Post-hoc matched-index block swaps across trajectories; no labels or feedback.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'frames':expected}))
if __name__=='__main__':main()

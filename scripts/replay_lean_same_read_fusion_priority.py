#!/usr/bin/env python3
"""Fresh-process, matched-key replay of captured author fusion and raw skeleton."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_memory_capture import clone_data
from modeling.lean_same_read_fusion import fused_skeleton,replace_fused_skeleton
from modeling.lean_region_read_intervention import locate_verified_span
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_fused_window_control import FORK
from scripts.run_lean_token_window import PATCHED
from scripts.run_lean_offline_memory_ablation import equal


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','capture','gate-verification','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();receipt=read(a.capture/'fusion_capture_receipt.json');verify=read(a.gate_verification/'verification.json')
    if not receipt['passed'] or not verify['passed'] or receipt['capture_sha256']!=digest(ROOT/'modeling/lean_same_read_fusion.py') or receipt['runner_sha256']!=digest(ROOT/'scripts/capture_lean_same_read_fusion_priority.py') or receipt['protocol_sha256']!=digest(ROOT/'protocols/lean_same_read_fusion_priority_v1.json'):
        raise ValueError('Capture implementation changed')
    if verify['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or verify['identity']['checkpoint_sha256']!=digest(a.checkpoint) or any(verify['identity']['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Fused author model changed')
    if digest(a.capture/'fused_control_receipt.json')!=receipt['output_receipt_sha256']:raise ValueError('Capture receipt changed')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong fused author package')
    before=parameter_fingerprints(model);offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();events=[]
    with torch.inference_mode(), indexed_spatial_rope(model,[]):
        for item in receipt['snapshots']:
            frame=item['frame'];snap=a.capture/'snapshots'/f'{frame:06d}.pt';src=a.capture/'fusion_sources'/f'{frame:06d}.pt'
            if digest(snap)!=item['read_sha256'] or digest(src)!=item['source_sha256']:raise ValueError('Snapshot changed')
            data=torch.load(snap,map_location='cpu',weights_only=True);source=torch.load(src,map_location='cpu',weights_only=True)
            args=data['attention_inputs'];ptr=int(args['num_obj_ptr_tokens'])
            if data['frame']!=frame or source['frame']!=frame or source['pointer_count']!=ptr or source['skeleton_frame']>=frame-1:raise ValueError('Source identity changed')
            if not torch.equal(source['fused'],fused_skeleton(source['raw'],source['history'],source['alpha'])):raise ValueError('Author fusion reconstruction changed')
            start=locate_verified_span(args['memory'],args['memory_pos'],ptr,source['fused'],source['spatial_pos'],offsets,[(0,4096)])
            if start!=source['start']:raise ValueError('Source position changed')
            if not equal(model.memory_attention(**args),data['attention_outputs']) or not equal(model._forward_sam_heads(**data['decoder_inputs']),data['decoder_outputs']):raise ValueError('Frozen baseline does not reproduce')
            masks={};differences={}
            for method in ('noop','fused_reconstructed','raw_skeleton'):
                call=clone_data(args)
                if method!='noop':call['memory']=replace_fused_skeleton(call['memory'],call['memory_pos'],ptr,source,'raw' if method=='raw_skeleton' else 'fused')
                if call['memory'].shape!=args['memory'].shape or not torch.equal(call['memory_pos'],args['memory_pos']):raise ValueError('Key count or positions changed')
                encoded=model.memory_attention(**call)[0];heads=clone_data(data['decoder_inputs']);heads['backbone_features']=encoded.permute(1,2,0).view(1,encoded.shape[2],64,64)
                result=model._forward_sam_heads(**heads)
                if method in ('noop','fused_reconstructed') and not equal(result,data['decoder_outputs']):raise ValueError('Fused no-op changed decoder')
                state={'device':torch.device('cpu'),'video_height':data['video_height'],'video_width':data['video_width']}
                _,video=model._get_orig_video_res_output(state,result[3]);image=(video[0,0].cpu().numpy()>0).astype(np.uint8)*255
                if receipt['viewport']:
                    from modeling.seq23_viewport import restore_mask
                    image=restore_mask(image>0).astype(np.uint8)*255
                path=out/f'{frame:06d}_{method}.png';Image.fromarray(image).save(path);masks[method]=digest(path)
                if method in ('noop','fused_reconstructed') and not np.array_equal(image>0,np.asarray(Image.open(a.capture/f'{frame:06d}.png'))>0):raise ValueError('Baseline mask changed')
                differences[method]=float((call['memory']-args['memory']).abs().max().item())
            events.append({'frame':frame,'source_frame':source['skeleton_frame'],'history_last_frame':source['history_last_frame'],'alpha':source['alpha'],'q':source['q'],
                'skeleton_tokens':len(source['raw']),'spatial_keys':int(args['memory'].shape[0]-ptr),'pointer_keys':ptr,
                'raw_fused_mean_abs':float((source['raw']-source['fused']).abs().mean().item()),'max_read_difference':differences,'masks':masks})
    assert_parameter_integrity(model,before,out,'after')
    result={'passed':True,'events':events,'capture_sha256':digest(a.capture/'fusion_capture_receipt.json'),
      'model_capture_sha256':digest(ROOT/'modeling/lean_same_read_fusion.py'),'runner_sha256':digest(Path(__file__)),
      'scope':'same-state local raw-skeleton replacement; no propagation or labels'}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'frames':[x['frame'] for x in events]}))
if __name__=='__main__':main()

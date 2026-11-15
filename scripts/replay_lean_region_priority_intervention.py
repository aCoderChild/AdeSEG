#!/usr/bin/env python3
"""Run matched deletions in a fresh process against verified captured reads."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.cross_frame_token_pool import TokenGroup
from modeling.lean_memory_capture import clone_data
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_region_read_intervention import remove_ranked_tokens,locate_verified_span
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED
from scripts.run_lean_offline_memory_ablation import equal
from datasets.video import load_manifest
from modeling.lean_rgb_contrast_control import rgb_reference,rgb_contrast
from modeling.seq23_viewport import crop_viewport,VIEWPORT


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','capture','pool-verification','merged','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();receipt=read(a.capture/'priority_capture_receipt.json');verified=read(a.pool_verification/'verification.json')
    if not receipt['passed'] or not verified['passed'] or receipt['sequence'] not in ('seq11','seq23') or receipt['viewport']!=(receipt['sequence']=='seq23'):
        raise ValueError('Invalid priority capture')
    for key,path in (('capture_sha256',ROOT/'modeling/lean_region_read_capture.py'),('selector_sha256',ROOT/'modeling/lean_region_read_intervention.py'),('priority_runner_sha256',ROOT/'scripts/capture_lean_region_priority_reads.py')):
        if digest(path)!=receipt[key]:raise ValueError('Capture code changed')
    if digest(a.capture/'token_window_observation.json')!=receipt['observation_sha256'] or digest(a.capture/'region_source_scores.json')!=receipt['source_scores_sha256'] or digest(Path(receipt['reference'])/'token_window_observation.json')!=receipt['reference_observation_sha256']:
        raise ValueError('Capture or reference changed')
    identity=verified['identity']
    if verified['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_token_pool_gates.py') or digest(a.checkpoint)!=identity['checkpoint_sha256'] or any(digest(FORK/x)!=identity['patch_sha256'][x] for x in PATCHED):raise ValueError('Pool gates, checkpoint or Lean worktree changed')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[receipt['sequence']]
    if receipt['viewport']:
        prompt_path=a.capture/'prompt.png';prompt=read(a.capture/'prompt.json')
    else:
        folder=Path(read(a.merged/'locations.json')[receipt['sequence']])/receipt['sequence']
        prompt_path=folder/'prompt_refined.png';prompt=read(folder/'prompt.json')
        if digest(prompt_path)!=prompt['refined_mask_sha256']:raise ValueError('Automatic prompt changed')
    initial_index=prompt['first_detection_index'];used_images={}
    def source_image(index):
        path=frames[index].image_path;used_images[str(index)]=digest(path)
        with Image.open(path) as im:image=np.asarray(im.convert('RGB'))
        return crop_viewport(image) if receipt['viewport'] else image
    initial=source_image(initial_index);mask=np.asarray(Image.open(prompt_path))>0
    if receipt['viewport']:
        x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
    if mask.shape!=initial.shape[:2]:raise ValueError('RGB prompt geometry differs')
    rgb_proto=rgb_reference(initial,mask)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Incorrect author worktree')
    before=parameter_fingerprints(model);events=[];offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu()
    with torch.inference_mode(), indexed_spatial_rope(model,[]):
        for item in receipt['snapshots']:
            frame=item['frame'];snap=a.capture/'snapshots'/f'{frame:06d}.pt';sp=a.capture/'source_snapshots'/f'{frame:06d}.pt'
            if digest(snap)!=item['read_sha256'] or digest(sp)!=item['sources_sha256']:raise ValueError('Snapshot changed')
            data=torch.load(snap,map_location='cpu',weights_only=True);source=torch.load(sp,map_location='cpu',weights_only=True)
            if data['frame']!=frame or source['frame']!=frame:raise ValueError('Snapshot frame differs')
            args=data['attention_inputs'];ptr=int(args['num_obj_ptr_tokens'])
            if ptr!=source['pointer_count']:raise ValueError('Pointer count differs')
            protected=[(0,4096)];segments=[];rgb_segments=[]
            for s in source['sources']:
                if type(s['frame']) is not int or s['frame']>=frame-1:
                    raise ValueError('Newest or invalid source cannot be intervened on')
                start=locate_verified_span(args['memory'],args['memory_pos'],ptr,s['features'],s['positions'],offsets,protected)
                if start!=s['start']:raise ValueError('Captured source span changed')
                protected.append((start,start+len(s['token_ids'])))
                group=TokenGroup(s['frame'],s['token_ids'].numpy(),s['usage'].numpy(),s['contrast'].numpy())
                segments.append({'start':start,'group':group,'boundary':s['boundary'].numpy()})
                rgb=rgb_contrast(source_image(s['frame']),rgb_proto)[s['token_ids']].numpy()
                rgb_group=TokenGroup(s['frame'],s['token_ids'].numpy(),s['usage'].numpy(),rgb)
                rgb_segments.append({'start':start,'group':rgb_group,'boundary':s['boundary'].numpy()})
            if sum(len(s['group'].token_ids) for s in segments)<1024:raise ValueError('Too few eligible historical tokens')
            attention=model.memory_attention(**args)
            if not equal(attention,data['attention_outputs']) or not equal(model._forward_sam_heads(**data['decoder_inputs']),data['decoder_outputs']):raise ValueError('Frozen baseline does not reproduce')
            masks={};deleted={}
            for method in ('noop','usage','contrast','distributed','three_region','foreground','shuffled','rgb_contrast'):
                call=clone_data(args)
                if method!='noop':
                    # Only captured verified sources may be removed; the prompt is explicitly protected.
                    chosen_segments=rgb_segments if method=='rgb_contrast' else segments
                    selection='contrast' if method=='rgb_contrast' else method
                    call['memory'],call['memory_pos'],removed=remove_ranked_tokens(call['memory'],call['memory_pos'],ptr,chosen_segments,[(0,4096)],selection,1024,seed=20261114)
                    deleted[method]=removed
                features=model.memory_attention(**call)[0]
                head_args=clone_data(data['decoder_inputs']);head_args['backbone_features']=features.permute(1,2,0).view(1,features.shape[2],64,64)
                result=model._forward_sam_heads(**head_args)
                if method=='noop' and not equal(result,data['decoder_outputs']):raise ValueError('No-op decoder differs')
                state={'device':torch.device('cpu'),'video_height':data['video_height'],'video_width':data['video_width']}
                _,video=model._get_orig_video_res_output(state,result[3]);image=(video[0,0].cpu().numpy()>0).astype(np.uint8)*255
                if receipt['viewport']:
                    from modeling.seq23_viewport import restore_mask
                    image=restore_mask(image>0).astype(np.uint8)*255
                path=out/f'{frame:06d}_{method}.png';Image.fromarray(image).save(path);masks[method]=digest(path)
                if method=='noop' and not np.array_equal(image>0,np.asarray(Image.open(a.capture/f'{frame:06d}.png'))>0):raise ValueError('No-op mask differs')
            events.append({'frame':frame,'masks':masks,'deleted':deleted,'source_frames':[s['frame'] for s in source['sources']]})
    assert_parameter_integrity(model,before,out,'after')
    result={'passed':True,'events':events,'capture_sha256':digest(a.capture/'priority_capture_receipt.json'),'runner_sha256':digest(Path(__file__)),
            'selector_sha256':digest(ROOT/'modeling/lean_region_read_intervention.py'),'rgb_module_sha256':digest(ROOT/'modeling/lean_rgb_contrast_control.py'),
            'rgb_prompt_sha256':digest(prompt_path),'rgb_source_image_sha256':used_images,'sequence':receipt['sequence'],'meaning':'matched local deletion only; no propagation or labels'}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'frames':[e['frame'] for e in events]}))
if __name__=='__main__':main()

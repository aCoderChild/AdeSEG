#!/usr/bin/env python3
"""Offline SAM2Long baseline using the frozen automatic Lean/SAMURAI prompt.

Runs author full-forward trajectory selection. Short engineering clips are
staged as complete clips: they are not prefix predictions for full videos,
because author SAM2Long chooses the final pathway after processing the clip.
No labels or new detector calls are used. Use a fresh process per sequence.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.seq23_viewport import restore_mask,stage_cropped_frames
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.run_samurai_automatic_priority import prepare_archived_prompt,model_state_hash,MANIFEST

CONFIG=ROOT/'external/SAM2Long/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
PATHWAY_CONFIG={'num_pathway':3,'iou_thre':0.3,'uncertainty':1}
DEPENDENCIES=('scripts/run_author_baseline.py','scripts/run_samurai_automatic_priority.py',
    'scripts/memory_token_io.py','datasets/video.py','modeling/samurai_auto_prompt.py',
    'modeling/seq23_viewport.py')


def read(path):return json.loads(path.read_text())


def dependency_hashes():return {name:digest(ROOT/name) for name in DEPENDENCIES}


def expected_identity(checkpoint,revision):
    return {'checkpoint_sha256':digest(checkpoint),'config_sha256':digest(CONFIG),
      'runner_sha256':digest(Path(__file__)),'manifest_sha256':digest(MANIFEST),
      'source_revision':revision,'torch':str(torch.__version__),'device':'mps',
      'platform':platform.platform(),'machine':platform.machine(),'seed':20261021,
      'dtype':'float32','fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
      'fast_math':os.environ.get('PYTORCH_MPS_FAST_MATH'),'pathways':PATHWAY_CONFIG,
      'dependencies':dependency_hashes(),'numpy':np.__version__,'python':sys.version}


def verify_gate(path,sequence,prompt,identity):
    gate=read(path)
    if (not gate.get('passed') or gate.get('sequence')!=sequence or gate.get('prompt')!=prompt
        or gate.get('identity')!=identity or gate.get('verifier_sha256')!=digest(ROOT/'scripts/verify_sam2long_automatic_gate.py')):
        raise ValueError('Matching offline short-clip engineering check required')
    return gate


def audit_pathway_state(state,start,end):
    """Check every stored author pathway before converting logits to masks."""
    if state.get('num_pathway')!=3:raise ValueError('Unexpected author pathway count')
    outputs=state['output_dict']['non_cond_frame_outputs']
    keys=('pred_masks','maskmem_features','obj_ptr','object_score_logits','ious')
    for frame in range(start+1,end+1):
        if frame not in outputs:raise ValueError('Missing author pathway output')
        entry=outputs[frame];expected=1 if frame==start+1 else 3
        for key in keys:
            value=entry.get(key)
            if not torch.is_tensor(value) or value.shape[-1]!=expected or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid/nonfinite {key} at frame {frame}')
        scores=entry.get('acc_score')
        if not isinstance(scores,list) or len(scores)!=3 or not np.isfinite(np.asarray(scores,dtype=np.float64)).all():
            raise ValueError(f'Invalid/nonfinite pathway scores at frame {frame}')
    return {'audited_post_prompt_frames':end-start,'pathways_after_initial_step':3,
            'checked_tensor_fields':list(keys)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=('seq11','seq20','seq23'),required=True)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    p.add_argument('--limit',type=int,help='Short, completely staged offline clip only')
    p.add_argument('--gate',type=Path)
    for key in ('checkpoint','preflight','cropped-reference','cropped-backup','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.limit is not None and a.limit<4:raise ValueError('A short engineering clip needs at least four post-prompt frames')
    if a.device=='mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0'):
        raise ValueError('MPS needs CPU fallback and fast math explicitly disabled')
    repo,revision=source_identity('sam2long')
    if digest(a.checkpoint)!=read(ROOT/'protocols/memory_token_regions.json')['model']['checkpoint_sha256']:
        raise ValueError('SAM2.1 B+ checkpoint differs')
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};all_frames=videos[a.sequence]
    start,prompt,viewport,provenance=prepare_archived_prompt(a.sequence,all_frames,a.preflight,a.cropped_reference,a.cropped_backup)
    end=min(len(all_frames)-1,start+a.limit-1) if a.limit is not None else len(all_frames)-1
    frames=all_frames[:end+1]
    identity=expected_identity(a.checkpoint,revision);identity['device']=a.device
    gate=None
    if a.limit is None:
        if a.device!='mps' or a.gate is None:raise ValueError('Full run needs a verified MPS gate')
        gate=verify_gate(a.gate,a.sequence,provenance,identity)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261021);torch.use_deterministic_algorithms(True)
    if a.device=='mps':torch.mps.manual_seed(20261021)
    sys.path.insert(0,str(repo));model=build_author_predictor(SimpleNamespace(method='sam2long',checkpoint=a.checkpoint,device=a.device),repo)
    import sam2
    if (not Path(sam2.__file__).resolve().is_relative_to(repo.resolve()) or model.fill_hole_area!=0
        or model.num_maskmem!=7 or model.memory_temporal_stride_for_eval!=1 or not model.iou_prediction_use_sigmoid):
        raise ValueError('Wrong author package or postprocessing')
    before=model_state_hash(model)
    if gate is not None and gate['state_sha256']!=before:raise ValueError('Loaded model state differs from gate')
    records=[]
    for i in range(start):
        with Image.open(frames[i].image_path) as image:size=image.size
        path=out/f'{i:06d}.png';Image.fromarray(np.zeros((size[1],size[0]),np.uint8)).save(path)
        records.append({'frame':i,'source_frame_index':frames[i].frame_index,'image_sha256':digest(frames[i].image_path),'mask_sha256':digest(path)})
    with tempfile.TemporaryDirectory(prefix='frames_',dir=out) as tmp:
        stage=Path(tmp)
        if a.sequence=='seq23':stage_cropped_frames(frames,stage)
        else:stage_frames(frames,stage)
        state=model.init_state(video_path=str(stage),offload_video_to_cpu=True,offload_state_to_cpu=False)
        if state['num_frames']!=len(frames):raise ValueError('Staged frame count differs')
        state.update(PATHWAY_CONFIG)
        try:
            with torch.inference_mode():
                model.add_new_mask(state,frame_idx=start,obj_id=1,mask=prompt)
                # Request a full forward pass of the *staged* clip. The author
                # partial/reverse return bounds are inconsistent; never use them.
                ids,masks=model.propagate_in_video(state,start_frame_idx=start,reverse=False)
                if ids!=[1] or len(masks)!=len(frames)-start:raise ValueError('Offline pathway coverage differs')
                pathway_audit=audit_pathway_state(state,start,end)
                for index,logits in enumerate(masks,start):
                    if logits.ndim!=4 or tuple(logits.shape[:2])!=(1,1) or not bool(torch.isfinite(logits).all()):
                        raise ValueError(f'Invalid/nonfinite returned logits at frame {index}')
                    mask=logits[0,0].detach().float().cpu().numpy()>0
                    if a.sequence=='seq23':mask=restore_mask(mask)
                    if mask.shape!=(viewport[3],viewport[2]):raise ValueError('Output geometry differs')
                    path=out/f'{index:06d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    records.append({'frame':index,'source_frame_index':frames[index].frame_index,
                      'image_sha256':digest(frames[index].image_path),'mask_sha256':digest(path)})
        finally:model.reset_state(state)
    if len(records)!=len(frames) or model_state_hash(model)!=before:raise ValueError('Output coverage or model state differs')
    receipt={'passed':True,'scope':'Exploratory offline SAM2Long with archived automatic mask; no label used in inference.',
      'noncausal':True,'sequence':a.sequence,'prompt_frame':start,'prompt':provenance,'end_frame':end,
      'full_video':a.limit is None,'identity':identity,'state_sha256':before,'gate_sha256':digest(a.gate) if a.gate else None,
      'deterministic':torch.are_deterministic_algorithms_enabled(),'pathway_audit':pathway_audit,'records':records}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'device':a.device,'prompt_frame':start,'frames':len(records),'offline':True}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verify the TCMN gate adapter on actual cropped seq23 SAM2 features.

Engineering only: all gate parameters are set to zero to produce uniform
weights, in explicit training mode. No trained checkpoint or video propagation
is claimed. Reads images and the archived automatic prompt only, not labels.
"""
import argparse,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.seq23_viewport import crop_viewport,VIEWPORT
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED


def read(p):return json.loads(p.read_text())
def tensor_hash(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','cropped-reference','prompt-gate-a','prompt-gate-b','gate-verification','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verification=read(a.gate_verification/'verification.json');base=read(a.cropped_reference/'completion.json')
    if not verification['passed'] or verification['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or digest(a.checkpoint)!=verification['identity']['checkpoint_sha256'] or any(digest(FORK/x)!=verification['identity']['patch_sha256'][x] for x in PATCHED):raise ValueError('Lean checkpoint or worktree differs')
    if not base['passed'] or digest(a.cropped_reference/'prompt.json')!=base['prompt_sha256'] or digest(a.cropped_reference/'frames.json')!=base['frames_sha256'] or digest(a.cropped_reference/'setup.json')!=base['setup_sha256']:raise ValueError('Cropped reference changed')
    prompt_gates=[]
    for path in (a.prompt_gate_a,a.prompt_gate_b):
        gate=read(path/'gate.json')
        if gate['setup']!=read(a.cropped_reference/'setup.json') or digest(path/'prompt.png')!=gate['files']['prompt.png'] or digest(a.cropped_reference/'prompt.png')!=gate['files']['prompt.png']:
            raise ValueError('Archived automatic prompt mask changed')
        prompt_gates.append(gate)
    if prompt_gates[0]!=prompt_gates[1]:raise ValueError('Prompt gates differ')
    frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];rows=read(a.cropped_reference/'frames.json')
    prompt=read(a.cropped_reference/'prompt.json');first=prompt['first_detection_index']
    if first!=0:raise ValueError('Unexpected archived prompt frame')
    targets=(first,8);out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong Lean worktree')
    before=parameter_fingerprints(model)
    with tempfile.TemporaryDirectory(prefix='channel_interface_',dir=out) as temp:
        for j,index in enumerate(targets):
            if digest(frames[index].image_path)!=rows[index]['image_sha256']:raise ValueError('Input frame changed')
            with Image.open(frames[index].image_path) as im:img=crop_viewport(np.asarray(im.convert('RGB')))
            Image.fromarray(img).save(Path(temp)/f'{j:06d}.jpg',format='PNG')
        state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        try:
            with torch.inference_mode():
                _,initial,_,_,_=model._get_image_feature(state,0,batch_size=1)
                _,source,_,_,_=model._get_image_feature(state,1,batch_size=1)
                initial=initial['backbone_fpn'][0][0].float();source=source['backbone_fpn'][0][0].float()
                if initial.shape!=(32,256,256) or source.shape!=initial.shape:raise ValueError('Early feature interface changed')
                mask=np.asarray(Image.open(a.cropped_reference/'prompt.png'))>0
                x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
                fg,bg=reference_prototypes(initial,mask)
                # The untrained gate is exercised only in an explicit
                # gradient-enabled engineering context, never inference mode.
                with torch.inference_mode(False), torch.enable_grad():
                    module=PromptContrastChannelAttention(initialize_for_training=True)
                    for parameter in module.gate.parameters():torch.nn.init.zeros_(parameter)
                    score,weights=module(source.clone(),fg.clone(),bg.clone())
                    score=score.detach();weights=weights.detach()
                old=blend_anchor_with_early_contrast(torch.zeros(4096),source,fg,bg,(64,64),1.)
                error=float((score-old).abs().max().item())
                if not torch.equal(weights,torch.full((32,),.5)) or error>1e-6:raise ValueError('Uniform gate did not reproduce baseline')
                result={'passed':True,'scope':'zero-weight engineering gate, no learned inference or video propagation','sequence':'seq23',
                  'frames':list(targets),'feature_shape':list(source.shape),'feature_sha256':tensor_hash(source),'foreground_sha256':tensor_hash(fg),'background_sha256':tensor_hash(bg),
                  'weights_sha256':tensor_hash(weights),'score_sha256':tensor_hash(score),'max_abs_baseline_error':error,
                  'checkpoint_sha256':digest(a.checkpoint),'prompt_sha256':digest(a.cropped_reference/'prompt.png'),
                  'input_sha256':[digest(frames[i].image_path) for i in targets], 'author_cbam_sha256':author_sha256(),
                  'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'runner_sha256':digest(Path(__file__))}
        finally:model.reset_state(state)
    assert_parameter_integrity(model,before,out,'after')
    (out/'interface_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

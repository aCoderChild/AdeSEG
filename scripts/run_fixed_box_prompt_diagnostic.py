#!/usr/bin/env python3
"""Decode saved initial boxes on fixed images; never run detector or read GT."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.seq23_viewport import crop_viewport,restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity
from scripts.run_prompt_feature_refinement import REPO,REVISION,predict_mask

PROTOCOL=ROOT/'protocols/lean_fixed_box_oracle_diagnostic_v1.json'
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','target'),required=True)
    for k in ('checkpoint','checkpoint-verification','preflight','crop-reference','gate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();manifest=read(a.preflight/'manifest.json');pre=read(a.preflight/'completion.json')
    if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_training_pilot_v1.json'):
        raise ValueError('Frame selection preflight changed')
    rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True)
    if rev!=REVISION or dirty:raise ValueError('Reference SAM2 source changed')
    crop=read(a.crop_reference/'completion.json')
    if not crop['passed'] or crop['prompt_sha256']!=digest(a.crop_reference/'prompt.json') or crop['setup_sha256']!=digest(a.crop_reference/'setup.json'):
        raise ValueError('Viewport prompt reference changed')
    verified=read(a.checkpoint_verification/'verification.json')
    if not verified['passed'] or verified['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or verified['identity']['checkpoint_sha256']!=digest(a.checkpoint):
        raise ValueError('Checkpoint does not match verified Lean baseline')
    identity={'checkpoint_sha256':digest(a.checkpoint),'sam2_revision':rev,'preflight_sha256':digest(a.preflight/'manifest.json'),
      'crop_prompt_sha256':digest(a.crop_reference/'prompt.json'),'protocol_sha256':digest(PROTOCOL),'runner_sha256':digest(Path(__file__)),
      'helper_sha256':digest(ROOT/'scripts/run_prompt_feature_refinement.py'),'viewport_sha256':digest(ROOT/'modeling/seq23_viewport.py'),'checkpoint_verification_sha256':digest(a.checkpoint_verification/'verification.json'),'torch':torch.__version__,'device':'cpu_fp32'}
    if a.mode=='target':
        v=read(a.gate_verification/'verification.json')
        if not v['passed'] or v['identity']!=identity or v['verifier_sha256']!=digest(ROOT/'scripts/verify_fixed_box_prompt_gates.py'):raise ValueError('Independent image predictor gates changed')
    names=['seq3'] if a.mode=='gate' else ['seq8','seq11','seq20','seq23'];cases=[]
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    for seq in names:
        meta=manifest['sequences'][seq];source=Path(meta['source'])/seq;prompt=read(source/'prompt.json')
        if digest(source/'prompt.json')!=meta['prompt_sha256']:raise ValueError('Archived prompt changed')
        if seq=='seq23':box=read(a.crop_reference/'prompt.json')['box_viewport_xyxy']
        else:box=prompt['box']
        if box is None:raise ValueError('Initial box unavailable')
        indices=meta['targets'][:3] if a.mode=='gate' else meta['targets']
        for i in indices:
            position=meta['targets'].index(i)
            if digest(videos[seq][i].image_path)!=meta['target_image_sha256'][position]:raise ValueError('Source image changed')
            cases.append({'sequence':seq,'frame':i,'image_sha256':meta['target_image_sha256'][position],'box':box,'viewport':seq=='seq23'})
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261114)
    sys.path.insert(0,str(REPO));import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(REPO.resolve()):raise ValueError('Incorrect SAM2 reference package')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False)
    before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);events=[]
    with torch.inference_mode():
        for c in cases:
            with Image.open(videos[c['sequence']][c['frame']].image_path) as im:image=np.array(im.convert('RGB'),copy=True)
            if c['viewport']:image=crop_viewport(image)
            x0,y0,x1,y1=c['box'];h,w=image.shape[:2]
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Box outside image viewport')
            predictor.set_image(image);boxes={'fixed_initial':c['box']}
            if a.mode=='gate':
                h,w=image.shape[:2];boxes['synthetic_control']=[w*.1,h*.1,w*.9,h*.9]
            hashes={}
            for name,box in boxes.items():
                mask=predict_mask(predictor,image.shape[:2],box)
                if c['viewport']:mask=restore_mask(mask)
                path=out/f"{c['sequence']}_{c['frame']:06d}_{name}.png";Image.fromarray(mask.astype(np.uint8)*255).save(path);hashes[name]=digest(path)
            events.append({**c,'masks':hashes})
    assert_parameter_integrity(model,before,out,'after')
    result={'passed':True,'mode':a.mode,'identity':identity,'events':events,'scope':'Fixed-frame saved-box inference; no detector, labels, or propagation.'}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'cases':len(events)}))
if __name__=='__main__':main()

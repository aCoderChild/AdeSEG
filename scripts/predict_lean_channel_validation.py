#!/usr/bin/env python3
"""Produce frozen validation feature scores without opening label masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast
from modeling.lean_rgb_contrast_control import rgb_reference,rgb_contrast
from modeling.lean_channel_feature_capture import sha
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('dataset','dataset-audit','training','training-audit','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();d=read(a.dataset/'dataset.json');da=read(a.dataset_audit/'verification.json');tr=read(a.training/'training_receipt.json');ta=read(a.training_audit/'verification.json');pre=read(a.preflight/'manifest.json')
    if not da['passed'] or da['dataset_sha256']!=digest(a.dataset/'dataset.json') or da['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_dataset.py') or not ta['passed'] or ta['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_repeat.py') or ta['first_receipt_sha256']!=digest(a.training/'training_receipt.json') or ta['dataset_sha256']!=digest(a.dataset/'dataset.json') or not tr['passed']:
        raise ValueError('Training or dataset audit changed')
    if d['preflight_sha256']!=digest(a.preflight/'manifest.json') or d['groups']['validation']!=['seq9','seq10'] or set(d['groups']['protected_priority'])!={'seq8','seq11','seq20','seq23'}:
        raise ValueError('Validation identity changed')
    samples=[x for x in d['samples'] if x['group']=='validation'];expected={'seq9','seq10'}
    if len(samples)!=10 or {x['sequence'] for x in samples}!=expected:raise ValueError('Unexpected validation scope')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    tensor_data={};prototypes={};rgb_prototypes={};initial_image_hashes={};prompt_hashes={}
    for item in samples:
        seq=item['sequence'];i=item['frame'];path=Path(item['feature_path'])
        if digest(path)!=item['feature_file_sha256'] or digest(videos[seq][i].image_path)!=item['image_sha256']:raise ValueError('Validation feature or input changed')
        sample=torch.load(path,map_location='cpu',weights_only=True)
        if sample['frame']!=i or sample['feature'].shape!=(32,256,256):raise ValueError('Validation feature geometry changed')
        pair=(sample['foreground'],sample['background']);prior=prototypes.setdefault(seq,pair)
        if not torch.equal(prior[0],pair[0]) or not torch.equal(prior[1],pair[1]):raise ValueError('Automatic reference changed within sequence')
        tensor_data[(seq,i)]=sample
    for seq in sorted(expected):
        meta=pre['sequences'][seq];source=Path(meta['source'])/seq;mask_path=source/'prompt_refined.png';first=meta['first_prompt']
        if digest(mask_path)!=meta['prompt_mask_sha256']:raise ValueError('Automatic prompt changed')
        image_path=videos[seq][first].image_path;initial_image_hashes[seq]=digest(image_path);prompt_hashes[seq]=digest(mask_path)
        with Image.open(image_path) as image:pixels=np.asarray(image.convert('RGB'))
        mask=np.asarray(Image.open(mask_path))>0
        if mask.shape!=pixels.shape[:2]:raise ValueError('RGB prototype geometry changed')
        rgb_prototypes[seq]=rgb_reference(pixels,mask)
    foreign={'seq9':prototypes['seq10'],'seq10':prototypes['seq9']};models={}
    for arm,info in tr['arms'].items():
        path=a.training/f'{arm}.pt'
        if digest(path)!=info['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        models[arm]=PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=d['backbone_sha256'],expected_feature_pipeline_sha256=d['feature_pipeline_sha256']).eval().requires_grad_(False)
    parameter_before={arm:parameter_hash(model) for arm,model in models.items()}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);events=[]
    with torch.inference_mode():
        for item in sorted(samples,key=lambda x:(x['sequence'],x['frame'])):
            seq=item['sequence'];frame=item['frame'];sample=tensor_data[(seq,frame)];x=sample['feature'];fg,bg=sample['foreground'],sample['background'];scores={}
            for arm,model in models.items():
                if arm=='prompt_conditioned':gfg,gbg=fg,bg
                elif arm=='descriptor_independent':gfg,gbg=torch.zeros_like(fg),torch.zeros_like(bg)
                elif arm=='mismatched_prompt':gfg,gbg=foreign[seq]
                else:raise ValueError('Unknown arm')
                score,weights=model(x,fg,bg,gate_foreground=gfg,gate_background=gbg);scores[arm]={'score':score.cpu(),'weights':weights.cpu()}
            scores['unweighted']={'score':blend_anchor_with_early_contrast(torch.zeros(4096),x,fg,bg,(64,64),1.).cpu()}
            with Image.open(videos[seq][frame].image_path) as image:rgb=np.asarray(image.convert('RGB'))
            scores['rgb']={'score':rgb_contrast(rgb,rgb_prototypes[seq]).cpu()}
            event={'sequence':seq,'frame':frame,'image_sha256':item['image_sha256'],'scores':{}}
            for arm,value in scores.items():
                if value['score'].shape!=(4096,) or not torch.isfinite(value['score']).all():raise ValueError('Invalid validation score')
                path=out/f'{seq}_{frame:06d}_{arm}.pt';torch.save(value,path)
                event['scores'][arm]={'file_sha256':digest(path),'score_sha256':sha(value['score'])}
            events.append(event)
    parameter_after={arm:parameter_hash(model) for arm,model in models.items()}
    if parameter_before!=parameter_after:raise ValueError('Channel gate weights changed during validation prediction')
    result={'passed':True,'scope':'Frozen validation feature scores; no validation labels opened.','events':events,'initial_image_sha256':initial_image_hashes,'prompt_sha256':prompt_hashes,'parameter_sha256':parameter_before,
      'dataset_sha256':digest(a.dataset/'dataset.json'),'training_receipt_sha256':digest(a.training/'training_receipt.json'),'training_audit_sha256':digest(a.training_audit/'verification.json'),
      'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'runner_sha256':digest(Path(__file__))}
    (out/'prediction_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'samples':len(events)}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Freeze source-prediction prototypes and score temporal background evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from evaluation.weighted_rank_auc import weighted_rank_auc
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity

PROTOCOL = ROOT/'protocols/seq11_temporal_background_v1.json'
MANIFEST = ROOT/'data/polypgen_sequence.jsonl'
def read(p): return json.loads(p.read_text())
def area(mask, shape):
    x = torch.from_numpy(np.asarray(mask, np.float32).copy())[None,None]
    return F.interpolate(x, size=shape, mode='area')[0,0].numpy()

def capture(a):
    protocol=read(PROTOCOL); pairs=protocol['pairs']; frames={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}['seq11']
    done=read(a.reference/'completion.json'); records=read(a.reference/'records.json'); setup=read(a.reference/'setup.json')
    if not done['passed'] or digest(a.reference/'records.json')!=done['records_sha256']: raise ValueError('Reference invalid')
    signature=setup['identity']['cpu_identity']['source_signature']
    if digest(a.checkpoint)!=signature['checkpoint_sha256']: raise ValueError('Checkpoint differs')
    source=Path(read(a.merged/'locations.json')['seq11']); source_record=read(source/'records.json')[0]
    prompt=read(source/'seq11/prompt.json')
    if digest(source/'seq11/prompt.json')!=source_record['prompt_sha256'] or prompt['first_detection_index']!=2:
        raise ValueError('Reference prompt differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);repo,revision=source_identity('lean')
    if revision!=signature['revision']: raise ValueError('Lean revision differs')
    model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
    from sam2.utils.misc import _load_img_as_tensor
    before=parameter_fingerprints(model); cache={}
    def feature(index):
        if index not in cache:
            x,_,_=_load_img_as_tensor(str(frames[index].image_path),model.image_size)
            x=x.float(); x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            with torch.inference_mode():
                f=model.forward_image(x[None])['backbone_fpn'][0][0].detach().float().cpu().contiguous()
            if index==2 and hashlib.sha256(f.unsqueeze(0).numpy().tobytes()).hexdigest()!=prompt['early_feature_sha256']:
                raise ValueError('Initial early feature does not match reference replay')
            cache[index]=f
        return cache[index]
    def rgb(index,shape):
        image=np.asarray(Image.open(frames[index].image_path).convert('RGB'),dtype=np.float32).copy()/255.
        return F.interpolate(torch.from_numpy(image).permute(2,0,1)[None],size=shape,mode='area')[0]
    outputs=[]
    for src,dst in pairs:
        mask_path=a.reference/f'{src:06d}.png'
        if digest(mask_path)!=records['mask_sha256'][src]:raise ValueError('Reference source mask changed')
        mask=np.asarray(Image.open(mask_path))>0
        fsrc=feature(src); ftgt=feature(dst)
        event={'source':src,'target':dst,'source_mask_sha256':digest(mask_path),'source_image_sha256':digest(frames[src].image_path),
               'target_image_sha256':digest(frames[dst].image_path)}
        try:
            fg,bg=reference_prototypes(fsrc,mask)
        except ValueError:
            event['status']='prototype_unavailable';outputs.append(event);continue
        f=F.normalize(ftgt,dim=0); sf=torch.einsum('chw,c->hw',f,fg);sb=torch.einsum('chw,c->hw',f,bg)
        weights=torch.from_numpy(area(mask,fsrc.shape[-2:]));blocked=F.max_pool2d((weights>0).float()[None,None],5,stride=1,padding=2)[0,0];back=(blocked==0).float()
        rgb_src=rgb(src,fsrc.shape[-2:]); rgb_dst=rgb(dst,fsrc.shape[-2:]); cf=(rgb_src*weights).sum((1,2))/weights.sum();cb=(rgb_src*back).sum((1,2))/back.sum()
        color_fg=-torch.linalg.vector_norm(rgb_dst-cf[:,None,None],dim=0);color_bg=-torch.linalg.vector_norm(rgb_dst-cb[:,None,None],dim=0)
        arrays={'foreground':sf.numpy(),'background_negative':(-sb).numpy(),'contrast':(sf-sb).numpy(),
                'rgb_foreground':color_fg.numpy(),'rgb_contrast':(color_fg-color_bg).numpy()}
        path=out/f'{src:06d}_{dst:06d}.npz';np.savez_compressed(path,**arrays)
        event.update(status='available',map_sha256=digest(path));outputs.append(event)
    assert_parameter_integrity(model,before,out,'after')
    result={'passed':True,'sequence':'seq11','protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__)),
            'prototype_sha256':digest(ROOT/'modeling/prompt_feature_refinement.py'),'reference_records_sha256':done['records_sha256'],
            'checkpoint_sha256':digest(a.checkpoint),'outputs':outputs}
    (out/'capture.json').write_text(json.dumps(result,indent=2)+'\n');print('Captured',len(outputs),'fixed pairs')

def audit(a):
    receipt=read(a.capture/'capture.json'); done=read(a.reference/'completion.json');records=read(a.reference/'records.json')
    if not receipt['passed'] or receipt['protocol_sha256']!=digest(PROTOCOL) or receipt['source_sha256']!=digest(Path(__file__)) or receipt['reference_records_sha256']!=done['records_sha256'] or digest(a.reference/'records.json')!=done['records_sha256']:
        raise ValueError('Capture or reference differs')
    frames={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}['seq11'];rows=[]
    for item in receipt['outputs']:
        src,dst=item['source'],item['target'];row=dict(item)
        if digest(frames[src].image_path)!=item['source_image_sha256'] or digest(frames[dst].image_path)!=item['target_image_sha256']:raise ValueError('Input changed')
        srcmask=np.asarray(Image.open(a.reference/f'{src:06d}.png'))>0;target_path=a.reference/f'{dst:06d}.png'
        if digest(a.reference/f'{src:06d}.png')!=item['source_mask_sha256'] or digest(target_path)!=records['mask_sha256'][dst]:raise ValueError('Predictions changed')
        srcgt=load_label_mask(frames[src].mask_path,[1])==1;gt=load_label_mask(frames[dst].mask_path,[1])==1;pred=np.asarray(Image.open(target_path))>0
        row.update(source_label_sha256=digest(frames[src].mask_path),target_label_sha256=digest(frames[dst].mask_path),source_gt_pixels=int(srcgt.sum()),
                   source_gt_outside_prediction=int((srcgt & ~srcmask).sum()),target_present=bool(gt.any()),target_predicted_pixels=int(pred.sum()))
        if item['status']=='available':
            path=a.capture/f'{src:06d}_{dst:06d}.npz'
            if digest(path)!=item['map_sha256']:raise ValueError('Score map changed')
            arrays=np.load(path,allow_pickle=False);row['scores']={}
            for name in arrays.files:
                z=arrays[name];shape=z.shape
                if not np.isfinite(z).all():raise ValueError('Invalid scores')
                row['scores'][name]={'global':weighted_rank_auc(z,area(gt,shape),area(~gt,shape)),
                  'within_prediction':weighted_rank_auc(z,area(gt&pred,shape),area((~gt)&pred,shape))}
        rows.append(row)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps({'capture_sha256':digest(a.capture/'capture.json'),'rows':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('capture','audit'))
    for k in ('reference','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('checkpoint','merged','capture'):p.add_argument('--'+k,type=Path)
    a=p.parse_args()
    if a.mode=='capture':
        if not a.checkpoint or not a.merged:raise ValueError('Capture needs checkpoint and merged reference')
        capture(a)
    else:
        if not a.capture:raise ValueError('Audit needs capture')
        audit(a)
if __name__=='__main__':main()

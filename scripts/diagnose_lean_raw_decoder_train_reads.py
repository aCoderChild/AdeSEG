#!/usr/bin/env python3
"""Replay frozen training reads and inspect raw masks before SAM2 suppression.

Offline diagnostic only: uses the original predicted-IoU mask selection, saves
both raw and native masks before opening labels, and never changes propagation,
object scores, pointer decisions, prompts or memory. Only training samples are
eligible. Labeled-empty frames are retrospective development examples, not a
representative estimate of negative-video performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask, save_binary_mask
from modeling.lean_memory_capture import clone_data
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture,to_video
from scripts.train_region_conditioned_hardcase_v1 import load_manifest

FORK=ROOT/'external/Lean-SAM2-token-window'


def select_raw_mask(masks, ious, multimask):
    """Match SAM2 selection using predicted IoU, never label information."""
    if masks.ndim!=4 or ious.ndim!=2 or masks.shape[:2]!=ious.shape or type(multimask) is not bool:
        raise ValueError('Unexpected decoder mask or predicted-IoU geometry')
    if not torch.isfinite(masks).all() or not torch.isfinite(ious).all():
        raise ValueError('Nonfinite raw decoder output')
    if multimask:
        best=ious.argmax(dim=-1)
        return masks[torch.arange(len(best),device=masks.device),best].unsqueeze(1),best
    if masks.shape[1]!=1:raise ValueError('Single-mask decoder returned multiple masks')
    return masks,torch.zeros(masks.shape[0],dtype=torch.long,device=masks.device)


def state_hash(model):
    """Fingerprint parameters and buffers before and after read-only replay."""
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-preflight','capture-reference','checkpoint','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();manifest,mhash=load_manifest(a.data_preflight)
    reference=json.loads((a.capture_reference/'channel_feature_receipt.json').read_text())
    rawid=reference['pipeline']['raw_identity']
    if (not reference.get('passed') or reference.get('sequence')!='seq4'
            or reference['feature_pipeline_sha256']!=manifest['feature_pipeline_sha256']
            or reference['checkpoint_sha256']!=manifest['checkpoint_sha256']
            or digest(a.checkpoint)!=manifest['checkpoint_sha256']):
        raise ValueError('Capture identity or model differs')
    if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=rawid['revision']:
        raise ValueError('Fork revision changed')
    if any(digest(FORK/path)!=expected for path,expected in rawid['patch_sha256'].items()):
        raise ValueError('Fork patch changed')
    for field,path in (('policy_sha256','modeling/token_memory_window.py'),('runtime_sha256','modeling/lean_token_window_storage.py')):
        if digest(ROOT/path)!=rawid[field]:raise ValueError('Token-window pipeline changed')
    _,rev=source_identity('lean')
    if rev!=rawid['revision']:raise ValueError('Author revision differs')
    rows=sorted((x for x in manifest['samples'] if x['split']=='train'),key=lambda x:(x['sequence'],x['frame']))
    if len(rows)!=19 or any(x['sequence'] not in ('seq2','seq3','seq4') for x in rows):
        raise ValueError('Training-only coverage differs')
    for row in rows:
        if digest(row['image'])!=row['image_sha256']:raise ValueError('Image changed')
        read_capture(row)
    torch.set_num_threads(2);sys.path.insert(0,str(FORK))
    model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong model worktree')
    if not model.pred_obj_scores:raise ValueError('Unexpected native presence policy')
    before=state_hash(model)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    predictions=out/'frozen_predictions';predictions.mkdir();frozen=[]
    with torch.inference_mode():
        for item in rows:
            snap=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
            if snap['frame']!=item['frame']:raise ValueError('Snapshot frame differs')
            captured=[]
            handle=model.sam_mask_decoder.register_forward_hook(lambda _m,_i,value:captured.append(clone_data(value)))
            try:post=model._forward_sam_heads(**snap['decoder_inputs'])
            finally:handle.remove()
            if len(captured)!=1 or not equal(post,snap['decoder_outputs']):raise ValueError('Native post-suppression decoder replay differs')
            raw_masks,raw_ious,_,raw_score=captured[0]
            if not torch.equal(raw_score,post[6]) or not torch.equal(raw_ious,post[2]):raise ValueError('Raw score or IoU differs')
            chosen,index=select_raw_mask(raw_masks.float(),raw_ious,snap['decoder_inputs']['multimask_output'])
            native=post[3].float();score=float(raw_score[0,0])
            if score>0 and not torch.equal(chosen,native):raise ValueError('Positive-score raw selection differs from native')
            if score<=0 and not torch.all(native==-1024):raise ValueError('Negative-score native suppression differs')
            shape=(int(snap['video_height']),int(snap['video_width']))
            raw_mask=(to_video(chosen,shape)>0).cpu().numpy()[0,0]
            native_mask=(to_video(native,shape)>0).cpu().numpy()[0,0]
            # Compare the native read directly to the prior frozen trajectory,
            # not just to another tensor produced by this replay.
            source_dir=Path(item['snapshot']).parent.parent
            receipt_path=source_dir/'extra_capture_receipt.json'
            if not receipt_path.exists():receipt_path=source_dir/'channel_feature_receipt.json'
            receipt=json.loads(receipt_path.read_text());record_path=source_dir/'records.json'
            if not receipt.get('passed') or receipt.get('records_sha256')!=digest(record_path):
                raise ValueError('Archived trajectory receipt changed')
            record=json.loads(record_path.read_text());prior=source_dir/f"{item['frame']:06d}.png"
            if digest(prior)!=record['mask_sha256'][item['frame']] or not np.array_equal(native_mask,np.asarray(Image.open(prior).convert('L'))>0):
                raise ValueError('Native read differs from archived trajectory mask')
            stem=f"{item['sequence']}_{item['frame']:06d}";paths={}
            for name,mask in (('raw',raw_mask),('native',native_mask)):
                path=predictions/f'{stem}_{name}.png';save_binary_mask(mask,path)
                paths[name]=str(path);paths[name+'_sha256']=digest(path)
            tensor_file=predictions/f'{stem}_raw_decoder.pt'
            torch.save({'raw_multimasks':raw_masks,'raw_ious':raw_ious,'object_score_logits':raw_score,
                        'selected_index':index,'native_selected':native,'raw_selected':chosen},tensor_file)
            frozen.append({'sequence':item['sequence'],'frame':item['frame'],'shape':list(shape),
                           'score':score,'selected_index':int(index[0]),'raw_candidates':raw_masks.shape[1],
                           'raw_min':float(chosen.min()),'raw_max':float(chosen.max()),
                           'tensor':str(tensor_file),'tensor_sha256':digest(tensor_file),**paths})
    if state_hash(model)!=before:raise ValueError('Model parameters or buffers changed')
    (out/'frozen_predictions.json').write_text(json.dumps({'model_state_sha256':before,'rows':frozen},indent=2)+'\n')
    # Ground truth is opened only after all 19 model outputs are frozen.
    scored=[]
    for item,entry in zip(rows,frozen,strict=True):
        if digest(item['label'])!=item['label_sha256']:raise ValueError('Training label changed')
        target=load_label_mask(Path(item['label']),[1])==1
        if list(target.shape)!=entry['shape']:raise ValueError('Label geometry differs')
        row={k:entry[k] for k in ('sequence','frame','score','selected_index','raw_candidates','raw_min','raw_max')}
        row.update({'present':bool(target.any()),'target_pixels':int(target.sum())})
        for name in ('native','raw'):
            pred=np.asarray(Image.open(entry[name]).convert('L'))>0;den=int(pred.sum()+target.sum())
            row[name]={'dice':2*int((pred&target).sum())/den if den else 1.,'predicted_pixels':int(pred.sum()),
                       'overlap_pixels':int((pred&target).sum())}
        scored.append(row)
    present=[r for r in scored if r['present']];empty=[r for r in scored if not r['present']]
    summary={mode:{'present_frames':len(present),'mean_present_dice':float(np.mean([r[mode]['dice'] for r in present])),
                   'empty_frames':len(empty),'empty_false_alarms':sum(r[mode]['predicted_pixels']>0 for r in empty),
                   'mean_empty_predicted_pixels':float(np.mean([r[mode]['predicted_pixels'] for r in empty]))} for mode in ('native','raw')}
    result={'passed':True,'scope':'Training-only frozen-read analysis. Raw masks are offline diagnostics, not predictions from a deployed no-object bypass.',
            'manifest_sha256':mhash,'checkpoint_sha256':digest(a.checkpoint),'capture_reference_sha256':digest(a.capture_reference/'channel_feature_receipt.json'),
            'runner_sha256':digest(Path(__file__)),'native_decoder_source_sha256':digest(FORK/'sam2/modeling/sam2_base.py'),
            'model_state_sha256':before,'frozen_predictions_sha256':digest(out/'frozen_predictions.json'),'summary':summary,'rows':scored}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':summary,'suppressed':[x for x in scored if x['score']<=0]},indent=2),flush=True)


if __name__=='__main__':main()

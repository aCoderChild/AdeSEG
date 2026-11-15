#!/usr/bin/env python3
"""Independently rescore and replay four frozen hard-case development arms.

This verifies internal reproducibility, not patient independence, statistical
power, or full-video accuracy. All inputs are existing artifacts; no training
or inference-time ground-truth information is introduced.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest,validate_output

ARMS=('region','global','shuffled','mask_only')


def read(path):return json.loads(Path(path).read_text())


def decode_label(path):
    """Decode PolypGen RGB/JPEG masks independently at project threshold 127."""
    with Image.open(path) as im:
        if im.format!='JPEG':raise ValueError('Unexpected non-JPEG label')
        gray=np.asarray(im.convert('L'))
    if ((gray>64)&(gray<192)).mean()>0.05:
        raise ValueError('Label does not resemble a binary JPEG mask')
    return gray>127


def compute(pred,truth):
    if pred.shape!=truth.shape:raise ValueError('Prediction/label shape differs')
    area=int(pred.sum()); den=area+int(truth.sum())
    return {'dice':2*int((pred&truth).sum())/den if den else 1.0,'predicted_pixels':area}


def summarize(rows,name):
    val=[x for x in rows if x['split']=='validation']
    p=[x[name]['dice'] for x in val if x['present']]
    e=[x[name]['predicted_pixels'] for x in val if not x['present']]
    if not p or not e:raise ValueError('Validation presence strata absent')
    return {'mean_present_dice':float(np.mean(p)),'present_frames':len(p),
            'empty_false_alarms':sum(x>0 for x in e),'empty_frames':len(e),
            'mean_empty_predicted_pixels':float(np.mean(e))}


def run(manifest_dir,inputs,output):
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
        raise RuntimeError('MPS replay requires available MPS and disabled CPU fallback')
    manifest_file=manifest_dir/'manifest.json'; manifest=read(manifest_file); done=read(manifest_dir/'completion.json')
    manifest_sha=digest(manifest_file)
    if not manifest['passed'] or not done['passed'] or done['manifest_sha256']!=manifest_sha:
        raise ValueError('Manifest receipt changed')
    if manifest['source_sha256']!=digest(ROOT/'scripts/preflight_refiner_hardcase_v3.py') or digest(manifest['previous_manifest_path'])!=manifest['previous_manifest_sha256'] or digest(manifest['extra_receipt_path'])!=manifest['extra_receipt_sha256']:
        raise ValueError('Manifest provenance changed')
    samples=manifest['samples']; bykey={(x['sequence'],x['frame']):x for x in samples}
    if len(samples)!=29 or len(bykey)!=29 or sum(x['split']=='train' for x in samples)!=19 or sum(x['split']=='validation' for x in samples)!=10:
        raise ValueError('Study split or duplicate identities differ')
    for item in samples:
        for key in ('image','label','feature','snapshot'):
            if digest(item[key])!=item[key+'_sha256']:raise ValueError('Source '+key+' changed')
    expected_keys={(x['sequence'],x['frame']) for x in samples if x['split']=='validation'}|{('seq4',21),('seq4',24)}
    if len(expected_keys)!=12:raise ValueError('Unexpected audit frame set')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    common=None;reference_order=None;baseline_hashes=None;report={};source_hashes={}
    for arm in ARMS:
        folder=inputs[arm];receipt=read(folder/'completion.json'); frozen_path=folder/'frozen_predictions.json'; frozen=read(frozen_path)
        checkpoint=folder/'checkpoint.pt'; chk_sha=digest(checkpoint)
        if not receipt.get('passed') or receipt['arm']!=arm or receipt['manifest_sha256']!=manifest_sha or receipt['checkpoint_sha256']!=chk_sha or receipt['frozen_predictions_sha256']!=digest(frozen_path) or frozen['checkpoint_sha256']!=chk_sha:
            raise ValueError('Frozen receipt or checkpoint differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        values={'format':'foreground_conditioned_refiner_v1','research_only':True,'training_steps':190,
            'backbone_sha256':manifest['checkpoint_sha256'],'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
            'train_manifest_sha256':manifest_sha,'author_sha256':author_sha256(),
            'conditioned_source_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
            'base_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
            'runner_source_sha256':digest(ROOT/'scripts/train_region_conditioned_hardcase_v1.py'),
            'study_arm':arm,'gate_mode':'global' if arm=='mask_only' else arm,
            'contrast_head':True,'arm':'mask_only' if arm=='mask_only' else 'multilevel',
            'epochs':10,'seed':20261114,'device':'mps'}
        if any(payload.get(k)!=v for k,v in values.items()) or receipt['seed']!=values['seed'] or receipt['epochs']!=10:
            raise ValueError('Checkpoint or experiment metadata differs')
        if len(receipt['history'])!=10 or any(len(x['samples'])!=19 for x in receipt['history']):raise ValueError('Training coverage differs')
        ordering=[[(x['sequence'],x['frame']) for x in epoch['samples']] for epoch in receipt['history']]
        if any(len(set(epoch))!=19 or set(epoch)!={(x['sequence'],x['frame']) for x in samples if x['split']=='train'} for epoch in ordering):raise ValueError('Training epoch samples differ')
        shared={k:payload[k] for k in ('backbone_sha256','feature_pipeline_sha256','train_manifest_sha256',
            'author_sha256','conditioned_source_sha256','base_source_sha256','runner_source_sha256',
            'epochs','seed','device','torch','parity_sha256','parameter_count')}
        if common is None:common=shared;reference_order=ordering
        elif shared!=common or ordering!=reference_order:raise ValueError('Arms differ in shared source/settings or sample order')
        if sum(x.numel() for x in payload['state_dict'].values())!=payload['parameter_count'] or any(not torch.isfinite(x).all() for x in payload['state_dict'].values()):
            raise ValueError('Parameter count or values differ')
        model=ForegroundConditionedRefiner(gate_mode=values['gate_mode'])
        model.load_trained_state(payload,backbone_sha256=manifest['checkpoint_sha256'],feature_pipeline_sha256=manifest['feature_pipeline_sha256'],train_manifest_sha256=manifest_sha)
        model.to('mps').eval()
        if len(frozen['rows'])!=12 or {(x['sequence'],x['frame']) for x in frozen['rows']}!=expected_keys or len(receipt['rows'])!=12:
            raise ValueError('Frozen or scored frame coverage differs')
        scored={(x['sequence'],x['frame']):x for x in receipt['rows']}
        if len(scored)!=12:raise ValueError('Duplicate scored frames')
        actual=[];current_baselines={};replay=[]
        for entry in frozen['rows']:
            key=(entry['sequence'],entry['frame']);item=bykey[key];truth=decode_label(item['label'])
            if entry['split']!=item['split'] or tuple(entry['shape'])!=truth.shape:raise ValueError('Split or target geometry differs')
            features=torch.load(item['feature'],map_location='cpu',weights_only=True)
            snapshot=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
            if features['frame']!=item['frame'] or snapshot['frame']!=item['frame']:raise ValueError('Feature frame differs')
            early=features['feature'].float().unsqueeze(0);deep=snapshot['decoder_inputs']['backbone_features'].float();logits=snapshot['decoder_outputs'][3].float()
            if arm=='mask_only':early=torch.zeros_like(early);deep=torch.zeros_like(deep)
            with torch.inference_mode():
                refined=model(early.to('mps'),deep.to('mps'),logits.to('mps'))
                pred=(F.interpolate(refined,size=truth.shape,mode='bilinear',align_corners=False)>0)[0,0].cpu().numpy()
                base=(F.interpolate(logits.to('mps'),size=truth.shape,mode='bilinear',align_corners=False)>0)[0,0].cpu().numpy()
            row={'sequence':key[0],'frame':key[1],'split':item['split'],'present':bool(truth.any())}
            diffs={}
            for name,repeat in (('baseline',base),('refined',pred)):
                path=Path(entry[name])
                if path.parent.resolve()!=(folder/'frozen_predictions').resolve() or digest(path)!=entry[name+'_sha256']:
                    raise ValueError('Frozen mask path or checksum differs')
                with Image.open(path) as img:mask=np.asarray(img.convert('L'))>0
                row[name]=compute(mask,truth);diffs[name]=int((mask!=repeat).sum())
            if row!=scored[key]:raise ValueError('Independently recomputed row differs')
            actual.append(row);replay.append({'sequence':key[0],'frame':key[1],**diffs});current_baselines[key]=entry['baseline_sha256']
        if baseline_hashes is None:baseline_hashes=current_baselines
        elif baseline_hashes!=current_baselines:raise ValueError('Baseline masks differ across arms')
        summary={n:summarize(actual,n) for n in ('baseline','refined')}
        if summary!=receipt['summary']:raise ValueError('Recomputed aggregate differs')
        report[arm]={'summary':summary,'rows':actual,'mps_replay_pixel_differences':replay,
                     'max_replay_pixel_difference':max(max(x['baseline'],x['refined']) for x in replay),
                     'checkpoint_sha256':chk_sha,'receipt_sha256':digest(folder/'completion.json'),
                     'frozen_predictions_sha256':digest(frozen_path)}
        source_hashes[arm]=digest(folder/'completion.json')
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':'Recomputed frozen fixed-read development metrics and same-device replay; not independent generalization or video efficacy.',
            'manifest_sha256':manifest_sha,'manifest_samples':29,'training_frames':19,'validation_frames':10,'sampled_train_debug_frames':[21,24],
            'common_checkpoint_metadata':common,'training_epoch_order_matches':True,'source_receipt_sha256':source_hashes,
            'runner_sha256':digest(Path(__file__)),'arms':report,
            'limitations':['One training seed; validation sequences previously inspected and reused; patient independence unknown.',
              'Training debug frames are label-selected and are not held-out evaluation.',
              'Same-device replay of twelve frozen reads does not establish propagation reliability or performance.',
              'Matching current source and sample order does not independently verify initial model state, optimizer state or all historical runtime flags.']}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'frames_per_arm':12,'max_replay_pixel_difference':{k:v['max_replay_pixel_difference'] for k,v in report.items()},'summaries':{k:v['summary']['refined'] for k,v in report.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for arm in ARMS:p.add_argument('--'+arm.replace('_','-'),type=Path,required=True)
    args=p.parse_args();run(args.manifest,{arm:getattr(args,arm) for arm in ARMS},args.output)

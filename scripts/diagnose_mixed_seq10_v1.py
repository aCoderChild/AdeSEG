#!/usr/bin/env python3
"""Replay frozen seq10 refiner outputs before label-only development analysis.

This is a retrospective diagnostic on previously inspected validation reads;
labels enter only after all requested frozen masks have been regenerated and
verified, and never enter a model. No video inference policy is changed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
from scripts.train_region_mixed_loss_v1 import environment_identity,inputs,mixed_manifest,model_for,to_video
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture

CASES=(14,16,19,22,24)
METHODS={'region_bce':'region','region_robust_bce':'region','global_bce':'global','mask_only_bce':'mask_only'}


def q(values):
    if values.size==0:return None
    return [float(x) for x in np.quantile(values,[.05,.5,.95])]


def metrics(prediction,truth):
    tp=int((prediction & truth).sum());fp=int((prediction & ~truth).sum());fn=int((~prediction & truth).sum())
    return {'predicted_pixels':int(prediction.sum()),'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-preflight','scores','run-root','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);environment=environment_identity()
    manifest,manifest_sha=mixed_manifest(a.data_preflight)
    report=json.loads((a.scores/'scores.json').read_text());done=json.loads((a.scores/'completion.json').read_text())
    if (not done['passed'] or done['scores_sha256']!=digest(a.scores/'scores.json')
            or not report['passed'] or report['manifest_sha256']!=manifest_sha
            or report['runner_source_sha256']!=digest(ROOT/'scripts/train_region_mixed_loss_v1.py')
            or report['scorer_source_sha256']!=digest(ROOT/'scripts/score_region_mixed_loss_v1.py')):
        raise ValueError('Independent frozen-study score receipt differs')
    cases=[next(x for x in manifest['samples'] if x['split']=='validation' and x['sequence']=='seq10' and x['frame']==i) for i in CASES]
    cached={x['frame']:{'sample':x,'methods':{}} for x in cases};run_hashes={};training={}
    feature_statistics=[]
    for item in manifest['samples']:
        early,deep,low,_=read_capture(item)
        snap=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
        feature_statistics.append({'sequence':item['sequence'],'frame':item['frame'],'split':item['split'],
            'capture_device':item['capture_device'],'early_mean_abs':float(early.abs().mean()),
            'early_std':float(early.std(unbiased=False)),'deep_mean_abs':float(deep.abs().mean()),
            'deep_std':float(deep.std(unbiased=False)),'original_positive_logit_fraction':float((low>0).float().mean()),
            'original_logit_median':float(low.median()),'object_score':float(snap['decoder_outputs'][6].reshape(-1)[0])})
    # Reproduce every frozen prediction before opening any validation label.
    for method,arm in METHODS.items():
        folder=a.run_root/method;complete=json.loads((folder/'completion.json').read_text())
        checkpoint=folder/'checkpoint.pt';frozen_path=folder/'frozen_predictions.json'
        if (not complete['passed'] or complete['manifest_sha256']!=manifest_sha
                or complete['checkpoint_sha256']!=digest(checkpoint) or complete['frozen_predictions_sha256']!=digest(frozen_path)
                or report['runs'][method]['checkpoint_sha256']!=digest(checkpoint)):
            raise ValueError('Run or checkpoint differs: '+method)
        frozen=json.loads(frozen_path.read_text());rows={(x['sequence'],x['frame'],x['split']):x for x in frozen['rows']}
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);model=model_for(arm)
        model.load_trained_state(payload,backbone_sha256=manifest['checkpoint_sha256'],
            feature_pipeline_sha256=manifest['feature_pipeline_sha256'],train_manifest_sha256=manifest_sha)
        model.to('mps').eval();run_hashes[method]=digest(checkpoint)
        history=complete['history'];training[method]={}
        for index in (0,4,9):
            epoch=history[index];selected={}
            for seq,frame in (('seq4',24),('seq6',90),('seq13',62)):
                entry=next(x for x in epoch['samples'] if x['sequence']==seq and x['frame']==frame)
                selected[f'{seq}/{frame}']={'loss':entry['loss'],'maximum_gradient':entry['maximum_gradient']}
            training[method][str(index+1)]={'mean_loss':epoch['mean_loss'],'selected_training_samples':selected}
        with torch.inference_mode():
            for item in cases:
                frame=item['frame'];key=('seq10',frame,'validation');row=rows[key]
                early,deep,logits,shape=inputs(item,arm,'mps')
                base=to_video(logits,shape).float().cpu().numpy()[0,0]
                refined=to_video(model(early,deep,logits),shape).float().cpu().numpy()[0,0]
                if list(shape)!=row['shape']:raise ValueError('Video geometry changed')
                for name,pred in (('baseline',base>0),('refined',refined>0)):
                    path=Path(row[name])
                    if not path.resolve().is_relative_to(folder.resolve()) or digest(path)!=row[name+'_sha256']:
                        raise ValueError('Frozen mask changed')
                    if not np.array_equal(np.asarray(Image.open(path).convert('L'))>0,pred):
                        raise ValueError('Frozen mask does not reproduce')
                snapshot=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
                cached[frame]['object_score']=float(snapshot['decoder_outputs'][6].reshape(-1)[0])
                cached[frame]['methods'][method]={'baseline_logits':base,'refined_logits':refined}
        del model
    results=[]
    for frame in CASES:
        entry=cached[frame];item=entry['sample']
        if digest(item['label'])!=item['label_sha256']:raise ValueError('Development label changed')
        truth=load_label_mask(Path(item['label']),[1])==1
        result={'sequence':'seq10','frame':frame,'present':bool(truth.any()),'resolution':list(truth.shape),
                'target_pixels':int(truth.sum()),'target_fraction':float(truth.mean()),
                'object_score':entry['object_score'],'methods':{}}
        for method,vals in entry['methods'].items():
            base,refined=vals['baseline_logits'],vals['refined_logits'];delta=refined-base
            if base.shape!=truth.shape or refined.shape!=truth.shape:raise ValueError('Label geometry differs')
            gt=truth;bg=~truth;predicted=base>0
            result['methods'][method]={'baseline':metrics(predicted,truth),'refined':metrics(refined>0,truth),
                'baseline_logit_quantiles_gt_foreground':q(base[gt]),'residual_quantiles_gt_foreground':q(delta[gt]),
                'refined_logit_quantiles_gt_foreground':q(refined[gt]),'residual_quantiles_gt_background':q(delta[bg]),
                'residual_quantiles_baseline_predicted_foreground':q(delta[predicted])}
        results.append(result)
    for entry in feature_statistics:
        item=next(x for x in manifest['samples'] if x['sequence']==entry['sequence'] and x['frame']==entry['frame'])
        if digest(item['label'])!=item['label_sha256']:raise ValueError('Development label changed')
        truth=load_label_mask(Path(item['label']),[1])==1
        entry['label_foreground_fraction']=float(truth.mean())
    by_sequence={}
    for seq in sorted({x['sequence'] for x in feature_statistics}):
        group=[x for x in feature_statistics if x['sequence']==seq]
        by_sequence[seq]={'split':group[0]['split'],'capture_device':group[0]['capture_device'],
            'reads':len(group),'label_foreground_fraction_range':[min(x['label_foreground_fraction'] for x in group),max(x['label_foreground_fraction'] for x in group)],
            'mean_early_mean_abs':float(np.mean([x['early_mean_abs'] for x in group])),
            'mean_deep_mean_abs':float(np.mean([x['deep_mean_abs'] for x in group])),
            'mean_original_positive_logit_fraction':float(np.mean([x['original_positive_logit_fraction'] for x in group]))}
    data={'passed':True,'scope':'Previously inspected seq10 fixed-read development diagnosis; no labels in model inference and no independent causality or generalization claim.',
          'source_sha256':digest(Path(__file__)),'manifest_sha256':manifest_sha,'scores_sha256':digest(a.scores/'scores.json'),
          'checkpoint_sha256':run_hashes,'environment':environment,'quantiles':[.05,.5,.95],
          'training_loss_trajectory':training,'seq10_rows':results,'feature_statistics':feature_statistics,
          'feature_statistics_by_sequence':by_sequence,
          'feature_statistics_limit':'Descriptive only: sequence, capture device and content are confounded; these cannot attribute cause.'}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    path=out/'diagnostic.json';path.write_text(json.dumps(data,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'diagnostic_sha256':digest(path)},indent=2)+'\n')
    print(json.dumps([{'frame':r['frame'],'present':r['present'],
        'baseline_recall':r['methods']['region_bce']['baseline']['recall'],
        'region_bce_recall':r['methods']['region_bce']['refined']['recall'],
        'region_robust_bce_recall':r['methods']['region_robust_bce']['refined']['recall']} for r in results],indent=2))


if __name__=='__main__':main()

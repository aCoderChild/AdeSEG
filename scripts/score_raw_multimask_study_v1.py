#!/usr/bin/env python3
"""Independently replay and score all six frozen mixed-device pilot arms.

All six completions, settings, model weights and frozen predictions are
validated before any validation label is read. This is development data,
not a patient-independent or locked final evaluation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.train_raw_multimask_study_v1 import (ARMS,LOSSES,EPOCHS,SEED,LOSS_FORMULAS,dependency_hashes,environment_identity,inputs,load_target,model_for,to_video,validate_plan,load_manifests,entry_for)


def validate_training_order(history, manifest):
    if len(history)!=EPOCHS or [x['epoch'] for x in history]!=list(range(1,EPOCHS+1)):
        raise ValueError('Training history incomplete')
    expected={(x['sequence'],x['frame'],x['capture_device']) for x in manifest['samples'] if x['split']=='train'}
    order=[]
    for entry in history:
        samples=entry['samples']
        keys=[(x['sequence'],x['frame'],x['capture_device']) for x in samples]
        if len(keys)!=39 or len(set(keys))!=39 or set(keys)!=expected:
            raise ValueError('Training order coverage differs')
        if any(not np.isfinite(x['loss']) or not np.isfinite(x['maximum_gradient']) for x in samples):
            raise ValueError('Nonfinite saved training diagnostic')
        order.append(keys)
    return order


def check_run(folder,arm,kind,manifest,manifest_sha,raw_sha,plan_path,parity_path):
    complete_path=folder/'completion.json';frozen_path=folder/'frozen_predictions.json';checkpoint_path=folder/'checkpoint.pt'
    complete=json.loads(complete_path.read_text());frozen=json.loads(frozen_path.read_text())
    if (not complete.get('passed') or complete.get('validation_labels_opened') is not False
            or complete.get('arm')!=arm or complete.get('loss_kind')!=kind
            or complete.get('seed')!=SEED or complete.get('epochs')!=EPOCHS
            or complete.get('manifest_sha256')!=manifest_sha or complete.get('raw_manifest_sha256')!=raw_sha or complete.get('plan_sha256')!=digest(plan_path)
            or complete.get('checkpoint_sha256')!=digest(checkpoint_path)
            or complete.get('frozen_predictions_sha256')!=digest(frozen_path)
            or frozen.get('checkpoint_sha256')!=digest(checkpoint_path)):
        raise ValueError(f'{arm}/{kind}: completion or artifacts differ')
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    expected={'format':'raw_multimask_refiner_v1','research_only':True,'raw_mode':arm,
              'gate_mode':'region','contrast_head':True,'loss_kind':kind,
              'loss_formula':LOSS_FORMULAS[kind],'seed':SEED,'epochs':EPOCHS,'training_steps':39*EPOCHS,
              'backbone_sha256':manifest['checkpoint_sha256'],'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
              'train_manifest_sha256':manifest_sha,'raw_manifest_sha256':raw_sha,
              'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path),'parameter_count':48453,
              'environment':environment_identity(),'dependency_sha256':dependency_hashes(),
              'initial_state_sha256':complete['initial_state_sha256']}
    if any(payload.get(key)!=value for key,value in expected.items()):
        raise ValueError(f'{arm}/{kind}: checkpoint settings differ')
    order=validate_training_order(complete['history'],manifest)
    expected_keys={(x['sequence'],x['frame'],x['split']) for x in manifest['samples'] if x['split']=='validation'} | {('seq4',21,'train'),('seq4',24,'train')}
    rows=frozen['rows'];mapping={(x['sequence'],x['frame'],x['split']):x for x in rows}
    if len(mapping)!=len(rows) or set(mapping)!=expected_keys:
        raise ValueError('Frozen prediction coverage differs')
    for row in rows:
        for name in ('baseline','refined'):
            path=Path(row[name]).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path)!=row[name+'_sha256']:
                raise ValueError('Frozen mask changed or path escaped run')
    return {'folder':folder,'complete':complete,'payload':payload,'mapping':mapping,'order':order,
            'completion_sha256':digest(complete_path),'checkpoint_sha256':digest(checkpoint_path),
            'frozen_sha256':digest(frozen_path)}


def evaluate(manifest_dir,raw_audit,plan_path,parity_path,run_root,output):
    manifest,manifest_sha,entries,raw_sha=load_manifests(manifest_dir,raw_audit)
    environment_identity();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    runs={}
    for arm in ARMS:
        for kind in LOSSES:
            validate_plan(plan_path,parity_path,manifest_sha,raw_sha)
            folder=run_root/f'{arm}_{kind}'
            runs[(arm,kind)]=check_run(folder,arm,kind,manifest,manifest_sha,raw_sha,plan_path,parity_path)
    initials={x['payload']['initial_state_sha256'] for x in runs.values()}
    orders=[x['order'] for x in runs.values()]
    if len(initials)!=1 or any(order!=orders[0] for order in orders[1:]):
        raise ValueError('Arms did not use identical initial weights and training order')
    targets=sorted((x for x in manifest['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    targets += [next(x for x in manifest['samples'] if x['sequence']=='seq4' and x['frame']==i and x['split']=='train') for i in (21,24)]
    baseline_hashes={}
    # No validation masks or labels are opened here until all six runs exist;
    # replay every saved mask from its frozen checkpoint before any label read.
    for (arm,kind),run in runs.items():
        model=model_for(arm)
        model.load_trained_state(run['payload'],backbone_sha256=manifest['checkpoint_sha256'],
            feature_pipeline_sha256=manifest['feature_pipeline_sha256'],train_manifest_sha256=manifest_sha,raw_manifest_sha256=raw_sha)
        if sum(p.numel() for p in model.parameters())!=48453:raise ValueError('Capacity differs')
        model.to('mps').eval()
        with torch.no_grad():
            for item in targets:
                key=(item['sequence'],item['frame'],item['split']);row=run['mapping'][key]
                args,shape=inputs(item,entry_for(entries,item),'mps')
                computed={'baseline':(to_video(args[2],shape)>0).cpu().numpy()[0,0],
                          'refined':(to_video(model(*args),shape)>0).cpu().numpy()[0,0]}
                if list(shape)!=row['shape']:raise ValueError('Video geometry differs')
                for name,mask in computed.items():
                    saved=np.asarray(Image.open(row[name]).convert('L'))>0
                    if saved.shape!=mask.shape or not np.array_equal(saved,mask):
                        raise ValueError(f'{arm}/{kind}/{key}: frozen {name} does not reproduce')
                baseline_hash=run['mapping'][key]['baseline_sha256']
                if key in baseline_hashes and baseline_hashes[key]!=baseline_hash:
                    raise ValueError('Frozen baseline masks differ across arms')
                baseline_hashes[key]=baseline_hash
        del model
    scores=[]
    # Labels are opened only after all 144 saved baseline/refined masks replayed.
    for item in targets:
        key=(item['sequence'],item['frame'],item['split'])
        shape=tuple(next(iter(runs.values()))['mapping'][key]['shape'])
        truth=load_target(item,shape)[0,0].numpy().astype(bool)
        row={'sequence':item['sequence'],'frame':item['frame'],'split':item['split'],'present':bool(truth.any()),'methods':{}}
        for (arm,kind),run in runs.items():
            saved=run['mapping'][key]
            for output_name in ('baseline','refined'):
                if output_name=='baseline' and 'baseline' in row['methods']:continue
                pred=np.asarray(Image.open(saved[output_name]).convert('L'))>0
                den=int(pred.sum()+truth.sum());name='baseline' if output_name=='baseline' else arm+'_'+kind
                row['methods'][name]={'dice':2*int((pred&truth).sum())/den if den else 1.0,'predicted_pixels':int(pred.sum())}
        scores.append(row)
    validation=[x for x in scores if x['split']=='validation'];present=[x for x in validation if x['present']];empty=[x for x in validation if not x['present']]
    names=['baseline']+[arm+'_'+kind for arm in ARMS for kind in LOSSES]
    videos=sorted({x['sequence'] for x in present})
    summary={name:{'present_frames':len(present),'mean_present_frame_dice':float(np.mean([x['methods'][name]['dice'] for x in present])),
                   'mean_per_video_present_dice':float(np.mean([np.mean([x['methods'][name]['dice'] for x in present if x['sequence']==video]) for video in videos])),
                   'empty_frames':len(empty),'empty_false_alarms':sum(x['methods'][name]['predicted_pixels']>0 for x in empty),
                   'mean_empty_predicted_pixels':float(np.mean([x['methods'][name]['predicted_pixels'] for x in empty]))} for name in names}
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':manifest['scope'],'warning':'Retrospective development samples and reused validation; not patient independent or a promotion decision.',
            'manifest_sha256':manifest_sha,'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path),
            'initial_state_sha256':next(iter(initials)),'runner_source_sha256':digest(ROOT/'scripts/train_raw_multimask_study_v1.py'),
            'scorer_source_sha256':digest(Path(__file__)),'runs':{arm+'_'+kind:{k:v for k,v in run.items() if k.endswith('_sha256')} for (arm,kind),run in runs.items()},
            'summary':summary,'rows':scores,'candidate_note':'Raw candidate residual can override original mask suppression; native presence and pointer are unchanged. No live inference was run.'}
    path=out/'scores.json';path.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'scores_sha256':digest(path)},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('data-preflight','raw-audit','plan','parity-receipt','run-root','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args();torch.use_deterministic_algorithms(True)
    evaluate(args.data_preflight,args.raw_audit,args.plan,args.parity_receipt,args.run_root,args.output)

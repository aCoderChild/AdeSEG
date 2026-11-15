#!/usr/bin/env python3
"""Independently rescore every frozen fidelity-development prediction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import check_manifest
from scripts.train_refiner_fidelity_pilot import SEEDS, WEIGHT


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-preflight',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest,mhash=check_manifest(a.data_preflight)
    labels={(x['sequence'],x['frame']):x for x in manifest['samples'] if x['split']=='validation'}
    all_runs=[];baseline_hashes=None
    for arm in ('multilevel','mask_only'):
        for seed in SEEDS:
            folder=a.runs/f'{arm}_{seed}';receipt=read(folder/'completion.json')
            if not receipt.get('passed') or receipt.get('manifest_sha256')!=mhash or receipt.get('seed')!=seed or receipt.get('arm')!=arm:raise ValueError('Run identity mismatch')
            if receipt['checkpoint_sha256']!=digest(folder/'checkpoint.pt') or receipt['frozen_predictions_sha256']!=digest(folder/'frozen_predictions.json'):raise ValueError('Checkpoint or frozen receipt changed')
            checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            if checkpoint.get('runner_source_sha256')!=digest(ROOT/'scripts/train_refiner_fidelity_pilot.py') or checkpoint.get('arm')!=arm or checkpoint.get('seed')!=seed or checkpoint.get('fidelity_weight')!=WEIGHT or checkpoint.get('epochs')!=30 or checkpoint.get('training_steps')!=510:raise ValueError('Training config mismatch')
            frozen=read(folder/'frozen_predictions.json')
            if frozen['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('Predictions bound to another model')
            result=[];current_hashes={};seen=set()
            if len(frozen['rows'])!=len(labels):raise ValueError('Validation coverage differs')
            for row in frozen['rows']:
                key=(row['sequence'],row['frame'])
                if key not in labels or key in seen:raise ValueError('Unexpected or duplicate validation frame')
                seen.add(key)
                item=labels[key]
                if digest(item['label'])!=item['label_sha256']:raise ValueError('Label changed')
                truth=(load_label_mask(Path(item['label']),[1])==1)
                scores={}
                for name in ('baseline','refined'):
                    path=Path(row[name])
                    if digest(path)!=row[name+'_sha256']:raise ValueError('Frozen prediction changed')
                    pred=np.asarray(Image.open(path).convert('L'))>0
                    if truth.shape!=pred.shape:raise ValueError('Prediction geometry differs')
                    den=int(pred.sum()+truth.sum());scores[name]={'dice':2*int((pred&truth).sum())/den if den else 1.,'predicted_pixels':int(pred.sum())}
                current_hashes[str(key)]=row['baseline_sha256']
                result.append({'sequence':key[0],'frame':key[1],'labeled_present':bool(truth.any()),**scores})
            if baseline_hashes is None:baseline_hashes=current_hashes
            if current_hashes!=baseline_hashes:raise ValueError('Baselines differ between runs')
            old={(x['sequence'],x['frame']):x for x in receipt['rows']}
            for row in result:
                prior=old[(row['sequence'],row['frame'])]
                if prior['labeled_present']!=row['labeled_present'] or any(prior[n]!=row[n] for n in ('baseline','refined')):raise ValueError('Independent rescore differs')
            present=[x for x in result if x['labeled_present']];empty=[x for x in result if not x['labeled_present']]
            summary={n:{'mean_present_dice':float(np.mean([x[n]['dice'] for x in present])),
                        'empty_false_alarms':sum(x[n]['predicted_pixels']>0 for x in empty),
                        'empty_frames':len(empty),'mean_empty_predicted_pixels':float(np.mean([x[n]['predicted_pixels'] for x in empty]))} for n in ('baseline','refined')}
            all_runs.append({'arm':arm,'seed':seed,'summary':summary,'frames':result,'checkpoint_sha256':receipt['checkpoint_sha256'],'completion_sha256':digest(folder/'completion.json')})
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    report={'passed':True,'scope':'Exploratory previously inspected validation; not independent or a video propagation result.',
            'manifest_sha256':mhash,'auditor_sha256':digest(Path(__file__)),'runs':all_runs}
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Fidelity control: independently rescored fixed-read development','',report['scope'],'',
           '| Model | Seed | Present Dice | Empty false alarms | Mean empty predicted pixels |','|---|---:|---:|---:|---:|']
    base=all_runs[0]['summary']['baseline']
    lines.append(f"| Captured baseline | — | {base['mean_present_dice']:.6f} | {base['empty_false_alarms']}/{base['empty_frames']} | {base['mean_empty_predicted_pixels']:.0f} |")
    for r in all_runs:
        s=r['summary']['refined'];lines.append(f"| {r['arm']} | {r['seed']} | {s['mean_present_dice']:.6f} | {s['empty_false_alarms']}/{s['empty_frames']} | {s['mean_empty_predicted_pixels']:.0f} |")
    (out/'analysis.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

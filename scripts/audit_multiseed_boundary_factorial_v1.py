#!/usr/bin/env python3
"""Independently rescore 32 frozen two-seed video runs with SciPy.

No labels are opened until launch/gate/source evidence and all predictions
have been rechecked. Gate and routing validation reuse reviewed helpers;
pixel arithmetic and boundary morphology use the independent audit functions.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.audit_boundary_video_factorial_v1 import sha,require,label,boundary_target,metrics,summarize,paired,compare
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.run_multiseed_boundary_factorial_v1 import BASE,SEEDS,SEQUENCES,ARMS
from scripts.score_multiseed_boundary_factorial_v1 import (mapped_runs,verify_launch,validate_all,
    label_metadata,install_prelabel_guard)


def read(path):return json.loads(Path(path).read_text())


def precheck(run):
    run=Path(run).resolve(strict=True);require(run.is_relative_to(BASE.resolve(strict=True)),'Run outside experiment root')
    saved=read(run/'scores/results.json')
    require(saved.get('passed') is True and
        saved.get('prelabel_guard',{}).get('open_attempts')==0 and
        saved.get('scorer_sha256')==sha(ROOT/'scripts/score_multiseed_boundary_factorial_v1.py') and
        saved.get('report_sha256')==sha(run/'scores/report.md') and
        saved.get('run_map_sha256')==sha(run/'run_map.json') and
        saved.get('launch_manifest_sha256')==sha(run/'run_manifest.json'),
        'Score artifacts or source differ')
    raw,mapping=mapped_runs(run);gates=verify_launch(run,raw,score_complete=True)
    expected,receipts,telemetry,timing=validate_all(mapping,gates)
    require(saved.get('suppression_telemetry')==telemetry and saved.get('timing')==timing,'Telemetry differs')
    for seed in SEEDS:
        key=str(seed)
        for seq in SEQUENCES:
            for arm in ARMS:
                require(saved['run_receipt_sha256'][key][seq][arm]==sha(mapping[key][seq][arm]/'completion.json'),
                    'Completion binding differs')
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
    return run,mapping,videos,receipts,saved


def audit(run):
    guard=install_prelabel_guard()
    try:
        run,mapping,videos,receipts,saved=precheck(run)
        # Labels are not opened until both seeds and all predictions validate.
        metadata=label_metadata(videos)
        require(guard['open_attempts']==0,'Label open attempted during independent precheck')
    finally:
        guard['active']=False
    errors=[];labels_checked=0;predictions_checked=0
    for seed in SEEDS:
        key=str(seed)
        for seq in SEQUENCES:
            rows=[];start=receipts[key][seq]['A']['prompt_frame']
            for i,frame in enumerate(videos[seq]):
                label_sha=metadata[seq][i]['label_sha256']
                require(sha(frame.mask_path)==label_sha,'Label hash changed')
                gt=label(frame.mask_path);labels_checked+=1;target=boundary_target(gt)
                require(sha(frame.mask_path)==label_sha,'Label changed while decoding')
                row={'post_prompt':i>start}
                for arm in ARMS:
                    path=mapping[key][seq][arm]/f'{i:06d}.png'
                    wanted=receipts[key][seq][arm]['masks'][i]['sha256']
                    require(sha(path)==wanted,'Prediction changed before decoding')
                    with Image.open(path) as image:
                        pred=np.asarray(image.convert('L'))>0
                    require(sha(path)==wanted,'Prediction changed while decoding')
                    row[arm]=metrics(pred,gt,target);predictions_checked+=1
                    compare(errors,[key,seq,i,arm],row[arm],saved['rows'][key][seq][i][arm])
                rows.append(row)
            for arm in ARMS:
                for name,post in (('post_prompt',True),('all',False)):
                    compare(errors,[key,seq,arm,name],summarize(rows,arm,post),saved['summaries'][key][seq][arm][name])
            for name,arm,base in (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A')):
                compare(errors,[key,seq,name],paired(rows,arm,base),saved['paired_descriptive'][key][seq][name])
            compare(errors,[key,seq,'preprompt'],{'frames':start,'present_misses':sum(x['A']['present'] for x in rows[:start])},
                saved['preprompt'][key][seq])
    return {'passed':not errors,'scope':'Independent SciPy metric audit; gate/routing checks reuse reviewed helpers.',
        'run':str(run),'score_sha256':sha(run/'scores/results.json'),'run_manifest_sha256':sha(run/'run_manifest.json'),
        'run_map_sha256':sha(run/'run_map.json'),'audit_source_sha256':sha(Path(__file__)),
        'label_files_checked':labels_checked,'prediction_files_checked':predictions_checked,
        'prelabel_guard':dict(guard),'error_count':len(errors),'errors':errors}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=validate_output(a.output)
    require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Audit outside experiment folder or exists')
    result=audit(a.run);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'errors':result['error_count'],'output':str(out)}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()

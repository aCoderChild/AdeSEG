#!/usr/bin/env python3
"""Independently rescore the 36 training/selection-exposed video outputs.

Routing/provenance checks reuse the reviewed primary verifier; segmentation and
boundary metrics use the existing independent SciPy implementation. This is
descriptive in-sample evidence, not a held-out or independent test.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.run_training_exposed_boundary_factorial_v1 import ARMS,BASE,SEQUENCES
from scripts.score_training_exposed_boundary_factorial_v1 import (
    install_prelabel_guard,mapped_runs,verify_launch,verify_runs)
from scripts.audit_boundary_video_factorial_v1 import sha,require,label,boundary_target,metrics,summarize,paired,compare


def read(path):return json.loads(Path(path).read_text())


def audit(run):
    run=Path(run).resolve(strict=True)
    require(run.is_relative_to(BASE.resolve(strict=True)), 'Run outside experiment root')
    guard=install_prelabel_guard()
    try:
        score_path=run/'scores/results.json';saved=read(score_path)
        require(saved.get('passed') is True and saved.get('prelabel_guard',{}).get('open_attempts')==0
            and saved.get('sequence_exposure')=={s:('selection_validation' if s in ('seq9','seq15') else 'training') for s in SEQUENCES}
            and saved.get('scorer_sha256')==sha(ROOT/'scripts/score_training_exposed_boundary_factorial_v1.py')
            and saved.get('report_sha256')==sha(run/'scores/report.md')
            and saved.get('run_map_sha256')==sha(run/'run_map.json')
            and saved.get('launch_manifest_sha256')==sha(run/'run_manifest.json'), 'Score identity differs')
        mapping=mapped_runs(run/'run_map.json',BASE)
        _,_,inputs,phase_receipts=verify_launch(run/'run_map.json',mapping,score_complete=True)
        videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
        receipts,telemetry,timing=verify_runs(mapping,videos,inputs['preflight'],inputs,phase_receipts)
        require(saved.get('suppression_telemetry')==telemetry and saved.get('timing')==timing,'Telemetry differs')
        pre=read(inputs['preflight']/'manifest.json'); metadata={}
        for seq in SEQUENCES:
            item=pre['sequences'][seq]; path=Path(item['source'])/seq/'frames.json'
            require(sha(path)==item['archived_frames_sha256'],'Archived metadata differs')
            # Load only after all outputs are verified. This contains historical outcomes.
            metadata[seq]=read(path)
            require(sha(path)==item['archived_frames_sha256'] and len(metadata[seq])==len(videos[seq]),
                    'Archived metadata changed or frame coverage differs')
            for arm in ARMS:
                require(saved['run_receipt_sha256'][seq][arm]==sha(mapping[seq][arm]/'completion.json'),
                        'Receipt changed after scoring')
        require(guard['open_attempts']==0,'Label opened before complete verification')
    finally:
        guard['active']=False
    errors=[]; label_count=0; prediction_count=0
    for seq in SEQUENCES:
        rows=[];start=receipts[seq]['A']['prompt_frame']
        for i,frame in enumerate(videos[seq]):
            target_hash=metadata[seq][i]['label_sha256']
            require(sha(frame.mask_path)==target_hash,'Label changed before decoding')
            gt=label(frame.mask_path);boundary=boundary_target(gt);label_count+=1
            require(sha(frame.mask_path)==target_hash,'Label changed during decoding')
            row={'post_prompt':i>start}
            for arm in ARMS:
                file=mapping[seq][arm]/f'{i:06d}.png';want=receipts[seq][arm]['masks'][i]['sha256']
                require(sha(file)==want,'Prediction changed before decoding')
                with Image.open(file) as image:pred=np.asarray(image.convert('L'))>0
                require(sha(file)==want,'Prediction changed during decoding')
                row[arm]=metrics(pred,gt,boundary);prediction_count+=1
                compare(errors,[seq,i,arm],row[arm],saved['rows'][seq][i][arm])
            rows.append(row)
        for arm in ARMS:
            for name,post in (('post_prompt',True),('all',False)):
                compare(errors,[seq,arm,name],summarize(rows,arm,post),saved['summaries'][seq][arm][name])
        for name,arm,base in (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A')):
            compare(errors,[seq,name],paired(rows,arm,base),saved['paired_descriptive'][seq][name])
        compare(errors,[seq,'preprompt'],{'frames':start,'present_misses':sum(x['A']['present'] for x in rows[:start])},saved['preprompt'][seq])
    return {'passed':not errors,'scope':'Independent metric recomputation; gate and routing checks reuse reviewed primary verifier. In-sample/selection exposed.',
            'run':str(run),'score_sha256':sha(score_path),'audit_source_sha256':sha(Path(__file__)),
            'run_map_sha256':sha(run/'run_map.json'),'run_manifest_sha256':sha(run/'run_manifest.json'),
            'prelabel_guard':dict(guard),'label_files_checked':label_count,'prediction_files_checked':prediction_count,
            'error_count':len(errors),'errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=validate_output(args.output)
    require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid or existing audit output')
    result=audit(args.run);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)
    print(json.dumps({'passed':True,'labels':result['label_files_checked'],'predictions':result['prediction_files_checked']}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Summarize frozen raw-study training traces; never read validation labels."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.raw_multimask_shared_v1 import ARMS,LOSSES
from scripts.score_raw_multimask_study_v1 import validate_training_order


def summarize(run_root,manifest_path,output):
    manifest=json.loads(manifest_path.read_text());rows={};checks=[]
    for arm in ARMS:
        for kind in LOSSES:
            folder=run_root/f'{arm}_{kind}';path=folder/'completion.json';done=json.loads(path.read_text())
            if (not done.get('passed') or done.get('arm')!=arm or done.get('loss_kind')!=kind or done.get('validation_labels_opened') is not False or done.get('manifest_sha256')!=digest(manifest_path)):
                raise ValueError(f'{arm}/{kind}: invalid completion')
            for name,key in (('checkpoint.pt','checkpoint_sha256'),('frozen_predictions.json','frozen_predictions_sha256')):
                if digest(folder/name)!=done[key]:raise ValueError('Frozen output changed')
            order=validate_training_order(done['history'],manifest)
            checks.append(order)
            stats=[]
            for epoch in done['history']:
                samples=epoch['samples'];losses=[x['loss'] for x in samples];grads=[x['maximum_gradient'] for x in samples]
                if not all(math.isfinite(x) for x in losses+grads):raise ValueError('Nonfinite training diagnostic')
                hard={str(x['frame']):{'loss':x['loss'],'maximum_gradient':x['maximum_gradient']} for x in samples if x['sequence']=='seq4' and x['frame'] in (21,24)}
                stats.append({'epoch':epoch['epoch'],'mean_loss':float(np.mean(losses)),'max_gradient':max(grads),'seq4_hard_reads':hard})
            rows[arm+'_'+kind]={'completion_sha256':digest(path),'checkpoint_sha256':done['checkpoint_sha256'],'epochs':stats}
    if any(x!=checks[0] for x in checks[1:]):raise ValueError('Training order differs across controls')
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':'Training trace only. No validation predictions or labels read; no efficacy conclusions.','manifest_sha256':digest(manifest_path),'methods':rows}
    report=out/'training_trace_summary.json';report.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(report),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
    print(json.dumps({k:{'first_loss':v['epochs'][0]['mean_loss'],'last_loss':v['epochs'][-1]['mean_loss'],'peak_gradient':max(x['max_gradient'] for x in v['epochs'])} for k,v in rows.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.run_root,a.manifest,a.output)

#!/usr/bin/env python3
"""Verify, score, or independently audit a matched asinh video batch.

Target masks are blocked until all twelve B/C/D videos and matched archived A
are frozen and the separate no-label verification is complete. The audit uses
independent CPU/SciPy metrics but shares the structural verifier.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from modeling.lean_channel_value_window_runtime_v1 import BASE, runtime_sources, require
from scripts.channel_value_fixed_read_v1 import install_phase_mask_guard
from scripts.crossfit_observer_capture_shared_v1 import DATA,archived_frame_metadata,phase_artifact,sequence_inputs,validate_plan
from scripts.memory_token_io import digest
from scripts.run_channel_value_video_v1 import SEQUENCES,compare_archived
from scripts.score_boundary_video_factorial_v1 import metric,summary,paired_summary
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_channel_value_video_smoke_v1 import verify_smoke_audit

ARMS=('A','B','C','D')


def read(path):return json.loads(Path(path).read_text())


def check_output(path,root):
    out=Path(path).resolve()
    require(out.is_relative_to(root) and not out.exists(),'Invalid or existing output')
    return out


def telemetry_counts(items,start,end):
    require([x.get('frame') for x in items]==list(range(start,end)), 'Telemetry frame coverage differs')
    for item in items:
        require(type(item.get('conditioning')) is bool and type(item.get('native_object_present')) is bool and
                type(item.get('native_exact_suppressed')) is bool and
                isinstance(item.get('object_score_logit'),(float,int)) and math.isfinite(item['object_score_logit']) and
                item['native_object_present']==(item['object_score_logit']>0), 'Object score telemetry differs')
        for name in ('native','refined','output','memory_high_res'):
            part=item[name]
            require(all(math.isfinite(part[k]) for k in ('min','max')) and type(part['positive_pixels']) is int and
                    0<=part['positive_pixels']<=part['numel'] and len(part['sha256'])==64,
                    'Mask telemetry differs')
    post=[x for x in items if not x['conditioning']]
    require([x['frame'] for x in post]==list(range(start+1,end)), 'Post-prompt telemetry differs')
    absent=[x for x in post if not x['native_object_present']]
    return {'post_prompt_frames':len(post),'native_absent_score_frames':len(absent),
        'native_exact_suppressed_frames':sum(x['native_exact_suppressed'] for x in post),
        'refined_positive_with_native_absent_score':sum(x['refined']['positive_pixels']>0 for x in absent),
        'output_positive_with_native_absent_score':sum(x['output']['positive_pixels']>0 for x in absent)}


def validate_frozen(root,*,completed=()):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve(strict=True)),'Batch outside Drive')
    launch_file=root/'run_manifest.json';launch=read(launch_file)
    require(launch.get('sequences')==list(SEQUENCES) and launch.get('full_arms')==['B','C','D'] and
            launch.get('runtime_sources')==runtime_sources() and launch.get('python_hash_seed')=='0' and
            launch.get('mps_fallback')=='0' and launch.get('mps_fast_math')=='0','Launch sources or settings differ')
    from scripts.run_channel_value_video_study_v1 import phases,source_hashes
    require(launch.get('sources')==source_hashes(),'Launcher/scorer sources changed')
    phase_rows=[json.loads(x) for x in (root/'phases.jsonl').read_text().splitlines() if x.strip()]
    expected=[name for name,_ in phases(root) if name not in ('verify','score','audit')]+list(completed)
    require([x.get('phase') for x in phase_rows]==expected and all(x.get('passed') is True and
        x.get('log_sha256')==digest(root/(x['phase']+'.log')) for x in phase_rows),'Batch phase coverage differs')
    cert=root/'certificate/certificate.json';cert_sha=phase_artifact(root,'certificate',cert)
    audit=root/'smoke_audit/audit.json';smoke_sha=phase_artifact(root,'smoke_audit',audit)
    verify_smoke_audit(audit,root,cert_sha,reconstruct=True)
    plan=validate_plan();inputs={};receipts={};references={};telemetry={}
    for seq in SEQUENCES:
        frames,records,start,_,_,baseline,old,prompt=sequence_inputs(plan,seq)
        metadata=archived_frame_metadata(plan,seq)
        inputs[seq]={'prompt_frame':start,'prompt':prompt,'images':records,'label_metadata':metadata}
        require(old.get('passed') is True and old.get('full_video') is True and
                len(old['masks'])==len(frames) and [r['frame'] for r in old['masks']]==list(range(len(frames))),
                'Archived matched A coverage differs')
        references[seq]={'path':str(baseline),'sha256':digest(baseline)}
        for row in old['masks']:
            i=row['frame'];path=baseline.parent/f'{i:06d}.png'
            require(digest(path)==row['sha256'],'Archived A mask changed')
            with Image.open(frames[i].image_path) as source,Image.open(path) as image:
                values=np.asarray(image.convert('L'))
                require(image.format=='PNG' and image.size==source.size and np.isin(values,(0,255)).all() and
                        (i>=start or not values.any()),'Archived A mask geometry/encoding differs')
            require(digest(path)==row['sha256'],'Archived A mask changed during reading')
        receipts[seq]={};telemetry[seq]={}
        old_telemetry=baseline.parent/'telemetry.json';old_telemetry_sha=old['telemetry_sha256']
        require(digest(old_telemetry)==old_telemetry_sha,'Archived A telemetry changed')
        old_items=read(old_telemetry);require(digest(old_telemetry)==old_telemetry_sha,'Archived A telemetry changed while reading')
        telemetry[seq]['A']=telemetry_counts(old_items,start,len(frames))
        for arm in ('B','C','D'):
            folder=root/'full'/seq/arm;file=folder/'receipt.json'
            sha=phase_artifact(root,f'full_{seq}_{arm}',file);rec=read(file)
            require(digest(file)==sha and rec.get('passed') is True and rec.get('full_video') is True and
                rec.get('limit')==0 and rec.get('sequence')==seq and rec.get('arm')==arm and
                rec.get('certificate_sha256')==cert_sha and rec.get('smoke_audit_sha256')==smoke_sha and
                rec.get('prompt')==prompt and rec.get('prompt_frame')==start and rec.get('input_images')==records and
                rec.get('baseline_completion_sha256')==digest(baseline) and
                rec.get('runner_sha256')==digest(ROOT/'scripts/run_channel_value_video_v1.py') and
                rec.get('label_guard',{}).get('open_attempts')==0,'Full video receipt differs')
            require([r['frame'] for r in rec['masks']]==list(range(len(frames))) and
                    [r['frame'] for r in rec['rows']]==list(range(start,len(frames))),'Full video frame coverage differs')
            for row in rec['masks']:
                i=row['frame'];path=folder/f'{i:06d}.png';require(digest(path)==row['sha256'],'Full mask changed')
                with Image.open(frames[i].image_path) as source,Image.open(path) as image:
                    m=np.asarray(image.convert('L'))
                    require(image.format=='PNG' and image.size==source.size and np.isin(m,(0,255)).all() and
                            (i>=start or not m.any()),'Full mask geometry/encoding differs')
                require(digest(path)==row['sha256'],'Full mask changed while reading')
            traces={}
            for key in ('memory_events','positions','selections','telemetry'):
                path=folder/(key+'.json');want=rec['trace_sha256'][key]
                require(digest(path)==want,'Full trace changed');traces[key]=read(path)
                require(digest(path)==want,'Full trace changed while reading')
            telemetry[seq][arm]=telemetry_counts(traces['telemetry'],start,len(frames))
            compare_archived(rec['rows'],{k:traces[k] for k in ('memory_events','positions','selections')},
                             baseline,old,arm,False)
            runtime=folder/'verification/runtime_verification.json';require(digest(runtime)==rec['runtime_verification_sha256'],
                                                                            'Runtime receipt changed')
            state=read(runtime);require(digest(runtime)==rec['runtime_verification_sha256'] and
                state.get('passed') is True and state.get('certificate_sha256')==cert_sha and
                state.get('runtime_sources')==runtime_sources() and state.get('device')=='mps' and
                state.get('memory_budget')==16384 and state.get('insurance_reserve')==3072,'Runtime identity differs')
            for name in ('model','refiner'):
                require(digest(folder/'verification'/(name+'_tensor_fingerprints.json'))==state[name+'_tensor_fingerprints_sha256'],
                        'Runtime fingerprint changed')
            receipts[seq][arm]={'path':str(file),'sha256':sha}
    return {'launch_sha256':digest(launch_file),'certificate_sha256':cert_sha,'smoke_audit_sha256':smoke_sha,
            'inputs':inputs,'receipts':receipts,'references':references,'telemetry':telemetry}


def proof_verified(root,verification,completed):
    proof=read(verification)
    require(proof.get('passed') is True and proof.get('label_guard',{}).get('open_attempts')==0 and
            proof.get('source_sha256')==digest(Path(__file__)),'Verification receipt differs')
    current=validate_frozen(root,completed=completed)
    require(all(current[k]==proof[k] for k in current),'Frozen predictions changed')
    return proof


def compute(root,proof,*,independent=False):
    if independent:
        import math
        from scripts.audit_boundary_video_factorial_v1 import label,metrics as calculate,summarize as summarise,paired
        average=lambda values:math.fsum(values)/len(values) if values else None
    else:
        label=lambda path:load_label_mask(path,[1])==1
        calculate=metric;summarise=summary;paired=paired_summary
        average=lambda values:float(np.mean(values)) if values else None
    plan=validate_plan();rows_all={};summaries={};pairs={};preprompt={}
    for seq in SEQUENCES:
        frames,_,start,_,_,baseline,_,_=sequence_inputs(plan,seq)
        meta=proof['inputs'][seq]['label_metadata'];path=Path(meta['path'])
        require(digest(path)==meta['sha256'],'Label metadata changed');metadata=read(path)
        require(digest(path)==meta['sha256'] and len(metadata)==len(frames),'Label metadata changed during read')
        folders={'A':baseline.parent}|{arm:root/'full'/seq/arm for arm in ('B','C','D')}
        hashes={}
        for arm,folder in folders.items():
            receipt=baseline if arm=='A' else folder/'receipt.json'
            expected=proof['references'][seq]['sha256'] if arm=='A' else proof['receipts'][seq][arm]['sha256']
            require(digest(receipt)==expected,'Receipt changed before scoring');rec=read(receipt)
            require(digest(receipt)==expected,'Receipt changed during scoring');hashes[arm]={x['frame']:x['sha256'] for x in rec['masks']}
        rows=[]
        for i,frame in enumerate(frames):
            image_sha=proof['inputs'][seq]['images'][i]['image_sha256'];target_sha=metadata[i]['label_sha256']
            require(digest(frame.image_path)==image_sha and digest(frame.mask_path)==target_sha,'Image or label changed')
            truth=label(frame.mask_path);require(digest(frame.mask_path)==target_sha,'Label changed during decode')
            row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,'initialization_frame':i==start}
            for arm,folder in folders.items():
                file=folder/f'{i:06d}.png';require(digest(file)==hashes[arm][i],'Prediction changed')
                with Image.open(file) as im:prediction=np.asarray(im.convert('L'))>0
                require(digest(file)==hashes[arm][i],'Prediction changed during decode')
                row[arm]=calculate(prediction,truth)
            require(digest(frame.image_path)==image_sha,'Image changed during score');rows.append(row)
        rows_all[seq]=rows
        if independent:
            summaries[seq]={a:{'post_prompt':summarise(rows,a,True),'all':summarise(rows,a,False)} for a in ARMS}
        else:
            summaries[seq]={a:{'post_prompt':summarise(rows,a),'all':summarise(rows,a,False)} for a in ARMS}
        pairs[seq]={f'{a}_minus_A':paired(rows,a,'A') for a in ('B','C','D')}
        preprompt[seq]={'frames':start,'present_misses':sum(r['A']['present'] for r in rows[:start])}
    overall={}
    for arm in ARMS:
        patient=[summaries[s][arm]['post_prompt']['mean_present_dice'] for s in SEQUENCES]
        patient=[v for v in patient if v is not None]
        present=[r[arm]['dice'] for s in SEQUENCES for r in rows_all[s] if r['post_prompt'] and r[arm]['present']]
        empty=[r[arm] for s in SEQUENCES for r in rows_all[s] if r['post_prompt'] and not r[arm]['present']]
        overall[arm]={'patients_with_present_frames':len(patient),
            'patient_macro_present_dice':average(patient),
            'present_frames':len(present),'frame_weighted_present_dice':average(present),
            'empty_frames':len(empty),'false_alarms':sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels':average([x['predicted_pixels'] for x in empty])}
    return {'rows':rows_all,'summaries':summaries,'paired':pairs,'preprompt':preprompt,'overall':overall}


def compare(actual,expected,path,errors):
    from scripts.audit_boundary_video_factorial_v1 import same
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected):errors.append([*path,'keys'])
        for k in actual.keys() & expected.keys():compare(actual[k],expected[k],[*path,k],errors)
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected):errors.append([*path,'length'])
        for i,(a,b) in enumerate(zip(actual,expected,strict=False)):compare(a,b,[*path,i],errors)
    elif isinstance(actual,str) or isinstance(expected,str):
        if actual!=expected:errors.append([*path,actual,expected])
    elif not same(actual,expected):errors.append([*path,actual,expected])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','score','audit'),required=True)
    p.add_argument('--batch-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.batch_root.resolve(strict=True);out=check_output(a.output,root)
    if a.mode=='verify':
        guard=install_mask_open_guard(DATA);result=validate_frozen(root)
        require(guard['open_attempts']==0,'Target label opened during verification')
        result.update(passed=True,label_guard=dict(guard),source_sha256=digest(Path(__file__)))
        out.mkdir(parents=True);(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        guard=install_phase_mask_guard(DATA);verification=root/'verification/verification.json'
        expected=phase_artifact(root,'verify',verification);completed=('verify',) if a.mode=='score' else ('verify','score')
        proof=proof_verified(root,verification,completed)
        require(digest(verification)==expected and guard['open_attempts']==0,'Verification changed or target opened')
        if a.mode=='audit':
            score_path=root/'scores/results.json';score_sha=phase_artifact(root,'score',score_path)
            saved=read(score_path);require(digest(score_path)==score_sha and saved.get('passed') is True and
                saved.get('verification_sha256')==expected and saved.get('prelabel_guard',{}).get('open_attempts')==0 and
                saved.get('report_sha256')==digest(root/'scores/report.md'),'Score changed')
        guard['active']=False
        result=compute(root,proof,independent=a.mode=='audit')
        out.mkdir(parents=True)
        if a.mode=='score':
            report=out/'report.md';lines=['# Exploratory asinh channel-value video comparison','',
                'A is the matched archived project Lean baseline; B output-only, C memory-mask-only, D both. '
                'These previously inspected sequences are not independent validation.','',
                '| Sequence | Present | Dice A / B / C / D | Empty alarms A / B / C / D |',
                '|---|---:|---:|---:|']
            for seq in SEQUENCES:
                s=result['summaries'][seq];d=' / '.join('n/a' if s[x]['post_prompt']['mean_present_dice'] is None else
                    f"{s[x]['post_prompt']['mean_present_dice']:.4f}" for x in ARMS)
                alarms=' / '.join(f"{s[x]['post_prompt']['false_alarms']}/{s[x]['post_prompt']['empty_frames']}" for x in ARMS)
                lines.append(f"| {seq} | {s['A']['post_prompt']['present_frames']} | {d} | {alarms} |")
            lines+=['','| Arm | Patient-macro Dice | Present patients | Frame-weighted Dice | Present frames | Empty false alarms |',
                    '|---|---:|---:|---:|---:|---:|']
            for arm in ARMS:
                s=result['overall'][arm]
                macro='n/a' if s['patient_macro_present_dice'] is None else f"{s['patient_macro_present_dice']:.4f}"
                micro='n/a' if s['frame_weighted_present_dice'] is None else f"{s['frame_weighted_present_dice']:.4f}"
                lines.append(f"| {arm} | {macro} | {s['patients_with_present_frames']} | {micro} | "
                             f"{s['present_frames']} | {s['false_alarms']}/{s['empty_frames']} |")
            lines+=['','Native-absence score and refined-mask disagreements are observational telemetry, not a gate.']
            for seq in SEQUENCES:
                entries=proof['telemetry'][seq]
                lines.append(f"- {seq}: refined-positive frames while native score is absent, B/C/D: " +
                    ' / '.join(f"{entries[x]['refined_positive_with_native_absent_score']}/{entries[x]['native_absent_score_frames']}" for x in ('B','C','D')))
            report.write_text('\n'.join(lines)+'\n')
            result.update(passed=True,verification_sha256=expected,prelabel_guard=dict(guard),
                          telemetry=proof['telemetry'],report_sha256=digest(report))
            (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        else:
            errors=[]
            for key in ('rows','summaries','paired','preprompt','overall'):compare(result[key],saved[key],[key],errors)
            (out/'audit.json').write_text(json.dumps({'passed':not errors,'scope':'Independent CPU metrics; shared structural verifier.',
                'score_sha256':score_sha,'prelabel_guard':dict(guard),'error_count':len(errors),'errors':errors},indent=2)+'\n')
            require(not errors,'Independent metric audit differs')


if __name__=='__main__':main()

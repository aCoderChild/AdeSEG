#!/usr/bin/env python3
"""Queue-only-by-default short gates or 143-read cross-fit observer capture.

One GPU subprocess runs at a time under the project's exclusive lock. The
launcher preserves partial attempts and stops on source, no-op or space failure.
No annotations, detector, refiner, or extra memory-selection rule is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.crossfit_observer_capture_shared_v2 import (BASE,DATA,FRESH,GUARD,MARGIN,OUT,PER_READ,PLAN_SHA,SHORT,
    phase_artifact,read,require,require_headroom,sequence_inputs,source_hashes,v1_source_hashes,validate_plan)

RUNNER=ROOT/'scripts/run_crossfit_observer_capture_v2.py'
SHORT_AUDITOR=ROOT/'scripts/audit_crossfit_observer_capture_v1.py'
FULL_AUDITOR=ROOT/'scripts/audit_crossfit_observer_full_sequence_v2.py'


def phases(stage,root,engineering=None,sequence=None,remaining=143):
    root=Path(root);tasks=[]
    if stage=='full':
        require(engineering is not None,'Full capture requires the audited short gate')
        require(sequence in FRESH,'Full stage requires one planned sequence')
        seq=sequence;capture=root/seq/'capture'
        cmd=[sys.executable,str(RUNNER),'--sequence',seq,'--mode','capture','--limit','0',
             '--remaining-reads',str(remaining),'--engineering-audit',str(engineering),'--output',str(capture)]
        tasks.append((f'{seq}_capture',cmd,capture))
        tasks.append((f'{seq}_audit',[sys.executable,str(FULL_AUDITOR),'--sequence',seq,
            '--capture',str(capture),'--engineering-audit',str(engineering),
            '--output',str(root/seq/'audit')],None))
    else:raise ValueError('Unknown capture stage')
    return tasks


def tensor_bytes(folder):
    if folder is None:return 0
    return sum(p.stat().st_size for directory in ('features','snapshots')
               for p in (folder/directory).glob('*.pt') if p.is_file())


def headroom_with_written(base,remaining,written,sequence_budget):
    """Reserve all remaining reads, less actual bytes already written for this sequence."""
    import shutil
    released=min(max(0,written),sequence_budget)
    free=shutil.disk_usage(base).free
    need=GUARD+MARGIN+max(0,remaining*PER_READ-released)
    require(free>=need,f'Insufficient headroom: {free} < {need}')
    return {'free_bytes':free,'minimum_bytes':need,'remaining_reads':remaining,
            'written_payload_bytes_accounted':released}


def run_phase(command,log,folder,remaining,*,deduct_output=False,written=0,sequence_budget=0):
    """Stop a child if reserve, margin and remaining-payload headroom are lost."""
    env=phase_environment();produced_before=tensor_bytes(folder)
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while True:
            try:
                code=process.wait(timeout=2);break
            except subprocess.TimeoutExpired:
                produced=max(0,tensor_bytes(folder)-produced_before) if deduct_output else 0
                import shutil
                free=shutil.disk_usage(BASE).free
                released=min(written+produced,sequence_budget)
                need=GUARD+MARGIN+max(0,remaining*PER_READ-released)
                if free<need:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError(f'Storage headroom lost during child: {free} < {need}')
        if code!=0:raise RuntimeError(f'Capture phase exited {code}; inspect {log}')
        produced=max(0,tensor_bytes(folder)-produced_before) if deduct_output else 0
        import shutil
        free=shutil.disk_usage(BASE).free;released=min(written+produced,sequence_budget)
        need=GUARD+MARGIN+max(0,remaining*PER_READ-released)
        require(free>=need, f'Storage headroom lost after child: {free} < {need}')


def checked_audits(paths,plan,sources,engineering):
    """Revalidate independently audited sequence batches; never open labels."""
    accepted={}
    for raw in paths:
        file=Path(raw).resolve(strict=True)
        require(file.is_relative_to(OUT.resolve()) and file.name=='audit.json','Invalid prior audit path')
        batch=file.parent.parent.parent;seq=file.parent.parent.name
        require(seq in FRESH and seq not in accepted and file==batch/seq/'audit/audit.json',
                'Duplicate or unexpected prior sequence')
        expected=phase_artifact(batch,f'{seq}_audit',file);record=read(file)
        ref=Path(plan['prompts'][seq]['baseline_completion_path']);cap=batch/seq/'capture/receipt.json'
        require(digest(file)==expected and record.get('passed') is True and record.get('sequence')==seq and
            record.get('plan_sha256')==PLAN_SHA and record.get('source_sha256')==sources and
            record.get('engineering_audit_sha256')==digest(engineering) and
            record.get('auditor_sha256')==digest(FULL_AUDITOR) and
            record.get('label_guard',{}).get('open_attempts')==0 and
            record.get('reference_kind')=='source_bound_archived_A' and
            record.get('reference_receipt_path')==str(ref) and
            record.get('reference_receipt_sha256')==plan['prompts'][seq]['baseline_completion_sha256']==digest(ref) and
            record.get('capture_receipt_sha256')==phase_artifact(batch,f'{seq}_capture',cap) and
            record.get('target_frames')==[r['frame'] for r in plan['fresh_reads'] if r['sequence']==seq],
            'Prior audit or phase binding differs')
        cap_sha=record['capture_receipt_sha256'];capture=read(cap)
        require(digest(cap)==cap_sha and capture.get('passed') is True and
                capture.get('sequence')==seq and capture.get('mode')=='capture' and
                [row['frame'] for row in capture.get('observations',[])]==record['target_frames'],
                'Prior capture receipt or grid differs')
        for row in capture['observations']:
            for key in ('feature','snapshot'):
                file=Path(row[key+'_file']).resolve(strict=True)
                require(file.is_relative_to(cap.parent.resolve()) and digest(file)==row[key+'_sha256'],
                        'Previously captured tensor missing or changed')
        require(digest(cap)==cap_sha and digest(batch/seq/'audit/audit.json')==expected,
                'Prior audit or capture receipt changed during validation')
        accepted[seq]=batch
    require(not accepted or 'seq6' in accepted,'Historical seq6 compatibility must be audited first')
    return accepted


def remaining_count(plan,accepted):
    return len([r for r in plan['fresh_reads'] if r['sequence'] not in accepted])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('full',),required=True);p.add_argument('--engineering-audit',type=Path)
    p.add_argument('--sequence',choices=FRESH);p.add_argument('--prior-audit',type=Path,action='append',default=[])
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    require((a.stage=='full')==(a.engineering_audit is not None),'Full stage alone requires a short audit')
    require((a.stage=='full')==(a.sequence is not None),'Full stage requires exactly one sequence')
    require(a.stage=='full' or not a.prior_audit,'Prior audits apply only to full stage')
    guard=install_mask_open_guard(DATA);plan=validate_plan();sources=source_hashes()
    if a.engineering_audit:
        from scripts.audit_crossfit_observer_capture_v1 import verify_engineering_audit
        verify_engineering_audit(a.engineering_audit,plan,v1_source_hashes())
    accepted=checked_audits(a.prior_audit,plan,sources,a.engineering_audit) if a.stage=='full' else {}
    require(a.stage!='full' or (a.sequence not in accepted and (accepted or a.sequence=='seq6')),
            'Start with seq6 and never rerun an accepted sequence')
    remaining=remaining_count(plan,accepted)
    # Validate all planned source bytes and prompts without reading labels.
    for seq in FRESH:sequence_inputs(plan,seq)
    require(guard['open_attempts']==0,'Annotation open attempted during launcher preflight')
    headroom=require_headroom(BASE,remaining)
    if not a.execute:
        planned=phases(a.stage,OUT/'DRY_RUN',a.engineering_audit,a.sequence,remaining)
        print(json.dumps({'ready_for_requested_stage':True,'stage':a.stage,
                          'scope':'Dry run only; no GPU inference or capture occurred.',
                          'full_capture_authorized_by_this_dry_run':False,
                          'phases':[x[0] for x in planned],
                          'headroom':headroom,'label_open_attempts':guard['open_attempts']}));return
    import torch
    require(torch.backends.mps.is_available(),'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        plan=validate_plan();require(source_hashes()==sources,'Source changed before launch')
        if a.stage=='full':
            accepted=checked_audits(a.prior_audit,plan,sources,a.engineering_audit)
            require(a.sequence not in accepted and (accepted or a.sequence=='seq6'),'Prior sequence state changed')
        remaining=remaining_count(plan,accepted);require_headroom(BASE,remaining)
        root=Path(tempfile.mkdtemp(prefix=f'crossfit_observer_{a.stage}_v2_',dir=OUT))
        tasks=phases(a.stage,root,a.engineering_audit,a.sequence,remaining);manifest=root/'run_manifest.json'
        manifest.write_text(json.dumps({'scope':'Exploratory previously inspected sequences; not held out.',
            'stage':a.stage,'plan_sha256':PLAN_SHA,'source_sha256':sources,'phases':[n for n,_,_ in tasks],
            'engineering_audit_sha256':digest(a.engineering_audit) if a.engineering_audit else None,
            'prior_audits':{seq:{'path':str(batch/seq/'audit/audit.json'),'sha256':digest(batch/seq/'audit/audit.json')} for seq,batch in accepted.items()},
            'remaining_payload_bytes_per_read':PER_READ,'reserve_bytes':GUARD,'margin_bytes':MARGIN,
            'label_guard':dict(guard)},indent=2)+'\n')
        frozen=digest(manifest)
        for name,command,folder in tasks:
            require(digest(manifest)==frozen and source_hashes()==sources and validate_plan()['passed'] is True,
                    'Frozen plan or sources changed')
            require(guard['open_attempts']==0,'Annotation open attempted in launcher')
            current_budget=sum(x['sequence']==a.sequence for x in plan['fresh_reads'])*PER_READ
            written=tensor_bytes(root/a.sequence/'capture') if name.endswith('_audit') else 0
            space=headroom_with_written(BASE,remaining,written,current_budget);start=time.monotonic()
            print(json.dumps({'stage':a.stage,'phase':name,'status':'starting','output':str(root),**space}),flush=True)
            run_phase(command,root/(name+'.log'),folder,remaining,
                      deduct_output=name.endswith('_capture'),written=written,sequence_budget=current_budget)
            require(digest(manifest)==frozen and source_hashes()==sources and validate_plan()['passed'] is True,
                    'Frozen plan or sources changed during phase')
            artifact=(folder/'receipt.json') if folder else (root/'engineering_audit/audit.json' if name=='engineering_audit'
                         else root/name.removesuffix('_audit')/'audit/audit.json')
            require(artifact.is_file(),f'Missing phase artifact: {name}')
            with (root/'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase':name,'passed':True,'artifact_path':str(artifact),'artifact_sha256':digest(artifact),
                    'log_sha256':digest(root/(name+'.log')),'remaining_reads_before':remaining,
                    'wall_seconds':time.monotonic()-start})+'\n')
            if a.stage=='full' and name.endswith('_audit'):
                seq=name.removesuffix('_audit');remaining-=sum(x['sequence']==seq for x in plan['fresh_reads'])
            print(json.dumps({'stage':a.stage,'phase':name,'status':'completed','remaining_reads':remaining}),flush=True)
        require(guard['open_attempts']==0,'Annotation open attempted')
        if a.stage=='full':
            accepted=checked_audits([*(batch/seq/'audit/audit.json' for seq,batch in accepted.items()),
                                     root/a.sequence/'audit/audit.json'],plan,sources,a.engineering_audit)
            require(remaining==remaining_count(plan,accepted),'Remaining count differs')
            if remaining:
                print(json.dumps({'passed':True,'stage':'full','output':str(root),
                    'audit_path':str(root/a.sequence/'audit/audit.json'),'remaining_reads':remaining,
                    'completed_sequences':sorted(accepted),'label_open_attempts':0}));return
            # A multi-batch run may span days: recheck every causal input before publishing.
            for seq in FRESH:sequence_inputs(plan,seq)
            observations=[];audits={}
            for seq in FRESH:
                batch=accepted[seq]
                audit=batch/seq/'audit/audit.json';capture_file=batch/seq/'capture/receipt.json'
                capture_sha=phase_artifact(batch,f'{seq}_capture',capture_file);rec=read(capture_file)
                require(digest(capture_file)==capture_sha,'Capture receipt changed during aggregate parsing')
                audit_phase_sha=phase_artifact(batch,f'{seq}_audit',audit)
                check=read(audit);require(check.get('passed') is True and
                    digest(audit)==audit_phase_sha and
                    check.get('plan_sha256')==PLAN_SHA and check.get('source_sha256')==sources and
                    check.get('engineering_audit_sha256')==digest(a.engineering_audit) and
                    check.get('auditor_sha256')==digest(FULL_AUDITOR) and
                    check.get('label_guard',{}).get('open_attempts')==0 and
                    check.get('reference_kind')=='source_bound_archived_A' and
                    check.get('reference_receipt_sha256')==digest(Path(plan['prompts'][seq]['baseline_completion_path'])) and
                    check.get('capture_receipt_sha256')==digest(batch/seq/'capture/receipt.json'), 'Full audit differs')
                audits[seq]={'path':str(audit),'sha256':digest(audit)}
                for row in rec['observations']:
                    source_row=plan['source_frames'][seq][row['frame']]
                    require(all(digest(Path(row[key+'_file']))==row[key+'_sha256'] for key in ('feature','snapshot')),
                            'Observed tensor changed after full audit')
                    observations.append({'sequence':seq,'frame':row['frame'],'view':rec['view'],'roi_xyxy':rec['roi_xyxy'],
                        'source_frame_index':source_row['source_frame_index'],'image_sha256':source_row['image_sha256'],
                        'label_metadata_path':rec['archived_frame_metadata']['path'],
                        'label_metadata_sha256':rec['archived_frame_metadata']['sha256'],
                        'label_metadata_locator':row['frame'],
                        'reference_kind':'source_bound_archived_A',
                        'reference_receipt_path':plan['prompts'][seq]['baseline_completion_path'],
                        'reference_receipt_sha256':plan['prompts'][seq]['baseline_completion_sha256'],
                        'capture_receipt_path':str(batch/seq/'capture/receipt.json'),
                        'capture_receipt_sha256':digest(batch/seq/'capture/receipt.json'),
                        'runtime_verification_path':str(batch/seq/'capture/verification/runtime_verification.json'),
                        'runtime_verification_sha256':rec['runtime_verification_sha256'],
                        'sequence_audit_path':str(audit),'sequence_audit_sha256':digest(audit),
                        'feature_file':row['feature_file'],'feature_sha256':row['feature_sha256'],
                        'snapshot_file':row['snapshot_file'],'snapshot_sha256':row['snapshot_sha256']})
            require(len(observations)==143 and {(x['sequence'],x['frame']) for x in observations}==
                    {(x['sequence'],x['frame']) for x in plan['fresh_reads']},'Final capture coverage differs')
            combined=root/'captures.json';combined.write_text(json.dumps({'passed':True,'plan_sha256':PLAN_SHA,
                'scope':'Label-free exploratory baseline observations; target ROI preserved.',
                'existing_manifest_sha256':plan['existing_manifest_sha256'],
                'existing_feature_pipeline_sha256':plan['feature_pipeline_sha256'],
                'label_metadata_note':'Historical frame metadata files contain label hashes and outcomes; only file hashes were checked during capture. A later fold-scoped loader must selectively project permitted label hashes.',
                'sources':sources,'audits':audits,'observations':observations},indent=2)+'\n')
            (root/'completion.json').write_text(json.dumps({'passed':True,'captures_sha256':digest(combined),
                'label_guard':dict(guard)},indent=2)+'\n')
        print(json.dumps({'passed':True,'stage':a.stage,'output':str(root),'label_open_attempts':guard['open_attempts']}))


if __name__=='__main__':main()

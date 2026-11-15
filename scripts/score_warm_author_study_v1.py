#!/usr/bin/env python3
"""Verify all 84 warm trajectories before separately scoring annotation masks."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.crossfit_observer_capture_shared_v1 import archived_frame_metadata as priority_metadata,validate_plan
from scripts.audit_warm_author_feature_access_v1 import audit as audit_feature_access
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import new_output
from scripts.run_warm_author_comparator_v1 import DATA,METHODS,SEQUENCES,frozen_preflight,identity,input_data
from scripts.run_warm_author_study_v1 import OUT,artifact,commands,source_hashes
from scripts.score_additional_author_comparators_v2 import archived_frame_metadata,install_phase_mask_guard
from scripts.score_author_lean_automatic_v1 import read_bound,require,overall_summary
from scripts.score_boundary_video_factorial_v1 import metric,summary,paired_summary
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_warm_author_comparator_gate_v1 import verify_gate

PRIORITY=('seq11','seq20','seq23')


def phase_bound_sha(folder,name):
    path=Path(folder)/'phases.jsonl';before=digest(path)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    require(digest(path)==before,'Phase log changed while reading')
    found=[r for r in rows if r.get('phase')==name and r.get('passed') is True]
    require(len(found)==1 and isinstance(found[0].get('artifact_sha256'),str),'Required phase identity missing')
    return found[0]['artifact_sha256']


def require_viewport_mask(values):
    x0,y0,x1,y1=VIEWPORT
    require(not values[:y0].any() and not values[y1:].any() and not values[:, :x0].any() and
            not values[:,x1:].any(),'Seq23 prediction extends outside its viewport')


def validate_frozen(folder,completed=()):
    """Validate launch, 8 smoke trajectories and 84 full runs without labels."""
    folder=Path(folder).resolve(strict=True);require(folder.is_relative_to(OUT.resolve(strict=True)),'Batch outside experiment root')
    plan=frozen_preflight();manifest_path=folder/'run_manifest.json';manifest_sha=digest(manifest_path)
    launch=read_bound(manifest_path,manifest_sha);phases=commands(folder)
    require(launch.get('sequences')==list(SEQUENCES) and launch.get('methods')==list(METHODS) and
            launch.get('phases')==[n for n,_ in phases] and launch.get('sources')==source_hashes() and
            launch.get('identities')=={m:identity(m,'seq6',plan) for m in METHODS} and
            launch.get('label_guard',{}).get('open_attempts')==0,'Frozen warm launch differs')
    phase_file=folder/'phases.jsonl';phase_sha=digest(phase_file)
    rows=[json.loads(s) for s in phase_file.read_text().splitlines()]
    require(digest(phase_file)==phase_sha,'Phase log changed')
    expected=[n for n,_ in phases if n not in ('verify','score','audit')]+list(completed)
    require([r.get('phase') for r in rows]==expected,'All 84 warm trajectories must complete before labels')
    for row in rows:
        name=row['phase'];require(row.get('passed') is True and row.get('artifact_sha256')==digest(artifact(folder,name)) and
            row.get('log_sha256')==digest(folder/(name+'.log')),'Frozen phase artifact changed')
    phase_map={r['phase']:r['artifact_sha256'] for r in rows};gates={};feature_controls={}
    for method in METHODS:
        path=folder/'gate'/method/'gate.json';gates[method]=verify_gate(path,identity(method,'seq6',plan),plan)
        control_path=folder/'feature_audit'/method/'audit.json'
        saved=read_bound(control_path,phase_map[f'featureaudit_{method}'])
        current=audit_feature_access(method,path,folder/'control'/method,plan)
        require(saved.get('label_guard',{}).get('open_attempts')==0 and
                {k:v for k,v in saved.items() if k!='label_guard'}==current,'Feature-access control differs')
        feature_controls[method]=current
    capture_plan=validate_plan();records={}
    for seq in SEQUENCES:
        frames,start,prompt,_=input_data(seq,plan)
        meta=priority_metadata(capture_plan,seq) if seq in PRIORITY else archived_frame_metadata(seq)
        records[seq]={'prompt_frame':start,'prompt':prompt,'images':[digest(f.image_path) for f in frames],
                      'frame_metadata':meta,'receipts':{}}
        for method in METHODS:
            path=folder/'full'/seq/method/'receipt.json';sha=phase_map[f'full_{seq}_{method}'];rec=read_bound(path,sha)
            require(rec.get('passed') is True and rec.get('full_video') is True and rec.get('method')==method and
                    rec.get('sequence')==seq and rec.get('identity')==identity(method,seq,plan) and
                    rec.get('prompt')==prompt and rec.get('prompt_frame')==start and
                    rec.get('state_sha256')==gates[method]['state_sha256'] and
                    rec.get('gate_sha256')==digest(folder/'gate'/method/'gate.json') and
                    rec.get('noncausal')==(method=='sam2long') and rec.get('label_guard',{}).get('open_attempts')==0 and
                    rec.get('feature_callback_enabled') is True and
                    isinstance(rec.get('prediction_wall_seconds'),(int,float)) and math.isfinite(rec['prediction_wall_seconds']) and
                    rec['prediction_wall_seconds']>0,
                    'Warm full-video receipt differs')
            feature=rec.get('warm_feature_callback') or {}
            require(all(isinstance(feature.get(k),str) and len(feature[k])==64 for k in
                        ('early_sha256','positive_sha256','negative_sha256')) and
                    isinstance(feature.get('cache_before'),list) and isinstance(feature.get('cache_after'),list),
                    'Warm full-video feature-access evidence missing')
            if method=='lean':require(bool(rec.get('positions')),'Lean full video lacks indexed attention evidence')
            if method=='sam2long':
                pathway=rec.get('pathway_audit') or {}
                require(pathway.get('audited_post_prompt_frames')==len(frames)-start-1 and
                        pathway.get('pathways_after_initial_step')==3,'SAM2Long full pathway audit differs')
            pred=rec.get('records',[]);require(len(pred)==len(frames),'Warm video coverage differs')
            for i,(frame,row) in enumerate(zip(frames,pred,strict=True)):
                file=path.parent/f'{i:06d}.png';sha_mask=row['mask_sha256']
                require(row.get('frame')==i and row.get('source_frame_index')==frame.frame_index and
                        row.get('image_sha256')==records[seq]['images'][i] and row.get('post_prompt')==(i>=start) and
                        digest(file)==sha_mask,'Warm frame or prediction differs')
                if i>=start:require(isinstance(row.get('logits_sha256'),str) and len(row['logits_sha256'])==64,
                                    'Warm logits fingerprint missing')
                with Image.open(frame.image_path) as source,Image.open(file) as image:
                    values=np.asarray(image.convert('L'))
                    require(image.format=='PNG' and image.size==source.size and np.isin(values,(0,255)).all() and
                            (i>=start or not values.any()),'Warm prediction geometry differs')
                    if seq=='seq23':require_viewport_mask(values)
                require(digest(file)==sha_mask,'Warm prediction changed during decode')
            records[seq]['receipts'][method]=sha
    return {'manifest_sha256':manifest_sha,'records':records,'feature_controls':feature_controls,
            'scorer_sha256':digest(Path(__file__))}


def score(folder):
    folder=Path(folder).resolve(strict=True);out=new_output(folder/'scores');guard=install_phase_mask_guard(DATA)
    verify=folder/'verification/verification.json';verify_sha=phase_bound_sha(folder,'verify');proof=read_bound(verify,verify_sha)
    require(proof.get('passed') is True and proof.get('label_guard',{}).get('open_attempts')==0 and
            proof.get('scorer_sha256')==digest(Path(__file__)),'Warm verification differs')
    current=validate_frozen(folder,('verify',))
    require(all(proof.get(k)==v for k,v in current.items()) and guard['open_attempts']==0,'Warm frozen inputs changed')
    out.mkdir(parents=True);guard['active']=False;plan=frozen_preflight();all_rows={};summaries={};paired={}
    all_frames={};preprompt={};timing={}
    for seq in SEQUENCES:
        frames,start,_,_=input_data(seq,plan);bound=proof['records'][seq]
        metadata=read_bound(bound['frame_metadata']['path'],bound['frame_metadata']['sha256'])
        receipts={m:read_bound(folder/'full'/seq/m/'receipt.json',bound['receipts'][m]) for m in METHODS}
        timing[seq]={m:{'seconds':receipts[m]['prediction_wall_seconds'],
                        'scope':receipts[m]['timing_scope']} for m in METHODS}
        require(len(metadata)==len(frames),'Warm label metadata coverage differs');rows=[]
        for i,frame in enumerate(frames):
            label_sha=metadata[i]['label_sha256'];image_sha=bound['images'][i]
            require(digest(frame.mask_path)==label_sha and digest(frame.image_path)==image_sha,'Warm image or label changed')
            truth=load_label_mask(frame.mask_path,[1])==1
            require(digest(frame.mask_path)==label_sha,'Warm label changed during decode')
            row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,'initialization_frame':i==start}
            for method in METHODS:
                path=folder/'full'/seq/method/f'{i:06d}.png';sha=receipts[method]['records'][i]['mask_sha256']
                require(digest(path)==sha,'Warm prediction changed')
                with Image.open(path) as image:mask=np.asarray(image.convert('L'))>0
                require(digest(path)==sha,'Warm prediction changed during decode');row[method]=metric(mask,truth)
                if seq=='seq23':
                    x0,y0,x1,y1=VIEWPORT;row[method+'_viewport']=metric(mask[y0:y1,x0:x1],truth[y0:y1,x0:x1])
            require(digest(frame.image_path)==image_sha,'Warm source image changed during scoring');rows.append(row)
        all_rows[seq]=rows;summaries[seq]={m:summary(rows,m) for m in METHODS}
        all_frames[seq]={m:summary(rows,m,post_only=False) for m in METHODS}
        paired[seq]={m:paired_summary(rows,m,'native') for m in METHODS if m!='native'}
        preprompt[seq]={'frames':start,'labeled_present':sum(r['native']['present'] for r in rows[:start]),
                        'present_misses':{m:sum(r[m]['present'] and r[m]['dice']==0 for r in rows[:start]) for m in METHODS}}
    overall={m:overall_summary({s:summaries[s][m] for s in SEQUENCES}) for m in METHODS}
    seq23_viewport={m:summary(all_rows['seq23'],m+'_viewport') for m in METHODS}
    lines=['# Warm-matched author comparator study','',
           'All methods use the archived box, points, feature-access operation and saved mask; matching inputs do not establish identical state. '
           'SAM2Long is offline; author Lean includes MPS compatibility corrections. Previously inspected patients; exploratory only.','',
           '| Sequence | Present | Native | Author Lean | SAMURAI | SAM2Long (offline) |','|---|---:|---:|---:|---:|---:|']
    for seq in SEQUENCES:
        row=summaries[seq];values=[row[m]['mean_present_dice'] for m in METHODS]
        lines.append(f"| {seq} | {row['native']['present_frames']} | "+' | '.join('N/A' if x is None else f'{x:.4f}' for x in values)+' |')
    lines += ['', '| Method | Patient macro Dice (patients) | Frame-weighted Dice (present frames) | Empty alarms / frames |',
              '|---|---:|---:|---:|']
    for m in METHODS:
        x=overall[m];lines.append(f"| {m} | {x['patient_macro_present_dice']} ({x['patients_with_present_frames']}) | "
            f"{x['frame_weighted_present_dice']} ({x['present_frames']}) | {x['labeled_empty_false_alarms']}/{x['labeled_empty_frames']} |")
    lines += ['', '| Sequence | Pre-prompt present / frames | Labeled-empty alarms: native / Lean / SAMURAI / SAM2Long |',
              '|---|---:|---|']
    for seq in SEQUENCES:
        x=preprompt[seq];alarms=[f"{summaries[seq][m]['false_alarms']}/{summaries[seq][m]['empty_frames']}" for m in METHODS]
        lines.append(f"| {seq} | {x['labeled_present']}/{x['frames']} | "+' / '.join(alarms)+' |')
    lines += ['', 'Post-prompt excludes initialization. Seq23 uses a retrospective crop; seq1/7 have no initialization. '
              'Labeled-empty does not establish clinical absence. Seq23 viewport-only results, per-frame precision, recall, '
              'boundary scores, predicted area and pre-prompt misses are in results.json. Timing includes staging, prompt, '
              'feature telemetry, propagation, synchronization and output writes, excludes model loading, and is not a '
              'real-time throughput benchmark; SAM2Long uses offline final paths.']
    lines += ['', 'The separate seq6 feature-access controls report output differences without an equality requirement; '
              'they do not establish equivalence of all internal state.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    result={'passed':True,'verification_sha256':verify_sha,'scorer_sha256':digest(Path(__file__)),
            'report_sha256':digest(out/'report.md'),'prelabel_guard':dict(guard),'rows':all_rows,
            'summaries':summaries,'all_frame_summaries':all_frames,'paired_descriptive':paired,
            'preprompt':preprompt,'overall':overall,'seq23_viewport_summaries':seq23_viewport,'timing':timing,
            'feature_controls':proof['feature_controls']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequences':len(all_rows)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=('verify','score'),required=True);a=p.parse_args()
    if a.mode=='score':score(a.run);return
    guard=install_mask_open_guard(DATA);result=validate_frozen(a.run)
    require(guard['open_attempts']==0,'Annotation open during warm verification')
    result.update(passed=True,label_guard=dict(guard));out=new_output(a.run/'verification');out.mkdir(parents=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequences':len(result['records'])}))


if __name__=='__main__':main()

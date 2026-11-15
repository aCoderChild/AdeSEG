#!/usr/bin/env python3
"""Label-free feasibility preflight for the exploratory common warm observer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.crossfit_observer_capture_shared_v1 import (BASE,DATA,EXISTING,FRESH,PLAN_SHA,SHORT,
    read,require,sequence_inputs,validate_plan)

SOURCES=('scripts/preflight_crossfit_warm_observer_v3.py',
         'protocols/EXPLORATORY_CROSSFIT_WARM_OBSERVER_V1.md',
         'scripts/run_source_group_capture_v1.py','scripts/audit_source_group_capture_v1.py',
         'scripts/run_expanded_train_capture_v1.py','scripts/audit_expanded_train_capture_v1.py',
         'scripts/source_group_refiner_shared_v2.py','modeling/lean_mps_decoder_observer.py',
         'modeling/lean_multiscale_anchor.py','modeling/token_memory_window.py',
         'modeling/lean_token_window_storage.py','scripts/resume_source_group_capture_v1.py')


def checked_prompt(plan,seq):
    frames,records,start,grid,mask,baseline,baseline_record,prompt=sequence_inputs(plan,seq)
    provenance=prompt['prompt_provenance'];path=Path(provenance['prompt_file']).parent/'prompt.json'
    sha=provenance['prompt_metadata_sha256'];require(digest(path)==sha,'Prompt metadata changed')
    rec=read(path);require(digest(path)==sha and rec['first_detection_index']==start,'Prompt metadata changed during read')
    box=rec['box_viewport_xyxy'] if seq=='seq23' else rec['box']
    points=rec['points_viewport'] if seq=='seq23' else rec['points']
    box=np.asarray(box,dtype=np.float64);points=np.asarray(points,dtype=np.float64)
    require(box.shape==(4,) and points.shape==(2,2) and np.isfinite(box).all() and np.isfinite(points).all(),
            'Archived box/point format differs')
    height,width=mask.shape
    require(0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height and
            bool(((points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)).all()),
            'Archived prompt coordinates are outside the view')
    return {'sequence':seq,'initial_frame':start,'target_frames':grid,'view':prompt['view'],
        'roi_xyxy':provenance['roi_xyxy'],'frame_count':len(frames),'prompt_metadata_path':str(path),
        'prompt_metadata_sha256':sha,'prompt_mask_path':provenance['prompt_file'],
        'prompt_mask_sha256':provenance['prompt_mask_sha256'],
        'box_field':'box_viewport_xyxy' if seq=='seq23' else 'box',
        'points_field':'points_viewport' if seq=='seq23' else 'points',
        'historical_direct_A_completion_sha256':prompt['baseline_completion_sha256']}


def verify_seq6_identity(common,old6,audit,fresh_targets,runner_sha,auditor_sha):
    identity=old6['identity']
    require(set(identity)==set(common)-{'runtime_environment','source_freeze_sha256'} and
            all(identity[k]==common[k] for k in identity if k not in {'runner_sha256','preflight_sha256'}) and
            identity['runner_sha256']==runner_sha and audit['auditor_sha256']==auditor_sha and
            all(i in old6['targets'] and i in fresh_targets for i in (24,90)),
            'Historical seq6 shared identity, auditor or overlap differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);plan=validate_plan();existing,manifest_sha=load_prepared(EXISTING)
    require(manifest_sha==plan['existing_manifest_sha256'] and len(existing['samples'])==214,'Existing manifest differs')
    sequences={};identities=[]
    for seq,evidence in existing['sequences'].items():
        cap=Path(evidence['capture_receipt_path']);ref=Path(evidence['reference_receipt_path']);audit=Path(evidence['audit_path'])
        value=read(cap);identity=value['identity'];identities.append(identity)
        require(value['sequence']==seq and value['full_sequence'] is True and value['mode']=='capture' and
                value['label_guard']['open_attempts']==0,'Historical capture differs')
        for key,path in (('runner_sha256','scripts/run_source_group_capture_v1.py'),
                         ('observer_sha256','modeling/lean_mps_decoder_observer.py'),
                         ('anchor_sha256','modeling/lean_multiscale_anchor.py'),
                         ('token_policy_sha256','modeling/token_memory_window.py'),
                         ('token_runtime_sha256','modeling/lean_token_window_storage.py')):
            require(identity[key]==digest(ROOT/path),f'Historical identity source changed: {key}')
        sequences[seq]={'capture_receipt_path':str(cap),'capture_receipt_sha256':digest(cap),
            'reference_receipt_path':str(ref),'reference_receipt_sha256':digest(ref),
            'audit_path':str(audit),'audit_sha256':digest(audit),'target_frames':value['targets']}
    require(len(identities)==9 and all(x==identities[0] for x in identities),'Historical runtime identities differ')
    fresh={seq:checked_prompt(plan,seq) for seq in FRESH}
    require(sum(len(x['target_frames']) for x in fresh.values())==143 and guard['open_attempts']==0,'Fresh coverage or label guard differs')
    seq6_reference=Path(BASE/'contrast_refinement_v1/expanded_train_seq6_reference_attempt01/receipt.json')
    seq6_capture=seq6_reference.parent.parent/'expanded_train_seq6_capture_attempt01/receipt.json'
    seq6_audit=seq6_reference.parent.parent/'expanded_train_seq6_verification_attempt01/verification.json'
    old6=read(seq6_capture);v6=read(seq6_audit)
    verify_seq6_identity(identities[0],old6,v6,fresh['seq6']['target_frames'],
                         digest(ROOT/'scripts/run_expanded_train_capture_v1.py'),
                         digest(ROOT/'scripts/audit_expanded_train_capture_v1.py'))
    require(v6['passed'] and v6['capture_receipt_sha256']==digest(seq6_capture) and
            v6['reference_receipt_sha256']==digest(seq6_reference) and
            old6['identity']['observer_sha256']==identities[0]['observer_sha256'] and
            old6['identity']['model_state_sha256']==identities[0]['model_state_sha256'] and
            all(i in old6['targets'] and i in fresh['seq6']['target_frames'] for i in (24,90)),'Historical seq6 anchor differs')
    result={'passed':True,'scope':'Read-only warm-prompt feasibility; no labels, MPS inference, or runtime equivalence established.',
        'capture_authorized':False,'frozen_index_plan_sha256':PLAN_SHA,'existing_manifest_sha256':manifest_sha,
        'old_identity':identities[0],'old_sequences':sequences,'fresh_prompts':fresh,
        'seq6_historical':{'reference_path':str(seq6_reference),'reference_sha256':digest(seq6_reference),
           'capture_path':str(seq6_capture),'capture_sha256':digest(seq6_capture),
           'audit_path':str(seq6_audit),'audit_sha256':digest(seq6_audit),'overlap_frames':[24,90]},
        'seq23_coordinates':'viewport-local','existing_reads':214,'fresh_reads':143,
        'source_sha256':{p:digest(ROOT/p) for p in SOURCES},'label_guard':dict(guard)}
    out.mkdir(parents=True,exist_ok=False);path=out/'plan.json';path.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'plan_sha256':digest(path),'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'label_open_attempts':0,'capture_authorized':False}))

if __name__=='__main__':main()

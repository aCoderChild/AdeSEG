#!/usr/bin/env python3
"""Verify extended learned-refiner A/B/C/D short runs before full videos.

No labels are read. Seq8 and seq15 each require A/A-repeat/B/C/D.
This is an engineering check, not an efficacy or selection screen.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.preflight_training_exposed_boundary_video_v1 import load_training_exposed_preflight, image_metadata_only, DETECTOR_SHA
from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate

ARMS=('A','A_repeat','B','C','D')
SEQS=('seq2','seq15')


def read(path): return json.loads(Path(path).read_text())


def verify_short_run(folder, seq, arm, *, training_exposed_preflight_sha, engineering_gate_sha):
    folder=Path(folder).resolve(); rec=read(folder/'completion.json')
    if (not rec.get('passed') or rec.get('sequence') != seq or rec.get('arm') != ('A' if arm=='A_repeat' else arm)
            or rec.get('full_video') is not False or rec.get('limit') != 4
            or rec.get('runner_source_sha256') != digest(ROOT/'scripts/run_training_exposed_boundary_refiner_v1.py')
            or rec.get('training_exposed_preflight_sha256') != training_exposed_preflight_sha
            or rec.get('engineering_gate_sha256') != engineering_gate_sha
            or rec.get('extension_gate_sha256') is not None or rec.get('extension_audit_sha256') is not None
            or rec.get('label_guard',{}).get('open_attempts') != 0):
        raise ValueError(f'{seq}/{arm}: not a valid short run')
    masks=rec['masks']; count=len(masks); start=rec['prompt_frame']
    if count != start+5 or [x['frame'] for x in masks] != list(range(count)):
        raise ValueError('Short-run frame coverage differs')
    for row in masks:
        if digest(folder/f"{row['frame']:06d}.png") != row['sha256']:
            raise ValueError('Short-run mask changed')
    for field,name in (('runtime_verification_sha256','verification/runtime_verification.json'),
                       ('memory_events_sha256','memory_events.json'),('positions_sha256','positions.json'),
                       ('selections_sha256','selections.json'),('telemetry_sha256','telemetry.json')):
        if digest(folder/name) != rec[field]: raise ValueError(f'Short-run {name} changed')
    runtime=read(folder/'verification/runtime_verification.json')
    if not runtime.get('passed') or runtime.get('candidate_checkpoint_sha256') != rec['candidate_checkpoint_sha256']:
        raise ValueError('Short-run runtime identity differs')
    for name in ('model','refiner'):
        if digest(folder/f'verification/{name}_tensor_fingerprints.json') != runtime[name+'_tensor_fingerprints_sha256']:
            raise ValueError('Short-run model fingerprint differs')
    rows=rec['post_prompt_state']; telemetry=read(folder/'telemetry.json')
    if [x['frame'] for x in rows] != list(range(start,count)):
        raise ValueError('Short-run state coverage differs')
    t=[x for x in telemetry if not x['conditioning']]
    if [x['frame'] for x in t] != list(range(start+1,count)):
        raise ValueError('Short-run telemetry coverage differs')
    return {'folder':str(folder),'completion_sha256':digest(folder/'completion.json'),
            'runtime_sha256':digest(folder/'verification/runtime_verification.json'),
            'rec':rec,'runtime':runtime,'telemetry':t}


def verify_mapping(mapping, preflight, engineering_gate):
    if set(mapping) != set(SEQS) or any(set(mapping[s]) != set(ARMS) for s in SEQS):
        raise ValueError('Two full short-run factorials and baseline repeats are required')
    paths=[Path(mapping[s][a]).resolve() for s in SEQS for a in ARMS]
    if len(set(paths)) != len(paths): raise ValueError('Short-run paths are not distinct')
    preflight_sha=digest(Path(preflight)/'manifest.json'); engineering_sha=digest(engineering_gate)
    evidence={s:{a:verify_short_run(mapping[s][a],s,a,training_exposed_preflight_sha=preflight_sha,
                                   engineering_gate_sha=engineering_sha) for a in ARMS} for s in SEQS}
    pre=read(Path(preflight)/'manifest.json')
    for seq,items in evidence.items():
        base,repeat=items['A'],items['A_repeat']; br,rr=base['rec'],repeat['rec']
        item=pre['sequences'][seq]
        indices, hashes = image_metadata_only(Path(item['source'])/seq/'frames.json')
        expected_images=[{'frame':i,'source_frame_index':index,'sha256':sha} for i,(index,sha) in enumerate(zip(indices,hashes,strict=True))]
        if (hashes != item['image_sha256'] or br['prompt_frame'] != item['prompt_frame'] or
                br['prompt_sha256'] != item['archived_prompt_sha256'] or
                br['prompt_mask_sha256'] != item['archived_prompt_mask_sha256'] or
                br['checkpoint_sha256'] != pre['checkpoint_sha256'] or br['detector_sha256'] != DETECTOR_SHA or
                br['view'] != {'kind':'full_source_frame'} or br['input_images'] != expected_images):
            raise ValueError(f'{seq}: extended prompt or images differ')
        if (br['masks'] != rr['masks'] or br['post_prompt_state'] != rr['post_prompt_state']
                or any(read(Path(base['folder'])/f) != read(Path(repeat['folder'])/f)
                       for f in ('memory_events.json','positions.json','selections.json'))):
            raise ValueError(f'{seq}: fresh baseline repeats differ')
        for arm in ARMS[1:]:
            x=items[arm]
            if x['runtime'] != base['runtime'] or any(x['rec'][k] != br[k] for k in
                  ('prompt_frame','prompt_sha256','prompt_mask_sha256','checkpoint_sha256','detector_sha256',
                   'candidate_checkpoint_sha256','control_checkpoint_sha256','view','input_images')):
                raise ValueError(f'{seq}/{arm}: runtime, inputs or prompt differ')
        ar=br['post_prompt_state'][1]; b=items['B']['rec']['post_prompt_state'][1]
        c=items['C']['rec']['post_prompt_state'][1]; d=items['D']['rec']['post_prompt_state'][1]
        if (ar['output_logits_sha256'] != c['output_logits_sha256'] or
                b['output_logits_sha256'] != d['output_logits_sha256'] or
                ar['output_logits_sha256'] == b['output_logits_sha256'] or
                c['maskmem_features_sha256'] != d['maskmem_features_sha256'] or
                c['maskmem_features_sha256'] == ar['maskmem_features_sha256']):
            raise ValueError(f'{seq}: learned first-frame output/memory routing not demonstrated')
        for arm in ('B','C','D'):
            initial=items[arm]['rec']['post_prompt_state'][0]
            expected_initial=br['post_prompt_state'][0]
            if any(initial[k] != expected_initial[k] for k in ('mask_sha256','pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256')):
                raise ValueError(f'{seq}/{arm}: conditioning state differs')
            row=items[arm]['rec']['post_prompt_state'][1]
            if any(row[k] != ar[k] for k in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256')):
                raise ValueError(f'{seq}/{arm}: native decoder/pointer/presence changed')
        b_rec=items['B']['rec'];
        for x,y in zip(b_rec['post_prompt_state'],br['post_prompt_state'],strict=True):
            if any(x[k] != y[k] for k in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256')):
                raise ValueError('B altered baseline compact memory state')
        if any(read(Path(items['B']['folder'])/f) != read(Path(base['folder'])/f)
               for f in ('memory_events.json','positions.json','selections.json')):
            raise ValueError('B altered memory/position/selection traces')
    return evidence


def build_receipt(mapping, *, training_exposed_preflight, engineering_gate):
    evidence=verify_mapping(mapping, training_exposed_preflight, engineering_gate)
    first=evidence['seq2']['A']['runtime']
    if evidence['seq15']['A']['runtime'] != first: raise ValueError('Runtime differs between sequences')
    original=read(engineering_gate)
    if not original.get('passed') or original['runtime_identity'] != first:
        raise ValueError('Extended and original runtime identities differ')
    return {'passed':True,'scope':'Real boundary-checkpoint short routing engineering only; no efficacy or promotion claim.',
            'runner_sha256':digest(ROOT/'scripts/run_training_exposed_boundary_refiner_v1.py'),
            'routing_sha256':digest(ROOT/'modeling/lean_boundary_refinement_routing_v1.py'),
            'runtime_sha256':digest(ROOT/'modeling/lean_boundary_window_runtime_v1.py'),
            'preflight_source_sha256':digest(ROOT/'scripts/preflight_training_exposed_boundary_video_v1.py'),
            'archived_loader_sha256':digest(ROOT/'scripts/preflight_author_trainval_v1.py'),
            'adapter_dependency_sha256': {str(p):digest(ROOT/p) for p in (
                'scripts/preflight_remaining_boundary_video_v1.py', 'scripts/preflight_extended_boundary_video_v1.py',
                'modeling/samurai_auto_prompt.py', 'scripts/source_group_label_guard_v1.py')},
            'training_exposed_preflight_sha256':digest(Path(training_exposed_preflight)/'manifest.json'),
            'engineering_gate_sha256':digest(engineering_gate),
            'verifier_sha256':digest(Path(__file__)), 'runtime_identity':first,
            'runs':{s:{a:{k:v for k,v in evidence[s][a].items() if k in ('folder','completion_sha256','runtime_sha256')}
                       for a in ARMS} for s in SEQS}}


def require_gate(path, *, training_exposed_preflight, engineering_gate):
    gate=read(path)
    if not gate.get('passed') or gate.get('verifier_sha256') != digest(Path(__file__)):
        raise ValueError('Short gate is incomplete or uses a different verifier')
    mapping={s:{a:gate['runs'][s][a]['folder'] for a in ARMS} for s in SEQS}
    if build_receipt(mapping,training_exposed_preflight=training_exposed_preflight,engineering_gate=engineering_gate) != gate:
        raise ValueError('Short gate or bound artifacts changed')
    return gate


def verify_extended_full_gate(path, *, training_exposed_preflight, detector, engineering_gate,
                              study, audit, checkpoint, split_preflight, baseline_reference, seq4_capture):
    """Recheck both short gates and the frozen extended input identity."""
    load_training_exposed_preflight(training_exposed_preflight, checkpoint, detector)
    verify_full_gate(engineering_gate, study=study, audit=audit, checkpoint=checkpoint,
                     split_preflight=split_preflight, baseline_reference=baseline_reference, seq4_capture=seq4_capture)
    receipt = require_gate(path, training_exposed_preflight=training_exposed_preflight,engineering_gate=engineering_gate)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--training-exposed-preflight',type=Path,required=True)
    p.add_argument('--engineering-gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    gate=build_receipt(read(a.mapping),training_exposed_preflight=a.training_exposed_preflight,engineering_gate=a.engineering_gate); out=validate_output(a.output)
    from scripts.resume_source_group_capture_v1 import BASE
    if not out.is_relative_to(BASE.resolve()): raise ValueError('Gate outside requested experiment folder')
    out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({'passed':True,'short_runs':10,'sequences':list(SEQS)}))


if __name__=='__main__': main()

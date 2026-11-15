#!/usr/bin/env python3
"""Verify real learned-refiner A/B/C/D short runs before full videos.

No labels are read. Seq4 and cropped seq23 each require A/A-repeat/B/C/D.
This is an engineering check, not an efficacy or selection screen.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.resume_source_group_capture_v1 import BASE

ARMS=('A','A_repeat','B','C','D')
SEQS=('seq4','seq23')


def read(path): return json.loads(Path(path).read_text())


def verify_short_run(folder, seq, arm, seed):
    folder=Path(folder).resolve(); rec=read(folder/'completion.json')
    if (not rec.get('passed') or rec.get('seed') != seed or rec.get('sequence') != seq or rec.get('arm') != ('A' if arm=='A_repeat' else arm)
            or rec.get('full_video') is not False or rec.get('limit') != 4
            or rec.get('runner_source_sha256') != digest(ROOT/'scripts/run_multiseed_boundary_refiner_v1.py')
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


def verify_mapping(mapping, seed):
    if seed not in (20261125, 20261135): raise ValueError('Unexpected study seed')
    if set(mapping) != set(SEQS) or any(set(mapping[s]) != set(ARMS) for s in SEQS):
        raise ValueError('Two full short-run factorials and baseline repeats are required')
    paths=[Path(mapping[s][a]).resolve() for s in SEQS for a in ARMS]
    if len(set(paths)) != len(paths): raise ValueError('Short-run paths are not distinct')
    evidence={s:{a:verify_short_run(mapping[s][a],s,a,seed) for a in ARMS} for s in SEQS}
    for seq,items in evidence.items():
        base,repeat=items['A'],items['A_repeat']; br,rr=base['rec'],repeat['rec']
        if (br['masks'] != rr['masks'] or br['post_prompt_state'] != rr['post_prompt_state']
                or any(read(Path(base['folder'])/f) != read(Path(repeat['folder'])/f)
                       for f in ('memory_events.json','positions.json','selections.json'))):
            raise ValueError(f'{seq}: fresh baseline repeats differ')
        for arm in ARMS[1:]:
            x=items[arm]
            if x['runtime'] != base['runtime'] or any(x['rec'][k] != br[k] for k in
                  ('prompt_frame','prompt_sha256','prompt_mask_sha256','checkpoint_sha256','detector_sha256',
                   'candidate_checkpoint_sha256','control_checkpoint_sha256','view','input_images',
                   'seq8_preflight_sha256','seq8_preflight_completion_sha256')):
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


def build_receipt(mapping, seed):
    evidence=verify_mapping(mapping,seed)
    first=evidence['seq4']['A']['runtime']
    if evidence['seq23']['A']['runtime'] != first: raise ValueError('Runtime differs between sequences')
    seq4=evidence['seq4']['A']['rec'];seq23=evidence['seq23']['A']['rec']
    for key in ('detector_sha256','seq8_preflight_sha256','seq8_preflight_completion_sha256'):
        if seq4.get(key) != seq23.get(key) or not isinstance(seq4.get(key),str) or len(seq4[key]) != 64:
            raise ValueError(f'Short sequence input binding differs: {key}')
    return {'passed':True,'seed':seed,'scope':'Real boundary-checkpoint short routing engineering only; no efficacy or promotion claim.',
            'runner_sha256':digest(ROOT/'scripts/run_multiseed_boundary_refiner_v1.py'),
            'routing_sha256':digest(ROOT/'modeling/lean_boundary_refinement_routing_v1.py'),
            'runtime_sha256':digest(ROOT/'modeling/lean_boundary_multiseed_window_runtime_v1.py'),
            'verifier_sha256':digest(Path(__file__)), 'runtime_identity':first,
            'detector_sha256':seq4['detector_sha256'],'seq8_preflight_sha256':seq4['seq8_preflight_sha256'],
            'seq8_preflight_completion_sha256':seq4['seq8_preflight_completion_sha256'],
            'runs':{s:{a:{k:v for k,v in evidence[s][a].items() if k in ('folder','completion_sha256','runtime_sha256')}
                       for a in ARMS} for s in SEQS}}


def require_gate(path, seed):
    gate=read(path)
    if not gate.get('passed') or gate.get('seed') != seed or gate.get('verifier_sha256') != digest(Path(__file__)):
        raise ValueError('Short gate is incomplete or uses a different verifier')
    mapping={s:{a:gate['runs'][s][a]['folder'] for a in ARMS} for s in SEQS}
    if build_receipt(mapping,seed) != gate:
        raise ValueError('Short gate or bound artifacts changed')
    return gate


def verify_full_gate(path, *, multiseed_root, seed, checkpoint, detector, seq8_preflight, split_preflight, baseline_reference, seq4_capture):
    """Recheck all artifacts plus the exact inputs before creating a full run."""
    receipt = require_gate(path,seed)
    if (receipt.get('detector_sha256') != digest(detector) or
            receipt.get('seq8_preflight_sha256') != digest(Path(seq8_preflight)/'manifest.json') or
            receipt.get('seq8_preflight_completion_sha256') != digest(Path(seq8_preflight)/'completion.json')):
        raise ValueError('Detector or seq8 preflight differs from the short gate')
    runtime = receipt['runtime_identity']
    study = Path(multiseed_root) / str(seed)
    expected = {'multiseed_root': str(Path(multiseed_root).resolve()), 'seed': seed,
                'checkpoint_sha256': digest(checkpoint),
                'split_preflight_sha256': digest(Path(split_preflight)/'manifest.json'),
                'baseline_reference_sha256': digest(Path(baseline_reference)/'receipt.json'),
                'seq4_capture_sha256': digest(Path(seq4_capture)/'channel_feature_receipt.json'),
                'candidate_checkpoint_sha256': digest(study/'runs/region_boundary/checkpoint.pt'),
                'control_checkpoint_sha256': digest(study/'runs/mask_only_boundary/checkpoint.pt')}
    if any(runtime.get(k) != v for k,v in expected.items()):
        raise ValueError('Full-run inputs differ from the short gate')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(20261125,20261135),required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    gate=build_receipt(read(a.mapping),a.seed); out=validate_output(a.output)
    if not out.is_relative_to(BASE.resolve()): raise ValueError('Gate outside requested experiment folder')
    out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({'passed':True,'short_runs':10,'sequences':list(SEQS)}))


if __name__=='__main__': main()

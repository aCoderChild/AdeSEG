#!/usr/bin/env python3
"""Independently audit the extended label-free short A/B/C/D gate.

This CPU-only verifier reads saved receipts, traces and predicted PNGs. It does
not call the gate's implementation, load target labels or run a model. Passing
supports only the recorded short-clip routing checks, not video efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output

ARMS = ('A', 'A_repeat', 'B', 'C', 'D')
SEQUENCES = ('seq8', 'seq17')
STATE = ('pred_masks_sha256', 'maskmem_features_sha256', 'obj_ptr_sha256',
         'object_score_logits_sha256')
FILES = (('memory_events_sha256', 'memory_events.json'), ('positions_sha256', 'positions.json'),
         ('selections_sha256', 'selections.json'), ('telemetry_sha256', 'telemetry.json'),
         ('runtime_verification_sha256', 'verification/runtime_verification.json'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(gate_path, preflight, engineering_gate, checkpoint, detector):
    gate_path = Path(gate_path).resolve(strict=True)
    preflight, engineering_gate = Path(preflight).resolve(strict=True), Path(engineering_gate).resolve(strict=True)
    gate = read(gate_path)
    require(gate.get('passed') is True, 'Short gate failed')
    for field, file in (('runner_sha256', 'scripts/run_extended_boundary_refiner_v1.py'),
                        ('routing_sha256', 'modeling/lean_boundary_refinement_routing_v1.py'),
                        ('runtime_sha256', 'modeling/lean_boundary_window_runtime_v1.py'),
                        ('verifier_sha256', 'scripts/verify_extended_boundary_video_short_gate_v1.py'),
                        ('preflight_source_sha256', 'scripts/preflight_extended_boundary_video_v1.py')):
        require(gate.get(field) == sha(ROOT / file), f'Gate source differs: {field}')
    from scripts.preflight_extended_boundary_video_v1 import load_extended_preflight
    from scripts.resume_source_group_capture_v1 import BASE
    pre = load_extended_preflight(preflight,checkpoint,detector)
    original = read(engineering_gate)
    require(preflight.is_relative_to(BASE.resolve(strict=True)) and
            gate.get('extended_preflight_sha256') == sha(preflight/'manifest.json') and
            gate.get('engineering_gate_sha256') == sha(engineering_gate) and
            original.get('passed') is True and gate.get('runtime_identity') == original.get('runtime_identity'),
            'Preflight/original gate differs')
    require(set(gate['runs']) == set(SEQUENCES), 'Short-gate sequence coverage differs')
    expected_runtime = gate['runtime_identity']
    evidence, seen = {}, set()
    for seq in SEQUENCES:
        require(set(gate['runs'][seq]) == set(ARMS), 'Short-gate arm coverage differs')
        item=pre['sequences'][seq]
        require(sha(item['prompt_metadata_path']) == item['prompt_metadata_sha256'] and
                sha(item['prompt_mask_path']) == item['prompt_mask_sha256'] and
                all(sha(r['image_path']) == r['image_sha256'] for r in item['frames']),
                'Frozen extension prompt or source image changed')
        entries = {}
        for arm in ARMS:
            entry = gate['runs'][seq][arm]
            folder = Path(entry['folder']).resolve(strict=True)
            require(folder not in seen, 'Reused output directory')
            seen.add(folder)
            require(sha(folder / 'completion.json') == entry['completion_sha256'], 'Completion changed')
            rec = read(folder / 'completion.json')
            expected_images=[{'frame':r['frame'],'source_frame_index':r['source_frame_index'],'sha256':r['image_sha256']}
                             for r in item['frames']]
            require(rec.get('passed') is True and rec.get('sequence') == seq and
                    rec.get('arm') == ('A' if arm == 'A_repeat' else arm) and
                    rec.get('limit') == 4 and rec.get('full_video') is False and
                    rec.get('label_guard', {}).get('open_attempts') == 0 and
                    rec.get('extended_preflight_sha256') == sha(preflight/'manifest.json') and
                    rec.get('engineering_gate_sha256') == sha(engineering_gate) and
                    rec.get('extension_gate_sha256') is None and rec.get('extension_audit_sha256') is None and
                    rec.get('prompt_frame') == item['first_detection_index'] and
                    rec.get('prompt_sha256') == item['prompt_metadata_sha256'] and
                    rec.get('prompt_mask_sha256') == item['prompt_mask_sha256'] and
                    rec.get('view') == item['view'] and rec.get('input_images') == expected_images and
                    rec.get('checkpoint_sha256') == sha(checkpoint) and rec.get('detector_sha256') == sha(detector),
                    'Short-run scope or label guard differs')
            start = rec['prompt_frame']
            require([r['frame'] for r in rec['masks']] == list(range(start + 5)) and
                    [r['frame'] for r in rec['post_prompt_state']] == list(range(start, start + 5)),
                    'Frame coverage differs')
            require(rec['runner_source_sha256'] == gate['runner_sha256'], 'Run source differs')
            for field, name in FILES:
                require(sha(folder / name) == rec[field], f'{name} changed')
            runtime = read(folder / 'verification/runtime_verification.json')
            require(runtime == expected_runtime and sha(folder / 'verification/runtime_verification.json') == entry['runtime_sha256'],
                    'Runtime identity differs')
            require(runtime.get('candidate_checkpoint_sha256') == rec.get('candidate_checkpoint_sha256'), 'Candidate mismatch')
            for name in ('model', 'refiner'):
                require(sha(folder / f'verification/{name}_tensor_fingerprints.json') ==
                        runtime[name + '_tensor_fingerprints_sha256'], 'Tensor fingerprints changed')
            for row in rec['masks']:
                file = folder / f"{row['frame']:06d}.png"
                require(sha(file) == row['sha256'], 'Saved mask hash differs')
                with Image.open(file) as im:
                    require(im.format == 'PNG' and im.size == tuple(item['image_size_wh']) and
                            set(np.unique(np.asarray(im.convert('L')))).issubset({0,255}),
                            'Saved mask is not binary PNG')
            later = [x for x in read(folder / 'telemetry.json') if not x['conditioning']]
            require([r['frame'] for r in later] == list(range(start + 1, start + 5)),
                    'Post-prompt telemetry differs')
            entries[arm] = {'folder': folder, 'rec': rec, 'telemetry': later}
        a, repeat, b, c, d = (entries[x] for x in ARMS)
        ar, rr = a['rec'], repeat['rec']
        require(ar['masks'] == rr['masks'] and ar['post_prompt_state'] == rr['post_prompt_state'],
                'Independent baseline runs differ')
        for name in ('memory_events.json','positions.json','selections.json'):
            base = read(a['folder'] / name)
            require(base == read(repeat['folder'] / name) == read(b['folder'] / name),
                    'A/A-repeat/B memory trajectory differs')
        for arm in ('A_repeat','B','C','D'):
            r = entries[arm]['rec']
            for key in ('prompt_frame','prompt_sha256','prompt_mask_sha256','checkpoint_sha256',
                        'detector_sha256','candidate_checkpoint_sha256','view','input_images'):
                require(r[key] == ar[key], f'{seq}/{arm}: matched input differs')
            for key in ('mask_sha256', *STATE):
                require(r['post_prompt_state'][0][key] == ar['post_prompt_state'][0][key],
                        f'{seq}/{arm}: conditioning differs')
        for actual, original in zip(b['rec']['post_prompt_state'], ar['post_prompt_state'], strict=True):
            require(all(actual[k] == original[k] for k in STATE), 'B changed the compact native state')
        first = {arm: entries[arm]['rec']['post_prompt_state'][1] for arm in ('A','B','C','D')}
        require(first['A']['output_logits_sha256'] == first['C']['output_logits_sha256'] and
                first['B']['output_logits_sha256'] == first['D']['output_logits_sha256'] and
                first['A']['output_logits_sha256'] != first['B']['output_logits_sha256'] and
                first['C']['maskmem_features_sha256'] == first['D']['maskmem_features_sha256'] and
                first['C']['maskmem_features_sha256'] != first['A']['maskmem_features_sha256'],
                'Learned first-frame routing was not demonstrated')
        for arm in ('B','C','D'):
            require(all(first[arm][k] == first['A'][k] for k in
                        ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256')),
                    'First native decoder state differs')
        evidence[seq] = {arm: {'completion_sha256': gate['runs'][seq][arm]['completion_sha256'],
                               'post_prompt_frames': len(entries[arm]['telemetry']),
                               'native_absence_frames': sum(not x['native_object_present'] for x in entries[arm]['telemetry']),
                               'native_absence_refiner_positive_frames': sum(not x['native_object_present'] and x['refined']['positive_pixels'] > 0 for x in entries[arm]['telemetry'])}
                         for arm in ARMS}
    return {'passed': True, 'scope': 'Independent extension short-run routing/artifact audit only; no labels or efficacy.',
            'extension_gate_sha256': sha(gate_path), 'auditor_sha256': sha(Path(__file__)),
            'extended_preflight_sha256': sha(preflight/'manifest.json'),
            'engineering_gate_sha256': sha(engineering_gate),
            'runner_sha256': gate['runner_sha256'], 'runs_verified': len(seen),
            'candidate_checkpoint_sha256': expected_runtime['candidate_checkpoint_sha256'], 'sequences': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--extended-preflight', type=Path, required=True)
    parser.add_argument('--engineering-gate', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.gate,args.extended_preflight,args.engineering_gate,args.checkpoint,args.detector)
    folder = validate_output(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(folder)}))


if __name__ == '__main__':
    main()

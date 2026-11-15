#!/usr/bin/env python3
"""Freeze a matched MPS factorial arm on one priority development sequence.

The verified runtime supports the original multilevel-refiner 17/10 schema
and the region-conditioned 19/10 schema with all matched controls. No current
checkpoint has passed its required development screen. Seq23 always uses the
audited no-inset crop.
The screening step reads validation labels; propagation never opens target
sequence mask files. This script makes no efficacy claim.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from modeling.lean_verified_window_runtime import verified_mps_window_refiner
from modeling.seq23_viewport import IMAGE_SIZE, VIEWPORT, restore_mask, stage_cropped_frames
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import prepare_archived_prompt
from scripts.verify_token_window_refiner_routing_mps import tensor_hash


def priority_inputs(sequence, split_preflight, seq23_reference, seq23_backup, checkpoint, detector):
    pre_path = split_preflight / 'manifest.json'
    pre = json.loads(pre_path.read_text())
    done = json.loads((split_preflight / 'completion.json').read_text())
    if not done['passed'] or digest(pre_path) != done['manifest_sha256']:
        raise ValueError('Split preflight changed')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    frames = videos[sequence]
    if sequence == 'seq23':
        if seq23_reference is None or seq23_backup is None:
            raise ValueError('Seq23 requires its audited cropped-prompt reference and backup')
        ref = seq23_reference
        completion = json.loads((ref / 'completion.json').read_text())
        for name in ('prompt', 'frames', 'setup'):
            if digest(ref / f'{name}.json') != completion[name + '_sha256']:
                raise ValueError(f'Seq23 cropped {name} changed')
        if not completion['passed']:
            raise ValueError('Seq23 cropped reference is incomplete')
        setup = json.loads((ref / 'setup.json').read_text())
        if (setup['viewport_xyxy'] != list(VIEWPORT) or setup['source_image_size'] != list(IMAGE_SIZE) or
                setup['checkpoint_sha256'] != digest(checkpoint) or setup['detector_sha256'] != digest(detector) or
                setup['preprocess_sha256'] != digest(ROOT / 'modeling/seq23_viewport.py')):
            raise ValueError('Seq23 crop, model or detector differs')
        archived = json.loads((ref / 'prompt.json').read_text())
        prompt = {'first_detection_index': archived['first_detection_index'],
                  'box': archived['box_viewport_xyxy'], 'points': archived['points_viewport']}
        metadata = json.loads((ref / 'frames.json').read_text())
        prompt_sha = digest(ref / 'prompt.json')
        view = {'kind': 'fixed_seq23_main_viewport', 'xyxy': list(VIEWPORT)}
    else:
        source = pre['sequences'][sequence]
        folder = Path(source['source']) / sequence
        if (digest(folder / 'prompt.json') != source['prompt_sha256'] or
                digest(folder / 'frames.json') != source['frames_metadata_sha256']):
            raise ValueError('Archived prompt or image metadata changed')
        prompt = json.loads((folder / 'prompt.json').read_text())
        metadata = json.loads((folder / 'frames.json').read_text())
        prompt_sha = digest(folder / 'prompt.json')
        view = {'kind': 'full_source_frame'}
    if (len(frames) != len(metadata) or any(digest(frame.image_path) != metadata[i]['image_sha256']
                                             for i, frame in enumerate(frames))):
        raise ValueError('Source sequence images changed')
    if (type(prompt['first_detection_index']) is not int or prompt['box'] is None or
            prompt['points'] is None or len(prompt['points']) != 2):
        raise ValueError('This runner requires the archived box and two-point prompt')
    start, prompt_mask, viewport, provenance = prepare_archived_prompt(
        sequence, frames, split_preflight, seq23_reference, seq23_backup)
    if start != prompt['first_detection_index'] or provenance['prompt_metadata_sha256'] != prompt_sha:
        raise ValueError('Archived prompt-mask metadata differs')
    if sequence == 'seq23' and tuple(viewport) != tuple(VIEWPORT):
        raise ValueError('Archived seq23 prompt viewport differs')
    # Historical prompts came from repeated detector calls until first detection;
    # target-sequence masks are never opened to construct the prediction.
    return frames, prompt, prompt_sha, view, prompt_mask, provenance


def verify_baseline_identity(baseline, *, sequence, prompt_sha, prompt_mask_sha, checkpoint, detector, candidate, control,
                             view, expected_frames, prompt_frame):
    receipt = json.loads((baseline / 'completion.json').read_text())
    if (not receipt.get('passed') or receipt.get('arm') != 'A' or receipt.get('sequence') != sequence or
            receipt.get('prompt_sha256') != prompt_sha or receipt.get('checkpoint_sha256') != digest(checkpoint) or
            receipt.get('prompt_mask_sha256') != prompt_mask_sha or
            receipt.get('detector_sha256') != digest(detector) or
            receipt.get('candidate_checkpoint_sha256') != digest(candidate / 'checkpoint.pt') or
            receipt.get('control_checkpoint_sha256') != digest(control / 'checkpoint.pt') or
            receipt.get('prompt_frame') != prompt_frame or
            receipt.get('view') != view or receipt.get('runner_source_sha256') != digest(Path(__file__))):
        raise ValueError('Matched MPS arm-A baseline identity differs')
    if ([row['frame'] for row in receipt['masks']] != list(range(expected_frames)) or
            [row['frame'] for row in receipt['post_prompt_state']] != list(range(prompt_frame, expected_frames))):
        raise ValueError('Matched arm-A frame coverage differs')
    for row in receipt['masks']:
        if digest(baseline / f"{row['frame']:06d}.png") != row['sha256']:
            raise ValueError('Matched arm-A mask changed')
    runtime_path = baseline / 'verification/runtime_verification.json'
    if (digest(runtime_path) != receipt['runtime_verification_sha256'] or
            digest(baseline / 'memory_events.json') != receipt['memory_events_sha256'] or
            digest(baseline / 'positions.json') != receipt['positions_sha256'] or
            digest(baseline / 'selections.json') != receipt['selections_sha256']):
        raise ValueError('Matched arm-A runtime verification changed')
    runtime = json.loads(runtime_path.read_text())
    if (not runtime.get('passed') or runtime.get('candidate_checkpoint_sha256') != digest(candidate / 'checkpoint.pt') or
            runtime.get('control_checkpoint_sha256') != digest(control / 'checkpoint.pt') or
            runtime.get('device') != 'mps float32' or runtime.get('torch_version') != str(torch.__version__) or
            runtime.get('memory_budget') != 16384 or runtime.get('insurance_reserve') != 3072 or
            runtime.get('source_sha256', {}).get('modeling/lean_verified_window_runtime.py') != digest(ROOT / 'modeling/lean_verified_window_runtime.py') or
            runtime.get('source_sha256', {}).get('modeling/lean_contrast_refinement_integration.py') != digest(ROOT / 'modeling/lean_contrast_refinement_integration.py') or
            digest(baseline / 'verification/model_tensor_fingerprints.json') != runtime['model_tensor_fingerprints_sha256'] or
            digest(baseline / 'verification/refiner_tensor_fingerprints.json') != runtime['refiner_tensor_fingerprints_sha256']):
        raise ValueError('Matched arm-A runtime configuration or tensor fingerprints differ')
    return receipt


def verify_cross_arm_runtime(baseline_verification, current_verification):
    """Compare recorded live tensor contents and runtime across factorial arms."""
    baseline = json.loads((baseline_verification / 'runtime_verification.json').read_text())
    current = json.loads((current_verification / 'runtime_verification.json').read_text())
    keys = ('candidate_checkpoint_sha256', 'control_checkpoint_sha256', 'selection_auditor_sha256',
            'underlying_screen_sha256',
            'baseline_reference_sha256', 'candidate_format', 'additional_controls',
            'model_tensor_count', 'refiner_tensor_count',
            'model_tensor_fingerprints_sha256', 'refiner_tensor_fingerprints_sha256', 'source_sha256',
            'device', 'torch_version', 'python_version', 'platform', 'machine',
            'mps_device_count', 'gpu_inventory', 'gpu_inventory_note',
            'fallback', 'fast_math', 'seed', 'baseline_replays',
            'memory_budget', 'insurance_reserve')
    if not baseline.get('passed') or not current.get('passed') or any(
            baseline.get(key) is None or baseline.get(key) != current.get(key) for key in keys):
        raise ValueError('Matched factorial arm live tensor or runtime identity differs')
    for folder, record in ((baseline_verification, baseline), (current_verification, current)):
        for name in ('model', 'refiner'):
            if digest(folder / f'{name}_tensor_fingerprints.json') != record[name + '_tensor_fingerprints_sha256']:
                raise ValueError('Recorded live tensor fingerprints changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('preflight', 'split-preflight', 'static-capture', 'seq4-capture',
             'checkpoint', 'detector', 'baseline-reference', 'candidate-run',
             'control-run', 'data-preflight', 'runner-source', 'output')
    for name in names:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sequence', choices=('seq4', 'seq8', 'seq11', 'seq20', 'seq23'), required=True)
    parser.add_argument('--arm', choices=('A', 'B', 'C', 'D'), required=True)
    parser.add_argument('--seq23-reference', type=Path)
    parser.add_argument('--seq23-backup', type=Path)
    parser.add_argument('--global-run', type=Path)
    parser.add_argument('--shuffled-run', type=Path)
    parser.add_argument('--matched-a', type=Path)
    args = parser.parse_args()
    frames, prompt, prompt_sha, view, prompt_mask, prompt_provenance = priority_inputs(
        args.sequence, args.split_preflight, args.seq23_reference, args.seq23_backup,
        args.checkpoint, args.detector)
    if args.arm != 'A' and args.matched_a is None:
        raise ValueError('B/C/D require a frozen, matching MPS arm-A baseline')
    baseline = (verify_baseline_identity(args.matched_a, sequence=args.sequence, prompt_sha=prompt_sha,
                                         prompt_mask_sha=prompt_provenance['prompt_mask_sha256'],
                                         checkpoint=args.checkpoint, detector=args.detector,
                                         candidate=args.candidate_run, control=args.control_run, view=view,
                                         expected_frames=len(frames), prompt_frame=prompt['first_detection_index'])
                if args.arm != 'A' else None)
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    options = {name.replace('-', '_'): getattr(args, name.replace('-', '_')) for name in names}
    options['output'] = out / 'verification'
    options['global_run'] = args.global_run
    options['shuffled_run'] = args.shuffled_run
    with verified_mps_window_refiner(**options) as (model, refiner, lease):
        if baseline is not None:
            verify_cross_arm_runtime(args.matched_a / 'verification', out / 'verification')
        start = prompt['first_detection_index']
        for i in range(start):
            with Image.open(frames[i].image_path) as image:
                width, height = image.size
            Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(out / f'{i:06d}.png')
        rows = []
        with tempfile.TemporaryDirectory(prefix='priority_frames_', dir=out) as temporary:
            if args.sequence == 'seq23':
                stage_cropped_frames(frames, temporary)
            else:
                stage_frames(frames, Path(temporary))
            state = model.init_state(video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False)
            if args.sequence == 'seq23' and (state['video_width'], state['video_height']) != (VIEWPORT[2]-VIEWPORT[0], VIEWPORT[3]-VIEWPORT[1]):
                raise ValueError('Seq23 viewport geometry changed')
            try:
                with torch.inference_mode(), route_lean_contrast_refinement(model, refiner, args.arm, runtime_lease=lease):
                    # Freeze the exact archived automatic mask used by the
                    # matched SAMURAI experiment. It was originally generated
                    # from this box and feature-selected points, not a label.
                    if not prompt_mask.any():
                        raise ValueError('Unexpected empty automatic prompt')
                    _, ids, _ = model.add_new_mask(state, frame_idx=start, obj_id=1, mask=prompt_mask)
                    if ids != [1]:
                        raise ValueError('Unexpected automatic prompt object')
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start, reverse=False):
                        if ids != [1] or frame != start + len(rows):
                            raise ValueError('Unexpected frame order')
                        mask = logits[0, 0].cpu().numpy() > 0
                        if args.sequence == 'seq23':
                            mask = restore_mask(mask)
                        path = out / f'{frame:06d}.png'
                        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                        row = {'frame': frame, 'mask_sha256': digest(path)}
                        if frame > start:
                            entry = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                            for key in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                                row[key + '_sha256'] = tensor_hash(entry[key])
                        rows.append(row)
            finally:
                model.reset_state(state)
        if len(rows) != len(frames)-start:
            raise ValueError('Incomplete video output')
        runtime = model.token_window_runtime
        writes = [x for x in runtime.events if x['stage'] == 'after_write']
        if any(x['total_tokens'] > 16384 or x['insurance_tokens'] > 3072 for x in writes):
            raise ValueError('Memory budget exceeded')
        if baseline is not None:
            if rows[0]['mask_sha256'] != baseline['post_prompt_state'][0]['mask_sha256']:
                raise ValueError('Conditioning mask differs from the matched baseline')
            if len(rows) > 1:
                original = baseline['post_prompt_state'][1]
                for key in ('pred_masks_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
                    if rows[1][key] != original[key]:
                        raise ValueError('First post-prompt native state differs from the matched baseline')
            if args.arm == 'B':
                for row, original in zip(rows[1:], baseline['post_prompt_state'][1:], strict=True):
                    for key in ('pred_masks_sha256', 'maskmem_features_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
                        if row[key] != original[key]:
                            raise ValueError('Output-only routing changed stored state')
                if runtime.events != json.loads((args.matched_a / 'memory_events.json').read_text()):
                    raise ValueError('Output-only routing changed token-window decisions')
        positions, selections = lease.observations()
        if baseline is not None and args.arm == 'B':
            if (positions != json.loads((args.matched_a / 'positions.json').read_text()) or
                    selections != json.loads((args.matched_a / 'selections.json').read_text())):
                raise ValueError('Output-only routing changed position or selection traces')
        (out / 'memory_events.json').write_text(json.dumps(runtime.events, indent=2) + '\n')
        (out / 'positions.json').write_text(json.dumps(positions, indent=2) + '\n')
        (out / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
        receipt = {'passed': True, 'arm': args.arm, 'sequence': args.sequence, 'view': view,
                   'scope': 'Frozen development predictions only; not scored or independently evaluated.',
                   'prompt_frame': start, 'prompt_sha256': prompt_sha, 'checkpoint_sha256': digest(args.checkpoint),
                   'prompt_mask_sha256': prompt_provenance['prompt_mask_sha256'],
                   'prompt_provenance': prompt_provenance,
                   'automatic_box': prompt['box'], 'automatic_points': prompt['points'],
                   'detector_sha256': digest(args.detector),
                   'candidate_checkpoint_sha256': digest(args.candidate_run / 'checkpoint.pt'),
                   'control_checkpoint_sha256': digest(args.control_run / 'checkpoint.pt'),
                   'runtime_verification_sha256': digest(out / 'verification/runtime_verification.json'),
                   'memory_events_sha256': digest(out / 'memory_events.json'),
                   'positions_sha256': digest(out / 'positions.json'),
                   'selections_sha256': digest(out / 'selections.json'),
                   'masks': [{'frame': i, 'sha256': digest(out / f'{i:06d}.png')} for i in range(len(frames))],
                   'post_prompt_state': rows, 'runner_source_sha256': digest(Path(__file__))}
    # Completion is written only after the owned runtime verifies state on exit.
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'arm': args.arm, 'sequence': args.sequence,
                      'frames': len(frames), 'prompt_frame': start}))


if __name__ == '__main__':
    main()

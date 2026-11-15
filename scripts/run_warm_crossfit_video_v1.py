#!/usr/bin/env python3
"""One label-free out-of-fold warm Lean B/C/D propagation run on MPS."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.video import stage_frames
from modeling.lean_channel_value_routing_v1 import route_channel_value_refinement
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.seq23_viewport import restore_mask, stage_cropped_frames
from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
from modeling.warm_crossfit_video_runtime_v1 import verified_warm_crossfit_runtime
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_token_window_refiner_routing_mps import tensor_hash
from scripts.warm_crossfit_video_contract_v1 import require


def compare_baseline_mask(array, reference, frame, diagnostic=None):
    row = next((row for row in reference['rows'] if row['frame'] == frame), None)
    require(row is not None, 'Reference missing frame')
    path = Path(reference['_receipt_path']).parent / f'{frame:06d}.png'
    require(digest(path) == row['mask_sha256'], 'Reference mask changed')
    with Image.open(path) as image:
        saved = np.asarray(image.convert('L')) > 0
    require(digest(path) == row['mask_sha256'], 'Reference mask changed during reading')
    if not np.array_equal(array, saved):
        if diagnostic is not None:
            Path(diagnostic).write_text(json.dumps({'frame': frame, 'native_shape': list(array.shape),
                'reference_shape': list(saved.shape), 'differing_pixels': int(np.count_nonzero(array != saved))
                if array.shape == saved.shape else None, 'native_pixels': int(array.sum()),
                'reference_pixels': int(saved.sum()), 'scope': 'Same-device saved binary masks; not cross-device parity.'},
                indent=2) + '\n')
        raise ValueError('Native warm A/B mask differs; diagnostic preserved')


def run(certificate, certificate_sha, sequence, arm, output, guard, *, limit=0, smoke_audit=None):
    require(arm in ('B', 'C', 'D') and limit in (0, 4), 'Unsupported route or smoke length')
    from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
    root = Path(output).resolve().parents[2]
    require(Path(certificate).resolve() == root / 'certificate/certificate.json' and
            phase_artifact(root, 'certificate', certificate) == certificate_sha,
            'Video certificate is not phase-bound to this batch')
    smoke_sha = None
    if limit == 0:
        from scripts.audit_warm_crossfit_video_smoke_v1 import audit as audit_smoke
        require(smoke_audit is not None and Path(smoke_audit).resolve() == root / 'smoke_audit/audit.json',
                'Full-video execution requires a phase-bound smoke audit')
        smoke_sha = phase_artifact(root, 'smoke_audit', smoke_audit)
        with Path(smoke_audit).open() as file: saved = json.load(file)
        require(digest(smoke_audit) == smoke_sha and saved.get('label_guard', {}).get('open_attempts') == 0 and
                {k: v for k, v in saved.items() if k != 'label_guard'} == audit_smoke(root, certificate, certificate_sha),
                'Full-video smoke audit differs')
    else:
        require(smoke_audit is None, 'Smoke route does not accept an existing smoke audit')
    output = Path(output).resolve(); require(output.is_relative_to(OUT.resolve(strict=True)) and not output.exists(),
                                              'Invalid video output')
    output.mkdir(parents=True, exist_ok=False)
    with verified_warm_crossfit_runtime(certificate, certificate_sha, sequence, output / 'verification') as \
            (model, refiner, lease, anchor, info):
        frames, start = info['frames'], info['start']; reference = dict(info['reference'])
        reference['_receipt_path'] = info['reference_path']
        end = len(frames) if limit == 0 else min(len(frames), start + limit + 1)
        require(end > start + 1, 'Insufficient post-prompt frames')
        for i in range(start):
            with Image.open(frames[i].image_path) as image:
                width, height = image.size
            Image.fromarray(np.zeros((height, width), np.uint8)).save(output / f'{i:06d}.png')
        rows = []; native_masks = []; telemetry = []; started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='frames_', dir=output) as staged:
            if sequence == 'seq23': stage_cropped_frames(frames, staged)
            else: stage_frames(frames, Path(staged))
            state = model.init_state(video_path=staged, offload_video_to_cpu=True, offload_state_to_cpu=False)
            require((state['video_height'], state['video_width']) == info['mask'].shape, 'Prompt view differs')
            try:
                with ExitStack() as stack:
                    stack.enter_context(torch.inference_mode())
                    def before_mask(initial):
                        _, backbone, _, _, _ = model._get_image_feature(state, start, batch_size=1)
                        early = backbone['backbone_fpn'][0][0].detach().float().cpu()
                        reference_prototypes(early, initial)
                    apply_archived_warm_prompt_v2(model, state, start, info['box'], info['points'],
                                                  info['mask'], before_mask=before_mask)
                    stack.enter_context(route_channel_value_refinement(model, refiner, arm,
                        runtime_lease=lease, telemetry=telemetry))
                    options = {} if limit == 0 else {'max_frame_num_to_track': limit}
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start,
                                                                       reverse=False, **options):
                        require(frame == start + len(rows) and ids == [1] and logits.ndim == 4 and
                                tuple(logits.shape[:2]) == (1, 1) and torch.isfinite(logits).all(),
                                'Invalid video output')
                        mask = logits[0, 0].cpu().numpy() > 0
                        if sequence == 'seq23': mask = restore_mask(mask)
                        with Image.open(frames[frame].image_path) as image:
                            require(mask.shape == (image.height, image.width), 'Output geometry differs')
                        file = output / f'{frame:06d}.png'
                        Image.fromarray(mask.astype(np.uint8) * 255).save(file)
                        if frame == start:
                            compare_baseline_mask(mask, reference, frame, output / 'conditioning_mismatch.json')
                        folder = 'cond_frame_outputs' if frame == start else 'non_cond_frame_outputs'
                        entry = state['output_dict_per_obj'][0][folder][frame]
                        row = {'frame': frame, 'output_logits_sha256': tensor_hash(logits)}
                        for key in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                            require(torch.isfinite(entry[key]).all(), f'Nonfinite state: {key}')
                            row[key + '_sha256'] = tensor_hash(entry[key])
                        if arm == 'B':
                            _, native = model._get_orig_video_res_output(state, entry['pred_masks'])
                            native_mask = native[0, 0].cpu().numpy() > 0
                            if sequence == 'seq23': native_mask = restore_mask(native_mask)
                            native_dir = output / 'native'; native_dir.mkdir(exist_ok=True)
                            native_file = native_dir / f'{frame:06d}.png'
                            Image.fromarray(native_mask.astype(np.uint8) * 255).save(native_file)
                            compare_baseline_mask(native_mask, reference, frame, output / 'native_mismatch.json')
                            native_masks.append({'frame': frame, 'sha256': digest(native_file)})
                        row['mask_sha256'] = digest(file); rows.append(row)
            finally:
                model.reset_state(state)
        torch.mps.synchronize(); require(len(rows) == end - start, 'Video frame coverage differs')
        positions, selection = lease.observations()
        trace = {'positions': positions, 'selection': selection, 'anchor_events': anchor,
                 'token_window_events': model.token_window_runtime.events}
        trace_hashes = {}
        for name, value in (*trace.items(), ('telemetry', telemetry)):
            path = output / f'{name}.json'; path.write_text(json.dumps(value, indent=2) + '\n')
            trace_hashes[name] = digest(path)
        if arm == 'B':
            for key, value in trace.items():
                expected = reference[key]
                if not (value == expected[:len(value)] and (limit != 0 or len(value) == len(expected))):
                    (output / 'trace_mismatch.json').write_text(json.dumps({'trace': key,
                        'observed_entries': len(value), 'reference_entries': len(expected),
                        'scope': 'Same-device named trace; not cross-device parity.'}, indent=2) + '\n')
                    raise ValueError(f'Output-only route changed warm A trace: {key}; diagnostic preserved')
        writes = [x for x in trace['token_window_events'] if x['stage'] == 'after_write']
        require(writes and all(x['total_tokens'] <= 16384 and x['insurance_tokens'] <= 3072 for x in writes),
                'Memory budget differs')
        # The warm prompt is committed before routing is installed. Its output
        # is checked above, while telemetry covers subsequent tracked frames.
        require([x['frame'] for x in telemetry] == list(range(start + 1, end)), 'Telemetry coverage differs')
        require(all(digest(frame.image_path) == sha for frame, sha in
                    zip(frames, info['image_sha256'], strict=True)), 'Input image changed during run')
        require(guard['open_attempts'] == 0, 'Annotation opened during video run')
        runtime = output / 'verification/runtime_verification.json'
        receipt = {'passed': True, 'sequence': sequence, 'arm': arm, 'limit': limit,
            'scope': 'Out-of-fold warm video; routing does not directly overwrite native pointer or object score, which may change on later frames through feedback; no efficacy claim.',
            'certificate_sha256': certificate_sha, 'reference_path': info['reference_path'],
            'smoke_audit_sha256': smoke_sha,
            'reference_sha256': info['reference_sha256'], 'prompt_frame': start, 'roi_xyxy': info['roi_xyxy'],
            'runtime_verification_sha256': digest(runtime), 'rows': rows, 'trace_sha256': trace_hashes,
            'native_masks': native_masks,
            'masks': [{'frame': i, 'sha256': digest(output / f'{i:06d}.png')} for i in range(end)],
            'label_guard': dict(guard), 'wall_seconds': time.monotonic() - started,
            'timing_scope': 'Includes staging, prompt, inference, synchronization, telemetry and mask writes; excludes model loading and pre-prompt mask writes. Not inference-only FPS.',
            'runner_sha256': digest(Path(__file__))}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate', type=Path, required=True); parser.add_argument('--certificate-sha256', required=True)
    parser.add_argument('--sequence', required=True); parser.add_argument('--arm', choices=('B', 'C', 'D'), required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--limit', type=int, choices=(0, 4), default=0)
    parser.add_argument('--smoke-audit', type=Path)
    args = parser.parse_args(); guard = install_mask_open_guard(DATA)
    run(args.certificate, args.certificate_sha256, args.sequence, args.arm, args.output, guard,
        limit=args.limit, smoke_audit=args.smoke_audit)
    print(json.dumps({'passed': True, 'sequence': args.sequence, 'arm': args.arm, 'output': str(args.output)}))


if __name__ == '__main__': main()

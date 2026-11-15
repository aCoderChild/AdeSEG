#!/usr/bin/env python3
"""Engineering-only A/B/C/D routing test on a real token-window MPS predictor.

Runs the verified seq4 automatic prompt on fresh model instances. The
refiners are deliberately synthetic: identity and a forced constant logit.
No label or trained refiner is used, and this test does not authorize an
efficacy experiment. All experiment artifacts must be written to Drive.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.verify_refiner_capture_static import verify as verify_capture

FORK = ROOT / 'external/Lean-SAM2-token-window'


class _Identity(torch.nn.Module):
    _trained_checkpoint_loaded = True  # Explicit synthetic test stimulus only.

    def forward(self, early, deep, low):
        return low.clone()


class _ForcedLogit(_Identity):
    def forward(self, early, deep, low):
        # A deliberately visible, nonzero stimulus; this is not a model output.
        return torch.full_like(low, 2.0)


def tensor_hash(tensor):
    if not torch.is_tensor(tensor):
        raise ValueError('Expected a tensor in compact memory state')
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def verify_archived_baseline(baseline, reference_dir, *, checkpoint, prompt, seq4_capture, end):
    """Bind the fresh, unwrapped baseline to the independently saved replay."""
    old = json.loads((reference_dir / 'receipt.json').read_text())
    expected = {
        'passed': True, 'device': 'mps', 'sequence': 'seq4', 'prompt_frame': 9,
        'checkpoint_sha256': digest(checkpoint), 'prompt_sha256': digest(prompt),
        'capture_receipt_sha256': digest(seq4_capture / 'channel_feature_receipt.json'),
        'runner_sha256': digest(ROOT / 'scripts/verify_lean_window_mps_replay.py'),
        'mps_rope_sha256': digest(ROOT / 'modeling/lean_indexed_spatial_rope_mps.py'),
        'mps_blocking_sha256': digest(ROOT / 'modeling/lean_mps_blocking_pointer_position.py'),
        'torch': str(torch.__version__), 'fallback': '0',
    }
    if any(old.get(key) != value for key, value in expected.items()) or old['end_frame'] < end:
        raise ValueError('Archived MPS baseline identity differs')
    old_rows = {row['frame']: row for row in old['rows']}
    for row in baseline['rows']:
        saved = reference_dir / f"{row['frame']:06d}.png"
        if (row['frame'] not in old_rows or digest(saved) != old_rows[row['frame']]['mask_sha256'] or
                row['mask_sha256'] != old_rows[row['frame']]['mask_sha256']):
            raise ValueError('Fresh baseline mask differs from archived MPS replay')
    old_events = [event for event in old['token_window_events'] if event.get('frame', end+1) <= end]
    if baseline['token_window_events'] != old_events:
        raise ValueError('Fresh baseline memory events differ from archived MPS replay')


def build_model(checkpoint, expected_state):
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=checkpoint, device='mps'), FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):
        raise ValueError('Wrong Lean checkout imported')
    if model.num_maskmem != 7 or model.mem_keep_ratio != .25 or model.insurance_max_frames != 3 or not model.enable_insurance_bank:
        raise ValueError('Unexpected memory configuration')
    if module_tensor_fingerprints(model) != expected_state:
        raise ValueError('Live predictor parameters or buffers differ')
    model.ema_history_replace_old_tail_in_attn = False
    model.ema_history_resident_prune = False
    runtime = TokenWindowRuntime(16384, 3072)
    model.token_window_runtime = runtime
    selections = []
    install_multiscale_anchor(model, .5, selection_events=selections)
    return model, runtime, selections


def run_one(name, arm, refiner, checkpoint, expected_state, frames, prompt, end, out,
            *, frozen_mask=None, warm_with_points=True):
    model, runtime, selections = build_model(checkpoint, expected_state)
    torch.manual_seed(20261021)
    torch.mps.manual_seed(20261021)
    start = prompt['first_detection_index']
    run_dir = out / name
    run_dir.mkdir()
    positions, rows = [], []
    with tempfile.TemporaryDirectory(prefix='lean_route_frames_', dir=run_dir) as temp:
        stage_frames(frames, Path(temp))
        state = model.init_state(video_path=temp, offload_video_to_cpu=True, offload_state_to_cpu=False)
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(indexed_spatial_rope_mps(model, positions))
            stack.enter_context(blocking_mps_pointer_position(model))
            if arm is not None:
                stack.enter_context(route_lean_contrast_refinement(model, refiner, arm, test_only=True))
            if warm_with_points:
                _, ids, logits = model.add_new_points_or_box(
                    state, frame_idx=start, obj_id=1, box=np.asarray(prompt['box'], np.float32))
                if ids != [1]:
                    raise ValueError('Unexpected prompt object')
                _, ids, logits = model.add_new_points_or_box(
                    state, frame_idx=start, obj_id=1, points=np.asarray(prompt['points'], np.float32),
                    labels=np.array([1, 0], np.int32), clear_old_points=False)
                generated = logits[0, 0].cpu().numpy() > 0
                if ids != [1]:
                    raise ValueError('Unexpected prompt object')
            elif frozen_mask is None:
                raise ValueError('Direct prompt requires an explicitly supplied frozen mask')
            initial = generated if frozen_mask is None else np.asarray(frozen_mask, dtype=bool)
            if not initial.any():
                raise ValueError('Invalid automatic prompt')
            model.add_new_mask(state, frame_idx=start, obj_id=1, mask=initial)
            for frame, ids, logits in model.propagate_in_video(
                    state, start_frame_idx=start, max_frame_num_to_track=end-start, reverse=False):
                if ids != [1] or frame != start + len(rows):
                    raise ValueError('Unexpected propagation order')
                path = run_dir / f'{frame:06d}.png'
                Image.fromarray((logits[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255).save(path)
                record = {'frame': frame, 'mask_sha256': digest(path), 'logits_sha256': tensor_hash(logits)}
                group = 'cond_frame_outputs' if frame == start else 'non_cond_frame_outputs'
                entry = state['output_dict_per_obj'][0][group][frame]
                for key in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                    record[key + '_sha256'] = tensor_hash(entry[key])
                rows.append(record)
        model.reset_state(state)
    if len(rows) != end-start+1 or module_tensor_fingerprints(model) != expected_state:
        raise ValueError('Incomplete replay or model tensor state changed')
    writes = [e for e in runtime.events if e['stage'] == 'after_write']
    if not writes or any(e['total_tokens'] > 16384 or e['insurance_tokens'] > 3072 for e in writes):
        raise ValueError('Memory budget exceeded')
    result = {'name': name, 'rows': rows, 'token_window_events': runtime.events,
              'selection_events': selections, 'positions': positions}
    (run_dir / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{name}: {len(rows)} frames, {len(writes)} writes', flush=True)
    return result


def verify_routes(runs):
    base = runs['baseline']
    for arm in ('A', 'B', 'C', 'D'):
        identity = runs['identity_' + arm]
        if identity['rows'] != base['rows'] or identity['token_window_events'] != base['token_window_events']:
            raise ValueError(f'Identity arm {arm} changed output or compact state')
        if identity['selection_events'] != base['selection_events'] or identity['positions'] != base['positions']:
            raise ValueError(f'Identity arm {arm} changed selection or positions')
    # On the first post-prompt frame, all arms see the same original history.
    first = base['rows'][1]
    for arm in ('B', 'C', 'D'):
        row = runs['forced_' + arm]['rows'][1]
        for key in ('pred_masks_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
            if row[key] != first[key]:
                raise ValueError(f'Forced arm {arm} changed native {key}')
        if (row['logits_sha256'] != first['logits_sha256']) != (arm in ('B', 'D')):
            raise ValueError(f'Forced arm {arm} used the wrong output route')
        if (row['maskmem_features_sha256'] != first['maskmem_features_sha256']) != (arm in ('C', 'D')):
            raise ValueError(f'Forced arm {arm} used the wrong memory route')
    b = runs['forced_B']
    for original, changed in zip(base['rows'], b['rows'], strict=True):
        for key in ('pred_masks_sha256', 'maskmem_features_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
            if key in original and original[key] != changed.get(key):
                raise ValueError(f'Output-only arm changed later stored {key}')
    if b['token_window_events'] != base['token_window_events'] or b['selection_events'] != base['selection_events']:
        raise ValueError('Output-only arm changed memory policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'split-preflight', 'static-capture', 'seq4-capture',
                 'checkpoint', 'detector', 'baseline-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--end-frame', type=int, default=13)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0':
        raise ValueError('Explicitly disable MPS fallback and fast math')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    static = verify_capture(args.preflight, args.split_preflight, args.static_capture,
                            args.checkpoint, args.detector)
    pre = json.loads((args.split_preflight / 'manifest.json').read_text())
    source = pre['sequences']['seq4']
    seq4 = json.loads((args.seq4_capture / 'channel_feature_receipt.json').read_text())
    if seq4['feature_pipeline_sha256'] != static['feature_pipeline_sha256']:
        raise ValueError('Seq4 pipeline differs from capture provenance')
    prompt_dir = Path(source['source']) / 'seq4'
    prompt_path = prompt_dir / 'prompt.json'
    if digest(prompt_path) != source['prompt_sha256']:
        raise ValueError('Archived prompt changed')
    prompt = json.loads(prompt_path.read_text())
    start = prompt['first_detection_index']
    if start != 9 or prompt['points'] is None or args.end_frame <= start:
        raise ValueError('Unexpected engineering prompt or range')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    frames = videos['seq4']
    metadata = json.loads((prompt_dir / 'frames.json').read_text())
    if args.end_frame >= len(frames) or len(frames) != len(metadata) or any(
            digest(frame.image_path) != metadata[i]['image_sha256'] for i, frame in enumerate(frames)):
        raise ValueError('Sequence images or frame count changed')

    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(FORK))
    reference = build_author_predictor(SimpleNamespace(method='lean', checkpoint=args.checkpoint, device='cpu'), FORK)
    expected_state = module_tensor_fingerprints(reference)
    del reference
    fingerprint_path = out / 'cpu_reference_fingerprints.json'
    fingerprint_path.write_text(json.dumps(expected_state, indent=2) + '\n')
    runs = {}
    runs['baseline'] = run_one('baseline', None, None, args.checkpoint, expected_state, frames, prompt, args.end_frame, out)
    verify_archived_baseline(runs['baseline'], args.baseline_reference, checkpoint=args.checkpoint,
                             prompt=prompt_path, seq4_capture=args.seq4_capture, end=args.end_frame)
    for arm in ('A', 'B', 'C', 'D'):
        runs['identity_' + arm] = run_one('identity_' + arm, arm, _Identity().eval(), args.checkpoint,
                                          expected_state, frames, prompt, args.end_frame, out)
    for arm in ('B', 'C', 'D'):
        runs['forced_' + arm] = run_one('forced_' + arm, arm, _ForcedLogit().eval(), args.checkpoint,
                                        expected_state, frames, prompt, args.end_frame, out)
    verify_routes(runs)
    receipt = {'passed': True, 'scope': 'Synthetic routing engineering only; no learned-refiner efficacy or authorization.',
               'sequence': 'seq4', 'prompt_frame': start, 'end_frame': args.end_frame,
               'checkpoint_sha256': digest(args.checkpoint), 'prompt_sha256': digest(prompt_path),
               'source_sha256': digest(Path(__file__)), 'static_manifest_sha256': static['manifest_sha256'],
               'cpu_model_tensors_verified': len(expected_state), 'seed': 20261021,
               'cpu_model_fingerprints_sha256': digest(fingerprint_path),
               'baseline_reference_receipt_sha256': digest(args.baseline_reference / 'receipt.json'),
               'source_files_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
                   ROOT / 'modeling/lean_live_state_audit.py',
                   ROOT / 'modeling/lean_contrast_refinement_integration.py',
                   ROOT / 'modeling/contrast_refinement_routing.py',
                   ROOT / 'modeling/lean_indexed_spatial_rope_mps.py',
                   ROOT / 'modeling/lean_mps_blocking_pointer_position.py',
                   ROOT / 'modeling/lean_token_window_storage.py',
                   ROOT / 'modeling/token_memory_window.py',
                   ROOT / 'modeling/lean_multiscale_anchor.py')},
               'torch_version': str(torch.__version__), 'python_version': sys.version,
               'platform': platform.platform(), 'machine': platform.machine(),
               'device': 'mps float32', 'fallback': 'disabled', 'fast_math': 'disabled',
               'memory_budget': 16384, 'insurance_reserve': 3072,
               'runs': {name: digest(out / name / 'receipt.json') for name in runs}}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

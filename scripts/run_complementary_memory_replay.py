"""Frozen Run 168 engineering checks followed by fixed-native-memory diagnosis.

Reuses author SAM2 tracking, the existing native-read capture and Cutie reader.
This runner adds comparison/receipt orchestration, not a new inference method.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.cfbi_training import capture_training_read, clone_tensors
from modeling.complementary_memory import ComplementaryMemoryReader, NativeComplementaryRead
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_pilot import (
    activate_author_repo, aggregate, atomic_json, init_predictor, prepare, state_digest,
)

PROTOCOL = 'protocols/complementary_memory_replay_v1.json'
OUTPUT_KEYS = ('pred_masks', 'obj_ptr', 'object_score_logits')
LAYOUT_KEYS = ('layers', 'memory_frames', 'spatial_keys', 'pointer_keys',
               'pointer_input_sha256', 'pointer_position_sha256',
               'stored_mask_alignment', 'native_positions_verified')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def content_hash(value):
    return fingerprint(value)['sha256']


def check_exact(actual, reference):
    for key in OUTPUT_KEYS:
        require(tensor_hash(actual[key]) == tensor_hash(reference[key]), f'Native identity failed: {key}')


def check_layout(actual, reference):
    require(actual['layers'] == [0, 1, 2, 3], 'Missing memory-attention layer')
    for key in LAYOUT_KEYS:
        require(actual[key] == reference[key], f'Replay layout differs: {key}')


def tensor_stats(value):
    value = value.detach().float()
    require(bool(torch.isfinite(value).all()), 'Nonfinite diagnostic tensor')
    return {'min': float(value.min()), 'max': float(value.max()), 'mean': float(value.mean()),
            'rms': float(value.square().mean().sqrt())}


@torch.inference_mode()
def replay(model, sample, reader=None, mode='regional'):
    """Private clone per condition; no memory encoder or state writeback allowed."""
    before = content_hash(sample)
    local = clone_tensors(sample)
    require(content_hash(local) == before, 'Replay clone changed tensor contents/layout')
    diagnostics = {}

    def decoded(module, args, output):
        diagnostics['raw_decoder_logits'] = output[0].detach().clone()
        diagnostics['decoder_iou_predictions'] = output[1].detach().clone()

    def residual(module, args, output):
        native = args[0]
        delta = output - native
        diagnostics['residual'] = {**tensor_stats(delta), 'l1': float(delta.abs().sum()),
                                   'native_rms': tensor_stats(native)['rms']}

    handles = [model.sam_mask_decoder.register_forward_hook(decoded)]
    if reader is not None:
        handles.append(reader.register_forward_hook(residual))
    try:
        with patch.object(model, '_encode_new_memory', side_effect=RuntimeError('Replay encoded new memory')):
            with NativeComplementaryRead(model, reader, mode, trace_only=reader is None,
                                         record_update_norm=reader is not None) as observer:
                result = model.track_step(
                    frame_idx=local['frame'], is_init_cond_frame=False,
                    current_vision_feats=local['vision'], current_vision_pos_embeds=local['positions'],
                    feat_sizes=local['sizes'], point_inputs=None, mask_inputs=None,
                    output_dict=local['output_dict'], num_frames=local['num_frames'],
                    track_in_reverse=False, run_mem_encoder=False)
        require(len(observer.calls) == 1, 'Expected one complete replay read')
        require(result['maskmem_features'] is None and result['maskmem_pos_enc'] is None,
                'Replay returned encoded memory')
        require(content_hash(local) == before and content_hash(sample) == before, 'Replay mutated input')
        require('raw_decoder_logits' in diagnostics, 'Missing decoder observation')
        for key in OUTPUT_KEYS:
            tensor_stats(result[key])
        tensor_stats(diagnostics['raw_decoder_logits'])
        return result, observer.calls[0], diagnostics
    finally:
        for handle in handles:
            handle.remove()


def score_row(frame, item, truth, logits, result, trace, diagnostics, target, arm):
    prediction = logits[0, 0].cpu().numpy() > 0
    scores = calculate_scores(prediction, truth)
    scores['boundary_f'] = float(db_eval_boundary(prediction, truth))
    folder = target / arm
    folder.mkdir(exist_ok=True)
    mask_path = folder / f'{frame:06d}.png'
    Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
    tensor_path = folder / f'{frame:06d}.pt'
    torch.save({key: result[key].detach().cpu() for key in OUTPUT_KEYS} | {
        'raw_decoder_logits': diagnostics['raw_decoder_logits'].cpu(),
        'decoder_iou_predictions': diagnostics['decoder_iou_predictions'].cpu()}, tensor_path)
    return {'frame': frame, 'source_frame_index': item.frame_index, 'present': bool(truth.any()),
            'scores': scores, 'absent_fp': bool(prediction.any()) if not truth.any() else None,
            'object_score_logit': float(result['object_score_logits'].item()),
            'final_logit_stats': tensor_stats(result['pred_masks']),
            'raw_decoder_logit_stats': tensor_stats(diagnostics['raw_decoder_logits']),
            'residual': diagnostics.get('residual'), 'read': trace,
            'prediction_sha256': tensor_hash(logits), 'mask_sha256': digest(mask_path),
            'tensor_sha256': digest(tensor_path)}


def arm_summary(rows, model_hash):
    present = [r for r in rows if r['present']]
    absent = [r for r in rows if not r['present']]
    result = {'target_frames': len(rows), 'present_frames': len(present), 'gt_empty_frames': len(absent),
              'original_sam2_parameter_sha256': model_hash,
              'all_frame_dice': float(np.mean([r['scores']['dice'] for r in rows])),
              'gt_empty_false_alarm_count': sum(r['absent_fp'] for r in absent),
              'gt_empty_false_alarm_rate': float(np.mean([r['absent_fp'] for r in absent])) if absent else None}
    for metric in ('dice', 'iou', 'boundary_f'):
        result[f'present_{metric}'] = float(np.mean([r['scores'][metric] for r in present])) if present else None
    return result


@torch.inference_mode()
def run_sequence(name, frames, out, source, setup, repo, readers, zeros, gate, progress):
    target = out / ('gate' if gate else 'full') / name
    target.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(setup['checkpoint']['path'])
    model = init_predictor(checkpoint, repo)
    model_hash = parameter_digest(model)
    rows = {arm: [] for arm in ('native', 'generic', 'regional')}
    references = None if gate else {
        r['frame']: r for r in map(json.loads, (source / 'evaluation/native' / name / 'frames.jsonl').read_text().splitlines())}
    receipts = {r['local_frame']: r for r in setup['input_receipts'][name]['frames']}
    for index, receipt in receipts.items():
        require(digest(frames[index].image_path) == receipt['image_sha256'], 'Input image changed')
        require(digest(frames[index].mask_path) == receipt['mask_sha256'], 'GT input changed')
        require(frames[index].frame_index == receipt['source_frame'], 'Manifest order changed')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='staged_frames_', dir=target) as directory:
        state, prompt = prepare(model, frames, Path(directory))
        require(prompt == setup['input_receipts'][name]['prompt'], 'Prompt mismatch')
        expected = list(range(prompt + 1, len(frames)))
        repeat_frames = {expected[0], expected[len(expected)//2], expected[-1]} if gate else set()
        for frame, ids, native_logits in model.propagate_in_video(state, start_frame_idx=prompt):
            require(ids == [1], 'Unexpected object identities')
            if frame == prompt:
                continue
            require(frame == expected[len(rows['native'])], 'Nonchronological native rollout')
            if references is not None:
                require(tensor_hash(native_logits) == references[frame]['prediction_sha256'], f'Run167 native mismatch: {name}/{frame}')
            sample = capture_training_read(model, state, frame)
            require(all(index < frame for bucket in sample['output_dict'].values() for index in bucket), 'Current/future memory leakage')
            own = state['output_dict_per_obj'][0]
            stored = own['non_cond_frame_outputs'][frame]
            # Guard all conditioning, recent spatial/pointer memory, and current image cache.
            signature = lambda: content_hash(bank_contents(state, frame, [], model.max_obj_ptrs_in_encoder))
            native_before = signature()
            restored, reference_trace, native_diagnostics = replay(model, sample)
            check_exact(restored, stored)
            require(tensor_hash(model._get_orig_video_res_output(state, restored['pred_masks'])[1]) == tensor_hash(native_logits), 'Native full-resolution replay mismatch')
            for mode in ('generic', 'regional'):
                zero_out, trace, diagnostic = replay(model, sample, zeros[mode], mode)
                check_exact(zero_out, stored)
                check_layout(trace, reference_trace)
                require(diagnostic['residual']['l1'] == 0., 'Zero control has a nonzero residual')
            outputs = {'native': (restored, reference_trace, native_diagnostics)}
            for mode in ('generic', 'regional'):
                outputs[mode] = replay(model, sample, readers[mode], mode)
                check_layout(outputs[mode][1], reference_trace)
            if frame in repeat_frames:
                for mode in ('regional', 'generic'):
                    repeated = replay(model, sample, readers[mode], mode)
                    check_exact(repeated[0], outputs[mode][0])
                    require(repeated[1] == outputs[mode][1], 'Repeat trace changed')
            final, final_trace, _ = replay(model, sample)
            check_exact(final, stored)
            check_layout(final_trace, reference_trace)
            require(signature() == native_before, 'Native bank/cache changed during replay')
            truth = load_label_mask(frames[frame].mask_path, [1]) == 1
            for arm, (result, trace, diagnostics) in outputs.items():
                logits = model._get_orig_video_res_output(state, result['pred_masks'])[1]
                row = score_row(frame, frames[frame], truth, logits, result, trace, diagnostics, target, arm)
                rows[arm].append(row)
                with (target / f'{arm}_frames.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            with (target / 'integrity.jsonl').open('a') as stream:
                stream.write(json.dumps({'frame': frame, 'native_exact': True, 'both_zero_controls_exact': True,
                    'native_restoration_exact': True, 'no_writeback': True, 'native_bank_cache_sha256': native_before,
                    'input_sample_sha256': content_hash(sample), 'reversed_repeat_checked': frame in repeat_frames}) + '\n')
            atomic_json(out / 'progress.json', {'phase': 'engineering_gate' if gate else 'fixed_native_replay',
                'sequence': name, 'completed_in_sequence': len(rows['native']), 'expected_in_sequence': len(expected),
                'previous_completed_targets': progress, 'pid': os.getpid(), 'updated_at': datetime.now(timezone.utc).isoformat()})
            if len(rows['native']) % 10 == 0 or frame == expected[-1]:
                print(f'{name}: {len(rows["native"])}/{len(expected)} targets; gate={gate}', flush=True)
        require(len(rows['native']) == len(expected), 'Incomplete sequence')
        model.reset_state(state)
    require(parameter_digest(model) == model_hash, 'Original SAM2 parameters changed')
    expected_hash = json.loads((source / 'training/seed_20261106/generic/training_receipt.json').read_text())['original_sam2_parameter_sha256']
    require(model_hash == expected_hash, 'Original weights differ from Run167')
    result = {a: arm_summary(r, model_hash) for a, r in rows.items()}
    atomic_json(target / 'summary.json', {'arms': result, 'targets': len(expected), 'passed': True,
        'elapsed_seconds': time.monotonic() - start,
        'receipt_hashes': {p.name: digest(p) for p in target.glob('*.jsonl')}})
    del model, sample, state, outputs, restored, final
    gc.collect()
    return result


def preflight(source, out):
    protocol_path = ROOT / PROTOCOL
    committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'HEAD:{PROTOCOL}'])
    require(committed == protocol_path.read_bytes(), 'Replay protocol must be committed before launch')
    protocol = json.loads(protocol_path.read_text())
    original = json.loads((source / 'setup.json').read_text())
    setup = original['setup_core']
    require(digest(ROOT / 'data/polypgen_sequence.jsonl') == setup['manifest_sha256'], 'Manifest changed')
    require(digest(Path(setup['checkpoint']['path'])) == protocol['checkpoint_sha256'], 'SAM2 checkpoint changed')
    for name, expected in setup['sources'].items():
        require(digest(ROOT / name) == expected, f'Run167 source changed: {name}')
    repo, revision = activate_author_repo()
    require(revision == setup['sam2_revision'], 'SAM2 revision changed')
    readers, zeros, state_hashes = {}, {}, {}
    for arm in ('generic', 'regional'):
        path = source / f'training/seed_20261106/{arm}/reader_final.pt'
        require(digest(path) == protocol['reader_files_sha256'][arm], 'Reader file changed')
        state = torch.load(path, map_location='cpu', weights_only=True)
        receipt = json.loads(path.with_name('training_receipt.json').read_text())
        require(state_digest(state) == receipt['reader_final_state_sha256'], 'Reader state mismatch')
        reader = ComplementaryMemoryReader().eval().requires_grad_(False)
        reader.load_state_dict(state, strict=True)
        readers[arm] = reader
        zeros[arm] = copy.deepcopy(reader)
        with torch.no_grad():
            zeros[arm].output_projection.weight.zero_()
        state_hashes[arm] = state_digest(reader.state_dict())
    sources = set(setup['sources']) | {PROTOCOL, 'scripts/run_complementary_memory_replay.py',
        'tests/test_complementary_memory_replay.py', 'external/sam2/sam2/modeling/sam/mask_decoder.py'}
    source_hashes = {name: digest(ROOT / name) for name in sorted(sources)}
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(ROOT / name, name)
    atomic_json(out / 'setup.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'command': sys.argv, 'protocol': protocol, 'protocol_sha256': digest(protocol_path),
        'protocol_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_run': str(source), 'source_run_setup_sha256': digest(source / 'setup.json'),
        'source_hashes': source_hashes, 'source_snapshot_sha256': digest(out / 'source_snapshot.zip'),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__}, 'reader_state_sha256': state_hashes})
    return protocol, setup, repo, readers, zeros, state_hashes, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        source = args.source_run.resolve()
        protocol, setup, repo, readers, zeros, reader_hashes, sources = preflight(source, out)
        videos = load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev') | load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')
        gate_targets = 0
        for name in protocol['engineering_sequences']:
            result = run_sequence(name, videos[name], out, source, setup, repo, readers, zeros, True, gate_targets)
            gate_targets += result['native']['target_frames']
        require(gate_targets == protocol['engineering_targets'], 'Engineering coverage mismatch')
        atomic_json(out / 'gate_completion.json', {'passed': True, 'targets': gate_targets, 'setup_sha256': digest(out / 'setup.json')})
        summaries = {a: {} for a in ('native', 'generic', 'regional')}
        count = 0
        for name in protocol['diagnostic_sequences']:
            result = run_sequence(name, videos[name], out, source, setup, repo, readers, zeros, False, count)
            for arm in summaries:
                summaries[arm][name] = result[arm]
            count += result['native']['target_frames']
        require(count == protocol['diagnostic_targets'], 'Diagnostic coverage mismatch')
        for arm, reader in readers.items():
            require(state_digest(reader.state_dict()) == reader_hashes[arm], 'Trained reader changed')
        require(all(digest(ROOT / name) == h for name, h in sources.items()), 'Sources changed during diagnostic')
        result = aggregate(summaries)
        full = json.loads((source / 'seed_20261106_summary.json').read_text())
        result['replay_minus_full_propagation'] = {a: {s: summaries[a][s]['present_dice'] - full['per_sequence'][s][a]['present_dice']
            for s in protocol['diagnostic_sequences']} for a in summaries}
        atomic_json(out / 'summary.json', result)
        atomic_json(out / 'completion.json', {'status': 'complete', 'gate_targets': gate_targets,
            'diagnostic_targets': count, 'elapsed_seconds': time.monotonic() - started,
            'summary_sha256': digest(out / 'summary.json'), 'setup_sha256': digest(out / 'setup.json'),
            'interpretation': 'Post-hoc fixed-native-history diagnostic; not propagated VOS efficacy or independent confirmation.'})
        atomic_json(out / 'progress.json', {'phase': 'complete', 'diagnostic_targets': count})
    except BaseException as error:
        atomic_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()

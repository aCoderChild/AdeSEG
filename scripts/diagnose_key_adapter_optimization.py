"""Repeated TRAIN-only clips to check optimization, not segmentation efficacy.

Reuse the verified SAM2/MoGA-derived adapter, loss, optimizer and replay helpers.
The old failed validation result and its unopened test split are unaffected.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.video import load_manifest
from modeling.complementary_memory.training import clone_tensors, reader_clip_loss
from modeling.token_conditioned_key_adapter import NativeTokenConditionedKeyAdapter
from scripts.mavos_value_utils import activate_author_repo, atomic_json, atomic_torch_save, prepare
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, GATE_DEFAULT, MANIFEST, PROTOCOL, _capture_rng,
    _check_training_trace, _guard_sources, _make_heads, _model_digest,
    _release_gate, _restore_rng, _runner_training_smoke, _source_hashes,
    _train_chunk, _tree_digest,
)
from scripts.verify_token_conditioned_key_adapter_runner import _adamw, _assert_equal, _samples

DRIVE = CHECKPOINT_DEFAULT.parents[1]
DESIGN = DRIVE / 'research_audits/key_adapter_optimization_20261109/next_diagnostic_design.json'
SMOKE = DRIVE / 'token_conditioned_key_adapter_polypgen_v1/attempt_03_runner_training_smoke_2026-11-09'
OUTPUT = DRIVE / 'key_adapter_optimization_validity_v1'
MEMORY_FIELDS = ('maskmem_pos_enc', 'pred_masks', 'obj_ptr', 'object_score_logits', 'maskmem_features')


def fixed_clip(frames):
    start = prompt_index(frames, {'initialization': 'first_labeled_frame'})
    if start + 5 > len(frames):
        raise ValueError('Need prompt plus four post-prompt manifest frames')
    return frames[start:start + 5]


def require_gate(path, identity):
    completion_path = path.parent / 'completion.json'
    completion = json.loads(completion_path.read_text())
    gate = json.loads(path.read_text())
    if (not completion.get('passed') or completion.get('results_sha256') != digest(path)
            or gate.get('scope') != 'engineering gate' or not gate.get('passed')
            or gate.get('identity') != identity):
        raise RuntimeError('Diagnostic gate is incomplete, failed, corrupted or uses different sources/data')


def evaluate(model, heads, mode, cache):
    result = {}
    for name, item in cache.items():
        with torch.no_grad(), NativeTokenConditionedKeyAdapter(model, heads, mode) as observer:
            loss, bank, _ = reader_clip_loss(model, item['samples'], item['bank'])
        _check_training_trace(observer.calls, 4)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite fixed-clip evaluation loss')
        if _tree_digest(item['bank']) != item['bank_sha256']:
            raise RuntimeError('Fixed starting bank mutated')
        result[name] = {'loss': float(loss), 'bank_sha256': _tree_digest(bank),
                        'mean_absolute_key_residual_by_layer': [
                            statistics.mean(x['key_delta_l1'] / (x['spatial_keys'] * 64)
                                            for x in observer.calls if x['layer'] == layer)
                            for layer in range(4)]}
    return result


def update(model, heads, optimizer, item, name, mode, protocol):
    result = _train_chunk(model, heads, optimizer, item['samples'], item['bank'],
                          name, mode, float(protocol['training']['optimizer']['gradient_clip_norm']))
    if _tree_digest(item['bank']) != item['bank_sha256']:
        raise RuntimeError('An optimizer exposure mutated its fixed starting bank')
    return result


def replay_update(model, heads, optimizer, item, name, mode, protocol, path):
    """Compare an actual update with the same update from a serialized checkpoint."""
    snapshot = {'heads': copy.deepcopy(heads.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': _capture_rng(),
                'bank_sha256': item['bank_sha256']}
    atomic_torch_save(path, snapshot)
    original = update(model, heads, optimizer, item, name, mode, protocol)
    original_rng = _capture_rng()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored = _make_heads(model, protocol)
    restored.load_state_dict(saved['heads'], strict=True)
    restored_optimizer = _adamw(restored, protocol)
    restored_optimizer.load_state_dict(saved['optimizer'])
    _restore_rng(saved['rng'])
    repeated = update(model, restored, restored_optimizer, item, name, mode, protocol)
    _assert_equal(heads.state_dict(), restored.state_dict(), 'resume.heads')
    _assert_equal(optimizer.state_dict(), restored_optimizer.state_dict(), 'resume.optimizer')
    _assert_equal(original_rng, _capture_rng(), 'resume.rng')
    _assert_equal(original[0], repeated[0], 'resume.loss')
    _assert_equal(original[3], repeated[3], 'resume.diagnostics')
    if _tree_digest(original[1]) != _tree_digest(repeated[1]):
        raise RuntimeError('Serialized resume produced a different predicted memory bank')
    return restored, restored_optimizer, repeated


def run(args):
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(out)
    started = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text())
    design = json.loads(DESIGN.read_text())
    names = design['data']['sequences']
    if names != ['seq2', 'seq3'] or not set(names) <= set(protocol['data']['train_sequences']):
        raise RuntimeError('Diagnostic sequences must be frozen TRAIN sequences')
    if design['data']['validation_access'] or design['data']['test_access']:
        raise RuntimeError('This diagnostic cannot access validation/test data')
    if digest(args.checkpoint) != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from frozen pretrained initialization')
    sources = _source_hashes()
    sources[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    test_source = 'tests/test_key_adapter_optimization_diagnostic.py'
    sources[test_source] = digest(ROOT / test_source)
    inherited = _release_gate(GATE_DEFAULT, args.checkpoint, sources)
    # The inherited smoke enumerates the original runner's sources exactly.
    _runner_training_smoke(SMOKE, args.checkpoint, _source_hashes())
    identity = {'sources': sources, 'design_sha256': digest(DESIGN),
                'protocol_sha256': digest(PROTOCOL), 'checkpoint_sha256': digest(args.checkpoint),
                'manifest_sha256': digest(MANIFEST), 'multi_object_gate_sha256': inherited['results_sha256']}
    if not args.gate_only:
        if args.gate_receipt is None:
            raise RuntimeError('A matching real-frame diagnostic gate is required')
        require_gate(args.gate_receipt, identity)
    atomic_json(out / 'identity.json', identity)
    atomic_json(out / 'design.json', design)
    atomic_json(out / 'pilot_protocol.json', protocol)
    for relative in sources:
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(protocol['training']['seed']))
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('SAM2 revision changed')
    model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
        checkpoint=args.checkpoint, model_config=protocol['model']['configuration']), repo)
    model.requires_grad_(False).eval()
    original_hash = _model_digest(model)
    videos = load_manifest(MANIFEST, 'dev')  # Only selected train image/mask files are opened below.
    cache, input_rows = {}, {}
    with tempfile.TemporaryDirectory(prefix='fixed_clips_', dir=out) as temp:
        for name in names:
            frames = fixed_clip(videos[name])
            input_rows[name] = [{'source_frame': f.frame_index, 'image': str(f.image_path),
                                 'mask': str(f.mask_path), 'image_sha256': digest(f.image_path),
                                 'mask_sha256': digest(f.mask_path)} for f in frames]
            directory = Path(temp) / name
            directory.mkdir()
            state, prompt = prepare(model, frames, directory)
            if prompt != 0:
                raise RuntimeError('Selected clip does not begin with the fixed prompt')
            with torch.inference_mode():
                model.propagate_in_video_preflight(state)
            bank = clone_tensors(state['output_dict_per_obj'][0])
            samples = _samples(model, state, frames, range(1, 5))
            native = {}
            with torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1]:
                            raise RuntimeError('Native fixture object IDs changed')
                        if frame:
                            value = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                            native[frame] = clone_tensors({k: value[k] for k in MEMORY_FIELDS})
                finally:
                    iterator.close()
                model.reset_state(state)
            cache[name] = {'bank': bank, 'samples': samples, 'bank_sha256': _tree_digest(bank),
                           'native': native}
        atomic_json(out / 'inputs.json', input_rows)
        for frames in input_rows.values():
            for frame in frames:
                if (digest(Path(frame['image'])) != frame['image_sha256']
                        or digest(Path(frame['mask'])) != frame['mask_sha256']):
                    raise RuntimeError('Selected training inputs changed during feature capture')
        arms, initial_hash = {}, None
        for mode in ('token', 'global'):
            heads = _make_heads(model, protocol)
            if initial_hash is None:
                initial_hash = _tree_digest(heads.state_dict())
            if initial_hash != _tree_digest(heads.state_dict()):
                raise RuntimeError('Arms do not share adapter initialization')
            optimizer = _adamw(heads, protocol)
            baseline = evaluate(model, heads, mode, cache)
            _assert_equal(baseline, evaluate(model, heads, mode, cache), 'repeat.fixed_forward')
            for name, item in cache.items():
                with torch.no_grad(), NativeTokenConditionedKeyAdapter(model, heads, mode):
                    _, predicted, _ = reader_clip_loss(model, item['samples'], item['bank'])
                for frame in range(1, 5):
                    for field in MEMORY_FIELDS:
                        _assert_equal(predicted['non_cond_frame_outputs'][frame][field],
                                      item['native'][frame][field], f'parity.{mode}.{name}.{frame}.{field}')
            rows = [{'step': 0, 'clips': baseline}]
            limit = 2 if args.gate_only else 32
            updates = []
            for step in range(1, limit + 1):
                _guard_sources(sources)
                if digest(DESIGN) != identity['design_sha256']:
                    raise RuntimeError('Frozen diagnostic design changed during the run')
                name = names[(step - 1) % len(names)]
                if step == (2 if args.gate_only else 17):
                    heads, optimizer, result = replay_update(model, heads, optimizer, cache[name],
                        name, mode, protocol, out / f'{mode}_resume_checkpoint.pt')
                else:
                    result = update(model, heads, optimizer, cache[name], name, mode, protocol)
                updates.append({'step': step, 'sequence': name, 'loss': result[0],
                                'gradient_l1_by_layer': result[3]['gradient_l1_by_layer']})
                if step % 8 == 0 or args.gate_only:
                    rows.append({'step': step, 'clips': evaluate(model, heads, mode, cache)})
                atomic_json(out / 'progress.json', {'phase': 'gate' if args.gate_only else 'training_diagnostic',
                            'mode': mode, 'step': step, 'total': limit})
                print(json.dumps({'mode': mode, 'step': step, 'loss': result[0]}), flush=True)
                atomic_json(out / f'{mode}_curves.json', {'evaluations': rows, 'updates': updates})
            if _model_digest(model) != original_hash:
                raise RuntimeError('SAM2 parameters/buffers changed')
            atomic_torch_save(out / f'{mode}_final.pt', heads.state_dict())
            final = rows[-1]['clips']
            ratio = statistics.mean(x['loss'] for x in final.values()) / statistics.mean(x['loss'] for x in baseline.values())
            arms[mode] = {'evaluations': rows, 'updates': updates,
                          'parameter_count': sum(x.numel() for x in heads.parameters()),
                          'final_mean_loss_ratio': ratio,
                          'optimization_signal': ratio <= 0.99 and all(
                              final[n]['loss'] <= baseline[n]['loss'] * 1.02 for n in names),
                          'exact_serialized_update_replay': True}
    _guard_sources(sources)
    result = {'passed': True, 'scope': 'engineering gate' if args.gate_only else 'TRAIN-only optimization diagnostic',
              'identity': identity, 'arms': arms, 'sam2_sha256': original_hash,
              'elapsed_seconds': time.monotonic() - started,
              'efficacy_run_authorized': False, 'test_accessed': False}
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'completion.json', {'passed': True, 'results_sha256': digest(out / 'results.json'),
                                         'scope': result['scope']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    existed_before = args.output_dir.exists()
    try:
        result = run(args)
        print(json.dumps({'passed': result['passed'], 'scope': result['scope']}), flush=True)
    except Exception as exc:
        if not existed_before and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise

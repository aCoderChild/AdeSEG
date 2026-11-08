"""Diagnose the fixed-clip parity failure without validation or test access.

Restore the diagnostic's serialized step-16 state and reuse its exact training
helpers for steps 17--32. Capture tensor differences rather than relaxing parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.video import load_manifest
from modeling.complementary_memory.training import clone_tensors, reader_clip_loss
from modeling.token_conditioned_key_adapter import NativeTokenConditionedKeyAdapter
from scripts.diagnose_key_adapter_optimization import (
    DRIVE, MEMORY_FIELDS, fixed_clip, update,
)
from scripts.mavos_value_utils import activate_author_repo, atomic_json, atomic_torch_save, prepare
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources,
    _make_heads, _model_digest, _restore_rng, _source_hashes, _tree_digest,
)
from scripts.verify_token_conditioned_key_adapter_runner import _adamw, _samples

FAILED = DRIVE / 'key_adapter_optimization_validity_v1/attempt_02_fixed_training'


def tensor_difference(left, right):
    # Native maskmem_pos_enc is a list of tensors; retain its structure.
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return {'equal': False, 'bitwise_equal': False, 'structure_mismatch': True}
        children = [tensor_difference(a, b) for a, b in zip(left, right)]
        return {'equal': all(x['equal'] for x in children),
                'bitwise_equal': all(x.get('bitwise_equal', False) for x in children),
                'children': children}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {'equal': False, 'bitwise_equal': False,
                'left_shape': list(left.shape), 'right_shape': list(right.shape),
                'left_dtype': str(left.dtype), 'right_dtype': str(right.dtype)}
    delta = (left.float() - right.float()).abs()
    return {'equal': torch.equal(left, right),
            'bitwise_equal': torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)),
            'changed_values': int((left != right).sum()),
            'max_absolute_difference': float(delta.max()),
            'mean_absolute_difference': float(delta.mean()),
            'positive_value_disagreements': int(((left > 0) != (right > 0)).sum())}


def compare_banks(left, right):
    return {str(frame): {field: tensor_difference(left['non_cond_frame_outputs'][frame][field],
                        right['non_cond_frame_outputs'][frame][field]) for field in MEMORY_FIELDS}
            for frame in range(1, 5)}


def latent_state(model):
    caches = {}
    for name, module in model.named_modules():
        for attribute in ('freqs_cis', 'cache'):
            value = getattr(module, attribute, None)
            if isinstance(value, torch.Tensor) or (isinstance(value, dict) and value):
                caches[f'{name}.{attribute}'] = _tree_digest(value)
    return {'training_flags': {n: m.training for n, m in model.named_modules()},
            'forward_hook_counts': {n: [len(m._forward_hooks), len(m._forward_pre_hooks)]
                                   for n, m in model.named_modules()
                                   if m._forward_hooks or m._forward_pre_hooks},
            'selected_cache_digests': caches}


def forward(model, item, heads=None, mode='global'):
    with torch.no_grad():
        if heads is None:
            loss, bank, _ = reader_clip_loss(model, item['samples'], item['bank'])
            calls = []
        else:
            with NativeTokenConditionedKeyAdapter(model, heads, mode) as observer:
                loss, bank, _ = reader_clip_loss(model, item['samples'], item['bank'])
            calls = observer.calls
    return bank, {'loss': float(loss), 'calls': calls}


def run(out, threads=2):
    out = validate_output(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    sources = _source_hashes()
    for relative in ('scripts/diagnose_key_adapter_optimization.py', 'scripts/trace_key_adapter_parity.py',
                     'tests/test_key_adapter_parity_trace.py'):
        sources[relative] = digest(ROOT / relative)
    snapshot_path = FAILED / 'token_resume_checkpoint.pt'
    final_path = FAILED / 'token_final.pt'
    identity = {'scope': 'engineering trace; never efficacy', 'sources': sources,
                'torch_version': torch.__version__, 'python_version': sys.version,
                'torch_num_threads': threads, 'torch_build': torch.__config__.show(),
                'checkpoint_sha256': digest(CHECKPOINT_DEFAULT), 'protocol_sha256': digest(PROTOCOL),
                'resume_sha256': digest(snapshot_path), 'final_adapter_sha256': digest(final_path),
                'schedule': 'Restore saved state before update 17; replay updates 17--32 on fixed seq2/seq3 train clips; measure zero-adapter parity before/after. No validation or test.'}
    if identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Pretrained checkpoint identity differs')
    atomic_json(out / 'frozen_trace.json', identity)
    for relative in sources:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(protocol['training']['seed']))
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('Pinned SAM2 revision changed')
    model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
        checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
    model.requires_grad_(False).eval()
    core_digest = _model_digest(model)
    cache = {}
    videos = load_manifest(MANIFEST, 'dev')
    names = ['seq2', 'seq3']
    if not set(names) <= set(protocol['data']['train_sequences']):
        raise RuntimeError('Trace cannot access non-training sequences')
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        for name in names:
            frames = fixed_clip(videos[name])
            directory = Path(temporary) / name
            directory.mkdir()
            state, prompt = prepare(model, frames, directory)
            if prompt != 0:
                raise RuntimeError('Fixed prompt changed')
            with torch.inference_mode():
                model.propagate_in_video_preflight(state)
            bank = clone_tensors(state['output_dict_per_obj'][0])
            samples = _samples(model, state, frames, range(1, 5))
            video_reference = {'non_cond_frame_outputs': {}}
            with torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1]:
                            raise RuntimeError('Native reference changed object ID')
                        if frame:
                            value = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                            video_reference['non_cond_frame_outputs'][frame] = clone_tensors(
                                {field: value[field] for field in MEMORY_FIELDS})
                finally:
                    iterator.close()
                model.reset_state(state)
            cache[name] = {'bank': bank, 'samples': samples, 'bank_sha256': _tree_digest(bank),
                           'video_reference': video_reference,
                           'source_inputs': [{'source_frame': f.frame_index,
                                              'image_sha256': digest(f.image_path),
                                              'mask_sha256': digest(f.mask_path)} for f in frames]}
    inputs_before = {name: _tree_digest(item['samples']) for name, item in cache.items()}
    native_before, native_log = forward(model, cache['seq2'])
    zero_heads = _make_heads(model, protocol)
    zero_before, zero_log = forward(model, cache['seq2'], zero_heads)
    report = {'identity': identity, 'native_before': native_log, 'zero_before': zero_log,
              'zero_before_vs_native': compare_banks(zero_before, native_before),
              'native_before_vs_video_reference': compare_banks(native_before, cache['seq2']['video_reference']),
              'source_inputs': {name: item['source_inputs'] for name, item in cache.items()},
              'inputs_before': inputs_before, 'latent_before': latent_state(model), 'updates': []}
    atomic_json(out / 'trace.json', report)
    if not all(row['equal'] for frame in report['zero_before_vs_native'].values() for row in frame.values()):
        raise RuntimeError('Fresh zero adapter already differs; inspect captured differences')
    if not all(row['equal'] for frame in report['native_before_vs_video_reference'].values()
               for row in frame.values()):
        raise RuntimeError('Fresh native cached-feature replay differs from video propagation')
    saved = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    if saved['bank_sha256'] != cache['seq2']['bank_sha256']:
        raise RuntimeError('Restored fixed bank identity differs')
    heads = _make_heads(model, protocol)
    heads.load_state_dict(saved['heads'], strict=True)
    optimizer = _adamw(heads, protocol)
    optimizer.load_state_dict(saved['optimizer'])
    _restore_rng(saved['rng'])
    for step in range(17, 33):
        _guard_sources(sources)
        name = names[(step - 1) % 2]
        loss, _, _, diagnostics = update(model, heads, optimizer, cache[name], name, 'token', protocol)
        current_inputs = {n: _tree_digest(item['samples']) for n, item in cache.items()}
        report['updates'].append({'step': step, 'sequence': name, 'loss': loss,
                                  'input_hashes_unchanged': current_inputs == inputs_before,
                                  'gradient_l1_by_layer': diagnostics['gradient_l1_by_layer']})
        atomic_json(out / 'trace.json', report)
        print(json.dumps(report['updates'][-1]), flush=True)
        if current_inputs != inputs_before:
            raise RuntimeError('Fixed features/targets changed during optimization')
    original_final = torch.load(final_path, map_location='cpu', weights_only=True)
    report['restored_final_adapter_equals_original'] = {
        name: tensor_difference(value, original_final[name]) for name, value in heads.state_dict().items()}
    native_after, after_native_log = forward(model, cache['seq2'])
    zero_heads_after = _make_heads(model, protocol)
    zero_after, after_zero_log = forward(model, cache['seq2'], zero_heads_after)
    report.update({'native_after': after_native_log, 'zero_after': after_zero_log,
                   'native_after_vs_before': compare_banks(native_after, native_before),
                   'zero_after_vs_native_before': compare_banks(zero_after, native_before),
                   'zero_after_vs_video_reference': compare_banks(zero_after, cache['seq2']['video_reference']),
                   'zero_after_vs_native_after': compare_banks(zero_after, native_after),
                   'latent_after': latent_state(model),
                   'sam2_digest_unchanged': _model_digest(model) == core_digest,
                   'replayed_adapter_sha256': _tree_digest(heads.state_dict()),
                   'replayed_optimizer_sha256': _tree_digest(optimizer.state_dict()),
                   'test_accessed': False})
    atomic_torch_save(out / 'replayed_state.pt', {'heads': heads.state_dict(),
                                                'optimizer': optimizer.state_dict()})
    _guard_sources(sources)
    atomic_json(out / 'trace.json', report)
    atomic_json(out / 'completion.json', {'trace_complete': True, 'efficacy_authorized': False,
                                         'trace_sha256': digest(out / 'trace.json')})
    print(json.dumps({'trace_complete': True, 'efficacy_authorized': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--threads', choices=(1, 2), type=int, default=2)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        run(args.output_dir, threads=args.threads)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise

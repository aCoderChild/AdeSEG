"""TRAIN-only uninterrupted/fresh-process comparison using existing SAM2 helpers."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.video import load_manifest
from modeling.complementary_memory.training import clone_tensors
from scripts.diagnose_key_adapter_optimization import MEMORY_FIELDS, evaluate, fixed_clip, update
from scripts.mavos_value_utils import activate_author_repo, atomic_json, atomic_torch_save, prepare
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _capture_rng, _guard_sources,
    _make_heads, _model_digest, _restore_rng, _source_hashes, _tree_digest,
)
from scripts.trace_key_adapter_parity import compare_banks, forward, latent_state
from scripts.verify_token_conditioned_key_adapter_runner import _adamw, _assert_equal, _samples

DESIGN = ROOT / 'protocols/key_adapter_process_resume_v1.json'


def require_reference(path, identity):
    completion = json.loads((path / 'completion.json').read_text())
    results_path = path / 'results.json'
    result = json.loads(results_path.read_text())
    if (not completion.get('passed') or completion.get('results_sha256') != digest(results_path)
            or result.get('phase') != 'reference' or result.get('identity') != identity
            or result.get('efficacy_authorized') is not False):
        raise RuntimeError('Reference incomplete, altered, unauthorized or from different sources/runtime')
    return result


def require_parity(model, cache, protocol, mode):
    evidence = {}
    heads = _make_heads(model, protocol)
    for name, item in cache.items():
        native, _ = forward(model, item)
        zero, _ = forward(model, item, heads, mode)
        evidence[name] = {'native': compare_banks(native, item['native']),
                          'zero': compare_banks(zero, item['native'])}
    if not all(v['bitwise_equal'] for clip in evidence.values()
               for arm in clip.values() for frame in arm.values() for v in frame.values()):
        raise RuntimeError('Native/zero parity differs: ' + json.dumps(evidence))
    return evidence


def fixture(model, names, out):
    cache, files = {}, {}
    videos = load_manifest(MANIFEST, 'dev')
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
            native = {'non_cond_frame_outputs': {}}
            with torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1]:
                            raise RuntimeError('Native fixture object ID differs')
                        if frame:
                            value = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                            native['non_cond_frame_outputs'][frame] = clone_tensors(
                                {field: value[field] for field in MEMORY_FIELDS})
                finally:
                    iterator.close()
                model.reset_state(state)
            cache[name] = {'bank': bank, 'samples': samples, 'native': native,
                           'bank_sha256': _tree_digest(bank)}
            files[name] = [{'frame': f.frame_index, 'image_sha256': digest(f.image_path),
                            'mask_sha256': digest(f.mask_path)} for f in frames]
    return cache, files


def run(args):
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    design, protocol = json.loads(DESIGN.read_text()), json.loads(PROTOCOL.read_text())
    names = design['data']['sequences']
    if names != ['seq2', 'seq3'] or not set(names) <= set(protocol['data']['train_sequences']):
        raise RuntimeError('Only frozen training fixtures permitted')
    if design['data']['validation_access'] or design['data']['test_access']:
        raise RuntimeError('No validation/test access permitted')
    sources = _source_hashes()
    for relative in ('scripts/diagnose_key_adapter_optimization.py',
                     'scripts/trace_key_adapter_parity.py',
                     'scripts/verify_key_adapter_process_resume.py',
                     'tests/test_key_adapter_process_resume.py',
                     'protocols/key_adapter_process_resume_v1.json'):
        sources[relative] = digest(ROOT / relative)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(protocol['training']['seed']))
    identity = {'sources': sources, 'manifest_sha256': digest(MANIFEST),
                'protocol_sha256': digest(PROTOCOL), 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
                'torch_version': torch.__version__, 'python': sys.version,
                'torch_build': torch.__config__.show(), 'threads': torch.get_num_threads()}
    if identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Pretrained checkpoint changed')
    reference = require_reference(args.reference, identity) if args.phase == 'resume' else None
    atomic_json(out / 'identity.json', identity)
    atomic_json(out / 'design.json', design)
    for relative in sources:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('SAM2 revision changed')
    model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
        checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
    model.requires_grad_(False).eval()
    core_hash = _model_digest(model)
    cache, files = fixture(model, names, out)
    fixed_hashes = {name: _tree_digest({'bank': v['bank'], 'samples': v['samples']})
                    for name, v in cache.items()}
    result = {'phase': args.phase, 'identity': identity, 'source_files': files,
              'fixed_hashes': fixed_hashes, 'arms': {}, 'efficacy_authorized': False,
              'validation_accessed': False, 'test_accessed': False}
    if reference:
        _assert_equal(files, reference['source_files'], 'resume.source_files')
        _assert_equal(fixed_hashes, reference['fixed_hashes'], 'resume.fixed_inputs')
    for mode in design['schedule']['arms']:
        parity_before = require_parity(model, cache, protocol, mode)
        heads = _make_heads(model, protocol)
        optimizer = _adamw(heads, protocol)
        rows, updates = [], []
        if reference:
            checkpoint_path = args.reference / f'{mode}_step16.pt'
            if digest(checkpoint_path) != reference['arms'][mode]['step16_sha256']:
                raise RuntimeError('Resume checkpoint altered')
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            heads.load_state_dict(saved['heads'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step16 = evaluate(model, heads, mode, cache)
            _assert_equal(step16, saved['evaluation'], f'{mode}.resume.step16_evaluation')
            _restore_rng(saved['rng'])
            first = 17
        else:
            baseline = evaluate(model, heads, mode, cache)
            _assert_equal(baseline, evaluate(model, heads, mode, cache), 'initial.repeat')
            rows.append({'step': 0, 'clips': baseline})
            first = 1
        for step in range(first, 33):
            _guard_sources(sources)
            name = names[(step - 1) % 2]
            loss, bank, writes, diagnostics = update(model, heads, optimizer, cache[name], name, mode, protocol)
            row = {'step': step, 'sequence': name, 'loss': loss, 'bank_sha256': _tree_digest(bank),
                   'writes': writes, 'diagnostics': diagnostics}
            updates.append(row)
            if reference:
                _assert_equal(row, reference['arms'][mode]['updates'][step - 1], f'{mode}.step{step}')
            if fixed_hashes != {n: _tree_digest({'bank': v['bank'], 'samples': v['samples']}) for n, v in cache.items()}:
                raise RuntimeError('Fixed fixture mutated')
            if step % 8 == 0:
                evaluation = evaluate(model, heads, mode, cache)
                rows.append({'step': step, 'clips': evaluation})
                if reference:
                    expected = next(v for v in reference['arms'][mode]['evaluations'] if v['step'] == step)
                    _assert_equal(rows[-1], expected, f'{mode}.evaluation{step}')
                elif step == 16:
                    atomic_torch_save(out / f'{mode}_step16.pt', {'heads': copy.deepcopy(heads.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': _capture_rng(),
                        'evaluation': evaluation})
            atomic_json(out / f'{mode}_progress.json', {'updates': updates, 'evaluations': rows})
            print(json.dumps({'phase': args.phase, 'mode': mode, 'step': step, 'loss': loss}), flush=True)
        final = {'heads': heads.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _capture_rng()}
        atomic_torch_save(out / f'{mode}_final.pt', final)
        final_hash = _tree_digest(final)
        if reference:
            if final_hash != reference['arms'][mode]['final_state_sha256']:
                raise RuntimeError(f'{mode} final adapter/optimizer/RNG differs')
        parity_after = require_parity(model, cache, protocol, mode)
        if _model_digest(model) != core_hash:
            raise RuntimeError('SAM2 weights changed')
        arm = {'updates': updates, 'evaluations': rows, 'final_state_sha256': final_hash,
               'parity_before': parity_before, 'parity_after': parity_after,
               'final_latent': latent_state(model)}
        if reference:
            _assert_equal(arm['final_latent'], reference['arms'][mode]['final_latent'], f'{mode}.latent')
        if not reference:
            arm['step16_sha256'] = digest(out / f'{mode}_step16.pt')
            initial, final_eval = rows[0]['clips'], rows[-1]['clips']
            ratio = statistics.mean(v['loss'] for v in final_eval.values()) / statistics.mean(v['loss'] for v in initial.values())
            arm['final_mean_loss_ratio'] = ratio
            arm['train_optimization_signal'] = ratio <= .99 and all(final_eval[n]['loss'] <= initial[n]['loss'] * 1.02 for n in names)
        result['arms'][mode] = arm
        atomic_json(out / 'results.json', result)
    _guard_sources(sources)
    atomic_json(out / 'completion.json', {'passed': True, 'efficacy_authorized': False,
                                         'results_sha256': digest(out / 'results.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('reference', 'resume'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if (args.phase == 'resume') != (args.reference is not None):
        parser.error('--reference is required only for resume')
    existed = args.output_dir.exists()
    try:
        run(args)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise

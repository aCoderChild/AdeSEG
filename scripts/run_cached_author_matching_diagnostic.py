"""Replay pinned native banks for the frozen broader causal matching study."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.native_history_cache import cached_read, load_bank
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_author_matching_diagnostic import test_read
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write


def verify_required_gate(path, field, sources):
    receipt = json.loads(path.read_text())
    if not receipt.get(field):
        raise RuntimeError(f'Required engineering gate failed: {path}')
    if sources is not None:
        setup = json.loads((path.parent / 'setup.json').read_text())
        for name in sources:
            if setup['sources'].get(name) != digest(ROOT / name):
                raise RuntimeError(f'Gate source differs for {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-gate', action='store_true')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        plan = json.loads(args.plan.read_text())
        protocol = plan['protocol']
        for name, sha in plan['source_hashes'].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError(f'Frozen diagnostic source changed: {name}')
        if digest(Path(protocol['checkpoint'])) != protocol['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        manifest = ROOT / protocol['manifest']
        if digest(manifest) != protocol['manifest_sha256']:
            raise RuntimeError('Manifest mismatch')
        verify_required_gate(Path(protocol['id_gate']), 'passed',
                             ['modeling/region_constrained_tome.py', 'modeling/author_token_matching.py'])
        verify_required_gate(Path(protocol['matching_replay_gate']), 'passed', None)
        if not args.cache_gate:
            if args.gate is None:
                raise RuntimeError('Broader diagnostic requires the cached-native engineering gate')
            gate = json.loads((args.gate / 'completion.json').read_text())
            if not gate.get('passed') or gate.get('scope') != 'cache_gate' or gate['plan_sha256'] != digest(args.plan):
                raise RuntimeError('Cache gate is missing, failed or belongs to a different plan')
        torch.set_num_threads(2)
        torch.manual_seed(20261111)
        torch.use_deterministic_algorithms(True)
        repo = ROOT / protocol['sam2_repository']
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != protocol['sam2_revision']:
            raise RuntimeError('SAM2 revision mismatch')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
            raise RuntimeError('SAM2 reference checkout changed')
        if 'sam2' in sys.modules:
            raise RuntimeError('SAM2 imported before pinned source activation')
        sys.path.insert(0, str(repo))
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        sequences = protocol['cache_gate_sequences'] if args.cache_gate else protocol['sequences']
        write(out / 'setup.json', {'plan': str(args.plan), 'plan_sha256': digest(args.plan),
              'scope': 'cache_gate' if args.cache_gate else 'broader_causal_diagnostic',
              'sources': plan['source_hashes'], 'checkpoint_sha256': protocol['checkpoint_sha256']})
        reference = json.loads((Path(protocol['matching_reference']) / 'results.json').read_text()) if args.cache_gate else None
        results = {}
        for sequence in sequences:
            specification = plan['sequences'][sequence]
            cache_folder = Path(specification['cache_directory'])
            receipt_path = cache_folder / 'receipt.json'
            if digest(receipt_path) != specification['receipt_sha256']:
                raise RuntimeError(f'Native bank receipt changed: {sequence}')
            receipt = json.loads(receipt_path.read_text())
            payload = load_bank(cache_folder / 'bank.pt', receipt,
                checkpoint_sha256=protocol['checkpoint_sha256'],
                manifest_sha256=protocol['manifest_sha256'], model_config=protocol['model_config'])
            original_payload_sha = fingerprint(payload)['sha256']
            frames = videos[sequence]
            if payload['num_frames'] != len(frames):
                raise RuntimeError('Cached native video coverage differs')
            selected = ([payload['prompt']+offset for offset in protocol['cache_gate_offsets']]
                        if args.cache_gate else [row['frame'] for row in specification['schedule']['targets']])
            folder = out / sequence
            folder.mkdir()
            model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
                checkpoint=Path(protocol['checkpoint']), model_config=protocol['model_config']), repo)
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
                raise RuntimeError('Native memory schedule or postprocessing differs')
            weights_before = parameter_digest(model)
            results[sequence] = []
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                state, prompt = prepare(model, frames, Path(temporary))
                # add_new_mask leaves a temporary prompt without encoded
                # memory. Use the native consolidation/memory-writing stage
                # before comparing it with the archived conditioning frame.
                model.propagate_in_video_preflight(state)
                if prompt != payload['prompt']:
                    raise RuntimeError('Current and archived prompts differ')
                index = state['obj_id_to_idx'][1]
                original = state['output_dict_per_obj'][index]
                current_prompt = original['cond_frame_outputs'][prompt]
                cached_prompt = payload['outputs']['cond_frame_outputs'][prompt]
                for field in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                    if not torch.equal(current_prompt[field], cached_prompt[field]):
                        raise RuntimeError(f'Prompt encoding differs from native cache: {sequence}/{field}')
                for frame in selected:
                    write(out / 'progress.json', {'sequence': sequence, 'frame': frame,
                          'completed_in_sequence': len(results[sequence]), 'targets_in_sequence': len(selected),
                          'elapsed_seconds': time.monotonic()-started})
                    with torch.inference_mode():
                        sample = cached_read(model, state, frame, payload)
                        with NativeReadCapture(model) as capture:
                            current = model.track_step(frame_idx=frame, is_init_cond_frame=False,
                                current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                                feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                                output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                                track_in_reverse=False, run_mem_encoder=False)
                        if not torch.equal(current['pred_masks'], sample['native_logits']):
                            raise RuntimeError(f'Cached native mask differs on replay: {sequence}/{frame}')
                        if not torch.equal(current['object_score_logits'], sample['native_object_score']):
                            raise RuntimeError(f'Cached native presence score differs: {sequence}/{frame}')
                        _, native_logits = model._get_orig_video_res_output(state, current['pred_masks'])
                    state['output_dict_per_obj'][index] = sample['output_dict']
                    try:
                        record = test_read(model, state, capture, native_logits, frames, frame,
                                           protocol['conditions'], folder)
                    finally:
                        state['output_dict_per_obj'][index] = original
                    if args.cache_gate:
                        expected = next(row for row in reference[sequence] if row['frame'] == frame)
                        if record['attention_audit'] != expected['attention_audit']:
                            raise RuntimeError('Cached and fresh-native token budgets differ')
                        for condition, row in record['rows'].items():
                            observed = {k: v for k, v in row.items() if k != 'seconds'}
                            prior = {k: v for k, v in expected['rows'][condition].items() if k != 'seconds'}
                            if observed != prior:
                                raise RuntimeError(f'Cached/fresh-native intervention differs: {sequence}/{frame}/{condition}')
                    results[sequence].append(record)
                    write(out / 'results.json', results)
                    del sample, current, capture, record
                model.reset_state(state)
            if parameter_digest(model) != weights_before or fingerprint(payload)['sha256'] != original_payload_sha:
                raise RuntimeError('Native parameters or archived bank changed')
            if [r['frame'] for r in results[sequence]] != selected:
                raise RuntimeError('Incomplete causal-diagnostic coverage')
            del model, state, payload
            gc.collect()
        if any(digest(ROOT / name) != sha for name, sha in plan['source_hashes'].items()):
            raise RuntimeError('Frozen diagnostic source changed during execution')
        write(out / 'completion.json', {'passed': True, 'scope': 'cache_gate' if args.cache_gate else 'broader_causal_diagnostic',
              'plan_sha256': digest(args.plan), 'results_sha256': digest(out / 'results.json'),
              'native_cache_replay_exact': True, 'unchanged_parameters_and_banks': True,
              'elapsed_seconds': time.monotonic()-started})
        write(out / 'progress.json', {'phase': 'completed', 'sequences': len(sequences)})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()

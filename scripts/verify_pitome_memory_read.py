"""Engineering-only real-frame PiToMe/ToMe replay on archived native histories."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.author_token_matching import use_matching
from modeling.native_history_cache import cached_read, load_bank
from modeling.pitome_matching import use_pitome, author_pitome, PITOME_SOURCE
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts import run_author_matching_diagnostic as reader
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        protocol_path = ROOT / 'protocols/pitome_protection_engineering_v1.json'
        protocol = json.loads(protocol_path.read_text())
        if protocol['merge_divisor'] != 4 or protocol['margin'] != .5 or protocol['alpha'] != 1.:
            raise ValueError('Engineering budget or upstream defaults changed')
        hashes = {name: digest(ROOT / name) for name in protocol['source_files']}
        repository = Path(protocol['pitome_repository'])
        author_pitome(repository)
        if digest(repository / PITOME_SOURCE) != protocol['pitome_source_sha256']:
            raise ValueError('PiToMe source hash differs')
        checkpoint = Path(protocol['checkpoint'])
        manifest = ROOT / protocol['manifest']
        if digest(checkpoint) != protocol['checkpoint_sha256'] or digest(manifest) != protocol['manifest_sha256']:
            raise ValueError('Checkpoint/manifest mismatch')
        repo = ROOT / protocol['sam2_repository']
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != protocol['sam2_revision']:
            raise ValueError('SAM2 revision mismatch')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
            raise ValueError('SAM2 reference checkout changed')
        if 'sam2' in sys.modules:
            raise RuntimeError('SAM2 imported before source activation')
        sys.path.insert(0, str(repo))
        torch.set_num_threads(2)
        torch.manual_seed(protocol['seed'])
        torch.use_deterministic_algorithms(True)
        write(out / 'setup.json', {'protocol': protocol, 'protocol_sha256': digest(protocol_path), 'sources': hashes})
        plan = json.loads(Path(protocol['native_history_plan']).read_text())
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        results = {}

        def manager_factory(*args, **kwargs):
            return NativeRegionConstrainedToMe(*args, **kwargs, merge_divisor=4)

        def matching_context(strategy):
            return use_pitome(repository) if strategy == 'pitome' else use_matching(strategy)

        for name in protocol['sequences']:
            specification = plan['sequences'][name]
            cache = Path(specification['cache_directory'])
            if digest(cache / 'receipt.json') != specification['receipt_sha256']:
                raise ValueError('Native cache receipt changed')
            payload = load_bank(cache / 'bank.pt', json.loads((cache / 'receipt.json').read_text()),
                checkpoint_sha256=protocol['checkpoint_sha256'], manifest_sha256=protocol['manifest_sha256'],
                model_config=protocol['model_config'])
            bank_sha = fingerprint(payload)['sha256']
            model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
                checkpoint=checkpoint, model_config=protocol['model_config']), repo)
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
                raise ValueError('Native memory schedule/postprocessing differs')
            weights_sha = parameter_digest(model)
            frames = videos[name]
            folder = out / name
            folder.mkdir()
            results[name] = []
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                state, prompt = prepare(model, frames, Path(temporary))
                model.propagate_in_video_preflight(state)
                if prompt != payload['prompt'] or len(frames) != payload['num_frames']:
                    raise ValueError('Native prompt or full-video metadata differs')
                index = state['obj_id_to_idx'][payload['object_id']]
                original = state['output_dict_per_obj'][index]
                for field in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                    if not torch.equal(original['cond_frame_outputs'][prompt][field],
                                       payload['outputs']['cond_frame_outputs'][prompt][field]):
                        raise ValueError('Native prompt encoding differs')
                for offset in protocol['target_offsets']:
                    frame = prompt + offset
                    write(out / 'progress.json', {'sequence': name, 'frame': frame,
                        'completed_targets': sum(map(len, results.values())), 'total_targets': 6})
                    with torch.inference_mode():
                        sample = cached_read(model, state, frame, payload)
                        with NativeReadCapture(model) as capture:
                            native = model.track_step(frame_idx=frame, is_init_cond_frame=False,
                                current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                                feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                                output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                                track_in_reverse=False, run_mem_encoder=False)
                        if not torch.equal(native['pred_masks'], sample['native_logits']) or not torch.equal(
                                native['object_score_logits'], sample['native_object_score']):
                            raise ValueError('Archived native output differs')
                        _, native_logits = model._get_orig_video_res_output(state, native['pred_masks'])
                    state['output_dict_per_obj'][index] = sample['output_dict']
                    try:
                        roles, spans = reader.memory_role_sequence(model, frame, sample['output_dict'], len(frames))
                        noop = NativeRegionConstrainedToMe(model, merge_divisor=4097)
                        noop.frame, noop.memberships, noop.spans = frame, roles, spans
                        with noop:
                            unchanged, presence = reader.decode(model, state, capture)
                        if not torch.equal(unchanged, native_logits) or not torch.equal(presence, sample['native_object_score']):
                            raise ValueError('Zero-removal parity failed')
                        if [c['layer'] for c in noop.calls] != [0, 1, 2, 3] or any(
                                c['spatial_keys_before'] != c['spatial_keys_after'] for c in noop.calls):
                            raise ValueError('Zero-removal path skipped a layer or removed tokens')
                        with patch.object(reader, 'NativeRegionConstrainedToMe', manager_factory), \
                             patch.object(reader, 'use_matching', matching_context):
                            record = reader.test_read(model, state, capture, native_logits, frames, frame,
                                                      protocol['conditions'], folder)
                        record['zero_removal_exact'] = True
                        results[name].append(record)
                        write(out / 'results.json', results)
                    finally:
                        state['output_dict_per_obj'][index] = original
                    del sample, capture, native, record
                model.reset_state(state)
            if parameter_digest(model) != weights_sha or fingerprint(payload)['sha256'] != bank_sha:
                raise ValueError('Native weights or bank changed')
            del model, state, payload
            gc.collect()
        if any(digest(ROOT / name) != sha for name, sha in hashes.items()) or digest(protocol_path) != json.loads((out / 'setup.json').read_text())['protocol_sha256']:
            raise ValueError('Engineering source/protocol changed during execution')
        write(out / 'completion.json', {'engineering_passed': True, 'targets': 6,
            'scope': 'Engineering only; no efficacy promotion', 'zero_removal_exact': True,
            'native_archive_replay_exact': True, 'parameters_and_banks_unchanged': True,
            'all_four_layers_and_matched_budgets': True, 'results_sha256': digest(out / 'results.json')})
        write(out / 'progress.json', {'phase': 'completed', 'targets': 6})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()

"""Fixed-native-trajectory 2x2 matching diagnostic; no altered memory writes."""
import argparse
from contextlib import nullcontext
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
from modeling.author_token_matching import use_matching
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame


@torch.inference_mode()
def decode(model, state, capture):
    conditioned = model.memory_attention(**capture.memory)
    image = conditioned.permute(1, 2, 0).reshape_as(capture.decoder['image_embeddings'])
    # Reuse the author's mask selection, presence handling, interpolation and
    # object-pointer calculation instead of reproducing those decisions here.
    sam = model._forward_sam_heads(backbone_features=image, point_inputs=None,
        mask_inputs=None, high_res_features=capture.decoder['high_res_features'],
        multimask_output=capture.decoder['multimask_output'])
    # The video predictor exports selected LOW-resolution masks directly to
    # video resolution. Going through the 1024 mask adds an interpolation and
    # changes logits and some contour pixels even for the native condition.
    _, logits = model._get_orig_video_res_output(state, sam[3])
    return logits, sam[6]


def test_read(model, state, capture, native_logits, frames, frame, conditions, folder):
    if capture.layers != [0, 1, 2, 3]:
        raise RuntimeError('Native capture missed an attention layer')
    bank_before = fingerprint(state['output_dict_per_obj'])['sha256']
    unchanged, native_score = decode(model, state, capture)
    torch.testing.assert_close(unchanged, native_logits, rtol=1e-6, atol=1e-6)
    roles, spans = memory_role_sequence(model, frame, state['output_dict_per_obj'][0], len(frames))
    rows = {'native': score_frame(native_logits, frames, frame)}
    audits = {}
    expected_budgets = None
    for condition in conditions:
        strategy, mode = condition.split('_')
        manager = NativeRegionConstrainedToMe(model, mode=mode)
        manager.frame, manager.memberships, manager.spans = frame, roles, spans
        started = time.monotonic()
        with use_matching(strategy), manager:
            logits, presence = decode(model, state, capture)
        if not torch.isfinite(logits).all() or not torch.isfinite(presence).all():
            raise RuntimeError('Nonfinite diagnostic output')
        if [call['layer'] for call in manager.calls] != [0, 1, 2, 3]:
            raise RuntimeError('Intervention missed a native memory layer')
        budget = [[(f['frame'], [(w['window'], w['merges']) for w in f['windows']])
                   for f in call['frames']] for call in manager.calls]
        if expected_budgets is None:
            expected_budgets = budget
        elif budget != expected_budgets:
            raise RuntimeError('Four-arm memory budgets or source-frame order differ')
        for call in manager.calls:
            if call['spatial_key_mass_sum'] != call['spatial_keys_before']:
                raise RuntimeError('Spatial token mass changed')
        rows[condition] = {**score_frame(logits, frames, frame),
                           'object_score': float(presence.item()),
                           'seconds': time.monotonic() - started}
        audits[condition] = manager.calls
        torch.save({'logits': logits.cpu(), 'object_score': presence.cpu()},
                   folder / f'{frame}_{condition}.pt')
    if bank_before != fingerprint(state['output_dict_per_obj'])['sha256']:
        raise RuntimeError('A read-only intervention modified the native bank')
    # Verify hooks/patches restored the author's original read, including after
    # the final ATC condition; then the native generator can advance normally.
    restored, _ = decode(model, state, capture)
    torch.testing.assert_close(restored, native_logits, rtol=1e-6, atol=1e-6)
    return {'frame': frame, 'native_parity': True, 'bank_unchanged': True,
            'native_restored': True, 'rows': rows, 'attention_audit': audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / 'protocols/author_matching_diagnostic_v2.json'
    protocol = json.loads(protocol_path.read_text())
    torch.set_num_threads(2)
    torch.manual_seed(20261111)
    torch.use_deterministic_algorithms(True)
    try:
        if digest(Path(protocol['checkpoint'])) != protocol['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        manifest = ROOT / protocol['manifest']
        if digest(manifest) != protocol['manifest_sha256']:
            raise RuntimeError('Manifest mismatch')
        repo = ROOT / protocol['sam2_repository']
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != protocol['sam2_revision']:
            raise RuntimeError('SAM2 revision mismatch')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
            raise RuntimeError('SAM2 checkout changed')
        if 'sam2' in sys.modules:
            raise RuntimeError('SAM2 imported before clean checkout activation')
        sys.path.insert(0, str(repo))
        sources = {p: digest(ROOT / p) for p in protocol['source_files']}
        write(out / 'setup.json', {'protocol': protocol, 'protocol_sha256': digest(protocol_path), 'sources': sources})
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        results = {}
        for name in protocol['sequences']:
            model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
                checkpoint=Path(protocol['checkpoint']), model_config=protocol['model_config']), repo)
            before = parameter_digest(model)
            frames = videos[name]
            prompt = prompt_index(frames, {'initialization': 'first_labeled_frame'})
            targets = {prompt + offset for offset in protocol['target_offsets']}
            if max(targets) >= len(frames):
                raise RuntimeError('Predeclared targets exceed available frames')
            folder = out / name
            folder.mkdir()
            results[name] = []
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                state, _ = prepare(model, frames, Path(temporary))
                trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                # Keep the full video length: SAM2 normalizes pointer time
                # positions by min(num_frames,max_obj_ptrs)-1. Only execution
                # is bounded; truncating the input changes the native model.
                for expected in range(prompt, max(targets) + 1):
                    with NativeReadCapture(model) as capture:
                        frame, ids, native_logits = next(trajectory)
                    if frame != expected or ids != [1]:
                        raise RuntimeError('Native trajectory frame/object order differs')
                    if frame in targets:
                        record = test_read(model, state, capture, native_logits, frames, frame,
                                           protocol['conditions'], folder)
                        results[name].append(record)
                        write(out / 'results.json', results)
                    write(out / 'progress.json', {'sequence': name, 'frame': frame,
                          'completed_reads': len(results[name]), 'time': time.time()})
                trajectory.close()
                model.reset_state(state)
            if parameter_digest(model) != before:
                raise RuntimeError('SAM2 parameters changed')
            if {r['frame'] for r in results[name]} != targets:
                raise RuntimeError('Incomplete target coverage')
            del model
        if any(digest(ROOT / p) != value for p, value in sources.items()):
            raise RuntimeError('Source changed during diagnostic')
        write(out / 'completion.json', {'engineering_passed': True,
              'scope': 'fixed-native-history immediate interventions, not full propagation',
              'results_sha256': digest(out / 'results.json')})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()

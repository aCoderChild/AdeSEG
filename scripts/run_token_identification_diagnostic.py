"""Bounded native-memory identification diagnostic, not an efficacy experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.token_partition_diagnostic import partition_tokens
from modeling.token_diagnostic_replay import NativeReadCapture, replay, orient_partition, ranking_metrics, diagnostic_target_plan
from modeling.prediction_conditioned_affinity import memory_role_sequence
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.memory_token_io import validate_output, digest, parameter_digest, prompt_index
from scripts.run_regional_refinement import write

CAM_REVISION = '9194156183c097b8bb2bd8f945a59c644c2733d9'


def load_cam(model):
    repo = ROOT / 'external/pytorch-grad-cam'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != CAM_REVISION:
        raise RuntimeError('Grad-CAM checkout differs from pin')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'pytorch_grad_cam'],
                   cwd=repo, check=True, capture_output=True)
    sys.path.insert(0, str(repo))
    from pytorch_grad_cam import GradCAM
    # Gradients come from our exact native decoder replay, not a classifier
    # wrapper. Reuse upstream get_cam_image/get_cam_weights without hooks.
    return GradCAM(model=model, target_layers=[])


def evaluate_source(model, frames, state, capture, frame, output, cam):
    actual, leaf = replay(model, capture)
    masks, ious, tokens, object_score = actual
    chosen = int(capture.outputs[1].argmax(dim=1).item())
    support = capture.outputs[0][0, chosen] > 0
    # Fixed native positive support. An empty support is reported, not replaced
    # with ground truth or silently changed to a whole-image target.
    targets = {'object_presence': object_score.sum()}
    if support.any():
        targets['native_mask_support'] = masks[0, chosen][support].mean()
    gradients = {name: torch.autograd.grad(value, leaf, retain_graph=True)[0].detach()
                 for name, value in targets.items()}
    bank = state['output_dict_per_obj'][0]
    roles, spans = memory_role_sequence(model, frame, bank, len(frames))
    pointers = capture.memory['num_obj_ptr_tokens']
    if spans[-1]['stop'] + pointers != leaf.shape[0]:
        raise RuntimeError('Spatial/pointer count mismatch')
    prompt_span = next(s for s in spans if s['t_pos'] == 0)
    prompt = prompt_span['frame']
    # Prompt annotation is the authorised initialization input, not target GT.
    initial = torch.from_numpy((load_label_mask(frames[prompt].mask_path, [1]) == 1).copy()).float()
    initial = F.interpolate(initial[None, None], size=(64, 64), mode='area').flatten()
    prompt_features = leaf.detach()[prompt_span['start']:prompt_span['stop'], 0]
    fg = (prompt_features * initial[:, None]).sum(0) / initial.sum().clamp_min(1e-8)
    bg = (prompt_features * (1-initial[:, None])).sum(0) / (1-initial).sum().clamp_min(1e-8)
    records = []
    for span in spans:
        source = span['frame']
        stored = bank['cond_frame_outputs'].get(source, bank['non_cond_frame_outputs'].get(source))
        features = leaf.detach()[span['start']:span['stop'], 0]
        expected = stored['maskmem_features'][0].flatten(1).T
        if not torch.equal(features, expected):
            raise RuntimeError('Captured memory differs from source-frame bank')
        predicted = F.interpolate(stored['pred_masks'].float().sigmoid(), size=(64, 64), mode='area')[0, 0].numpy()
        # All inference-derived maps are constructed before evaluation GT.
        maps = {'predicted_probability': predicted}
        partition_error = None
        try:
            partition = partition_tokens(features, 64, 64)
            oriented, margin = orient_partition(partition['bipartition'], features, fg, bg)
            maps['prompt_oriented_partition'] = oriented.astype(np.float32)
            maps['partition_eigenvector'] = partition['eigenvector']
        except ValueError as exc:
            partition_error = str(exc)
            margin = None
        activation = features.T.reshape(1, -1, 64, 64).numpy()
        for name, gradient in gradients.items():
            grad = gradient[span['start']:span['stop'], 0].T.reshape(1, -1, 64, 64).numpy()
            if not np.isfinite(grad).all():
                raise RuntimeError('Nonfinite memory gradient')
            signed = cam.get_cam_image(None, None, None, activation, grad)[0]
            maps[name + '_signed_cam'] = signed
            maps[name + '_gradcam'] = np.maximum(signed, 0)
            maps[name + '_gradient_norm'] = np.linalg.norm(grad[0], axis=0)
        target = torch.from_numpy((load_label_mask(frames[source].mask_path, [1]) == 1).copy()).float()
        target = F.interpolate(target[None, None], size=(64, 64), mode='area')[0, 0].numpy() >= 0.5
        metrics = {name: ranking_metrics(value, target) for name, value in maps.items()
                   if name != 'partition_eigenvector'}
        path = output / f'target_{frame}_source_{source}.npz'
        np.savez_compressed(path, **maps, target_for_evaluation=target,
                            memory_features=features.numpy(), predicted_roles=roles[0, span['start']:span['stop']].numpy())
        records.append({'target_frame': frame, 'source_frame': source, 'object_id': 1,
                        'temporal_slot': span['t_pos'], 'partition_error': partition_error,
                        'orientation_margin': margin, 'metrics': metrics,
                        'artifact': str(path), 'sha256': digest(path)})
    return {'frame': frame, 'native_replay_passed': True, 'layers': capture.layers,
            'pointer_tokens': pointers, 'empty_native_support': not bool(support.any()),
            'object_score': float(object_score.detach().item()), 'sources': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'protocols/token_identification_diagnostic_v1.json')
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261111)
    torch.use_deterministic_algorithms(True)
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    checkpoint = Path(protocol['checkpoint'])
    if digest(checkpoint) != protocol['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    manifest = ROOT / protocol['manifest']
    if digest(manifest) != protocol['manifest_sha256']:
        raise RuntimeError('Manifest mismatch')
    sources = {p: digest(ROOT / p) for p in protocol['source_files']}
    write(out / 'setup.json', {'pid': os.getpid(), 'started': time.time(),
                              'protocol': protocol, 'sources': sources,
                              'protocol_sha256': digest(protocol_path)})
    videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    plans = {name: diagnostic_target_plan(
        prompt_index(videos[name], {'initialization': 'first_labeled_frame'}),
        len(videos[name]), protocol['target_offsets'],
        protocol.get('allow_short_sequences', False)) for name in protocol['sequences']}
    write(out / 'target_plan.json', plans)
    repo = ROOT / protocol['sam2_repository']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
        raise RuntimeError('Diagnostic SAM2 checkout is not clean')
    if revision != protocol['sam2_revision']:
        raise RuntimeError('SAM2 revision mismatch')
    if 'sam2' in sys.modules:
        raise RuntimeError('SAM2 imported before pinned repository activation')
    sys.path.insert(0, str(repo))
    records = {}
    try:
        for name in protocol['sequences']:
            model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
                checkpoint=checkpoint, model_config=protocol['model_config']), repo)
            before = parameter_digest(model)
            frames = videos[name]
            prompt = prompt_index(frames, {'initialization': 'first_labeled_frame'})
            # Bound execution, preserving native full-video pointer-time metadata.
            selected = set(plans[name]['frames'])
            folder = out / name
            folder.mkdir()
            records[name] = []
            with tempfile.TemporaryDirectory(dir=folder) as temporary, load_cam(model) as cam:
                state, actual_prompt = prepare(model, frames, Path(temporary))
                if actual_prompt != prompt:
                    raise RuntimeError('Prompt changed after prefix staging')
                trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                for expected_frame in range(prompt, max(selected) + 1):
                    # Remove capture hooks before replay to avoid overwriting
                    # native snapshots and recursively recording replay calls.
                    with NativeReadCapture(model) as capture:
                        frame, ids, logits = next(trajectory)
                    if frame != expected_frame or ids != [1]:
                        raise RuntimeError('Unexpected native trajectory order')
                    write(out / 'progress.json', {'phase': 'native_capture', 'sequence': name, 'frame': frame})
                    if frame in selected:
                        record = evaluate_source(model, frames, state, capture, frame, folder, cam)
                        records[name].append(record)
                        write(out / 'results.json', records)
                        write(out / 'progress.json', {'phase': 'diagnostic_complete', 'sequence': name, 'frame': frame})
                trajectory.close()
                model.reset_state(state)
            if before != parameter_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Frozen SAM2 parameters changed or accumulated gradients')
            if {r['frame'] for r in records[name]} != selected:
                raise RuntimeError('Incomplete diagnostic coverage')
            del model
        if any(digest(ROOT / p) != value for p, value in sources.items()):
            raise RuntimeError('Source changed during diagnostic')
        write(out / 'completion.json', {'engineering_passed': True,
              'efficacy_evaluated': False, 'results_sha256': digest(out / 'results.json'),
              'note': 'Region and attribution diagnostics only; no ToMe Dice gain tested.'})
    except Exception as exc:
        write(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Include setup failures, before the model exists, in the audit record.
        # Never overwrite a completed attempt or an earlier failure receipt.
        if '--output-dir' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--output-dir') + 1])
            if folder.is_dir() and not (folder / 'completion.json').exists() and not (folder / 'failure.json').exists():
                write(folder / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise

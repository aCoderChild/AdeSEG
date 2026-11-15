"""Exploratory CFBI-gate pilot with matched controls and native SAM2 propagation."""
import argparse
import gc
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.cfbi_region_gate import NativeCFBIGate, RegionGate
from modeling.cfbi_training import capture_training_read, object_presence_loss, replay_training_read
from modeling.regional_refinement import refinement_loss
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity, tensor_hash
from scripts.run_regional_refinement import load, save_tensor, write

PROTOCOL = 'protocols/cfbi_region_gate_pilot_v1.json'
SOURCES = [PROTOCOL, 'protocols/cfbi_region_gate_development_v1.json', 'scripts/run_cfbi_region_pilot.py', 'modeling/cfbi_region_gate.py',
           'modeling/cfbi_training.py', 'tests/test_cfbi_region_pilot.py',
           'tests/test_cfbi_region_gate.py', 'tests/test_cfbi_training.py',
           'modeling/region_definitions.py', 'modeling/region_tokens.py', 'modeling/region_bias.py',
           'modeling/regional_refinement.py', 'scripts/run_author_baseline.py',
           'scripts/run_regional_refinement.py', 'scripts/bank_integrity.py',
           'scripts/memory_token_io.py', 'scripts/false_alarm_statistics.py',
           'evaluation/segmentation.py', 'evaluation/davis2017/metrics.py',
           'datasets/io.py', 'datasets/video.py']
CORE = ('prediction_sha256', 'low_res_sha256', 'bank_sha256', 'cache_sha256')


def activate_author_repo():
    repo, revision = source_identity('sam2')
    path = str(repo)
    if path not in sys.path:
        sys.path.insert(0, path)
    return repo, revision


def training_targets(prompt, length):
    targets = list(range(prompt + 1, length))
    if len(targets) < 4:
        raise ValueError('Pilot requires four post-prompt training targets')
    return [targets[i] for i in np.linspace(0, len(targets) - 1, 4, dtype=int)]


def native_reference(reference_dir, name):
    with zipfile.ZipFile(reference_dir / f'full_{name}_C0.zip') as archive:
        return {r['frame']: r for r in map(json.loads, archive.read('integrity.jsonl').decode().splitlines())}


def core_hashes(state, frame, logits):
    outputs = state['output_dict_per_obj'][0]
    stored = outputs['cond_frame_outputs'].get(frame, outputs['non_cond_frame_outputs'].get(frame))
    if stored is None:
        raise RuntimeError('Missing native output')
    return {'prediction_sha256': tensor_hash(logits), 'low_res_sha256': tensor_hash(stored['pred_masks']),
            'bank_sha256': fingerprint(state['output_dict_per_obj'])['sha256'],
            'cache_sha256': fingerprint(state['cached_features'])['sha256']}, stored


def prepare(model, frames, directory):
    stage_frames(frames, directory)
    state = model.init_state(video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=False)
    prompt = prompt_index(frames, {'initialization': 'first_labeled_frame'})
    mask = torch.from_numpy(load_label_mask(frames[prompt].mask_path, [1]) == 1)
    model.add_new_mask(state, frame_idx=prompt, obj_id=1, mask=mask)
    return state, prompt


def training_capture(args, name, frames, repo):
    model = build_author_predictor(args, repo)
    expected = native_reference(args.reference_dir, name)
    folder = args.output_dir / 'training_cache' / name
    folder.mkdir(parents=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='frames_', dir=args.output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        selected = set(training_targets(prompt, len(frames)))
        for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
            if ids != [1] or frame not in expected:
                raise RuntimeError(f'Unexpected native frame {name}/{frame}')
            core, stored = core_hashes(state, frame, logits)
            if any(core[k] != expected[frame][k] for k in CORE):
                raise RuntimeError(f'Native capture mismatch {name}/{frame}')
            if frame in selected:
                sample = capture_training_read(model, state, frame)
                gt = load_label_mask(frames[frame].mask_path, [1]) == 1
                sample['target'] = F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None],
                                                 size=(256, 256), mode='nearest')
                sample['native_logits'] = stored['pred_masks'].detach().float().clone()
                sample['sequence'] = name
                path = folder / f'{frame:06d}.pt'
                save_tensor(sample, path)
                rows.append({'sequence': name, 'frame': frame, 'path': str(path), 'sha256': digest(path)})
            write(args.output_dir / 'progress.json', {'phase': 'capture', 'sequence': name, 'frame': frame})
        if {r['frame'] for r in rows} != selected:
            raise RuntimeError('Training capture incomplete')
        model.reset_state(state)
    del model
    gc.collect()
    return rows


def train(args, samples, repo, mode, schedule, *, presence_weight=0.0):
    model = build_author_predictor(args, repo)
    torch.manual_seed(20261031)
    gate = RegionGate().train()
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.001)
    losses, gradients = [], []
    for step, selected in enumerate(schedule):
        row = samples[int(selected)]
        sample = load(row['path'], row['sha256'])
        before = fingerprint(sample['output_dict'])['sha256']
        optimizer.zero_grad()
        if presence_weight:
            prediction, object_score, trace = replay_training_read(
                model, gate, sample, mode, row['sequence'], return_object_score=True)
        else:
            prediction, trace = replay_training_read(model, gate, sample, mode, row['sequence'])
        if step == 0 and not torch.equal(prediction.detach(), sample['native_logits']):
            raise RuntimeError(f'{mode} zero-gate replay mismatch')
        loss = refinement_loss(prediction, sample['target'])
        if presence_weight:
            loss = loss + presence_weight * object_presence_loss(object_score, sample['target'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad = gate.gate.IA.weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError('Missing/nonfinite gate gradient')
        gradients.append(float(grad.abs().sum()))
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen SAM2 acquired parameter gradients')
        optimizer.step()
        if fingerprint(sample['output_dict'])['sha256'] != before:
            raise RuntimeError('Training replay changed stored memory')
        losses.append(float(loss.detach()))
        write(args.output_dir / 'progress.json', {'phase': 'training', 'mode': mode, 'step': step + 1, 'total': len(schedule)})
    if not any(g > 0 for g in gradients):
        raise RuntimeError('All gate gradients were zero')
    folder = args.output_dir / mode
    folder.mkdir()
    save_tensor(gate.state_dict(), folder / 'head.pt')
    write(folder / 'training.json', {'schedule': schedule.tolist(), 'losses': losses, 'gradient_l1': gradients, 'presence_weight': presence_weight,
                                     'checkpoint_sha256': digest(folder / 'head.pt')})
    del model
    gc.collect()


def validate(args, frames, repo, mode, name, checkpoint, checkpoint_hash):
    model = build_author_predictor(args, repo)
    gate = RegionGate().eval()
    gate.load_state_dict(load(checkpoint, checkpoint_hash), strict=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='frames_', dir=args.output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        with NativeCFBIGate(model, gate, mode, name) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError('Unexpected object IDs')
                if frame == prompt:
                    continue
                read = observer.trace[-1]
                if read['frame'] != frame or read['layers'] != [0, 1, 2, 3]:
                    raise RuntimeError('Incomplete native attention read')
                gt = load_label_mask(frames[frame].mask_path, [1]) == 1
                pred = logits[0, 0].cpu().numpy() > 0
                scores = {**calculate_scores(pred, gt), 'boundary_f': float(db_eval_boundary(pred, gt))}
                rows.append({'frame': frame, 'present': bool(gt.any()), 'scores': scores,
                             'absent_fp': bool(pred.any()) if not gt.any() else None,
                             'prediction_sha256': tensor_hash(logits), 'read': read})
                write(args.output_dir / 'progress.json', {'phase': 'validation', 'mode': mode, 'sequence': name, 'frame': frame})
        if [r['frame'] for r in rows] != list(range(prompt + 1, len(frames))):
            raise RuntimeError('Incomplete validation trajectory')
        model.reset_state(state)
    del model
    gc.collect()
    present = [r for r in rows if r['present']]
    summary = propagation_summary(rows)
    summary.update({key: float(np.mean([r['scores'][key] for r in present])) for key in ('iou', 'boundary_f')})
    return rows, summary


def verify_baseline_coverage(rows, prompt, length):
    if [r['frame'] for r in rows] != list(range(prompt + 1, length)):
        raise RuntimeError('Native baseline does not cover the full validation trajectory')
    if any(not isinstance(r['present'], bool) for r in rows):
        raise RuntimeError('Native baseline presence labels are invalid')


def report(results, names):
    lines = ['# Exploratory CFBI region-gate pilot', '',
             'Outcome-informed stress cases; four sequences, one seed, no independent confirmation.', '',
             '| Condition | Mean Dice | Mean IoU | Mean Boundary F | False alarms |',
             '|---|---:|---:|---:|---:|']
    groups = {'native': results['native'], **results['conditions']}
    for mode, sequences in groups.items():
        values = [float(np.mean([sequences[n][key] for n in names])) for key in ('present_dice', 'iou', 'boundary_f')]
        false_alarms = sum(sequences[n]['gt_empty_false_alarm_count'] for n in names)
        lines.append(f'| {mode} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {false_alarms} |')
    lines += ['', '| Sequence | Native | Regional | Shuffled | Generic |', '|---|---:|---:|---:|---:|']
    for name in names:
        values = [groups[mode][name]['present_dice'] for mode in ('native', 'regional', 'shuffled', 'generic')]
        lines.append(f"| {name} | " + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', 'Inspect per-sequence false alarms and return events in results.json. The three trained arms share',
              'the same sample order and parameter count; generic context contains only one distinct mean.',
              'This small, outcome-informed development pilot is not a reliable improvement claim.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference-dir', 'source', 'engineering-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--integration-gate', type=Path, default=None)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        args.method, args.device = 'sam2', 'cpu'
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *SOURCES], cwd=ROOT, check=True)
        subprocess.run(['git', 'ls-files', '--error-unmatch', *SOURCES], cwd=ROOT, check=True, capture_output=True)
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        smoke = json.loads((args.engineering_dir / 'nonzero_propagation_gate/completion.json').read_text())
        zero = json.loads((args.engineering_dir / 'verified_gate/gate.json').read_text())
        gradient = json.loads((args.engineering_dir / 'training_adapter_gate/gradient_gate.json').read_text())
        serialized = json.loads((args.engineering_dir / 'serialized_adapter_gate/optimizer_gate.json').read_text())
        tests = json.loads((args.engineering_dir / 'pilot_tests_receipt.json').read_text())
        integration_path = args.integration_gate or (args.engineering_dir / 'pilot_integration_gate/completion.json')
        integration = json.loads(integration_path.read_text())
        if not (smoke['passed'] and zero['passed'] and gradient['passed'] and serialized['passed'] and tests['passed'] and integration['passed']):
            raise RuntimeError('Real-data gates incomplete')
        if tests['sources'] != hashes or integration['sources'] != hashes:
            raise RuntimeError('Test or integration receipt source differs')
        if any(hashes[name] != expected for name, expected in smoke['source_hashes'].items()):
            raise RuntimeError('Nonzero smoke source differs')
        repo, revision = activate_author_repo()
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        prior = json.loads((args.source / 'setup.json').read_text())
        if digest(args.checkpoint) != prior['checkpoint_sha256'] or digest(manifest) != prior['manifest_sha256']:
            raise RuntimeError('Checkpoint/manifest mismatch')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        inputs_path = args.source / 'inputs.json'
        inputs_hash = digest(inputs_path)
        input_receipts = json.loads(inputs_path.read_text())
        names = protocol['training_sequences'] + protocol['validation_sequences']
        seen = {}
        for name in names:
            for frame, receipt in zip(videos[name], input_receipts[name], strict=True):
                if frame.frame_index != receipt['frame'] or digest(frame.image_path) != receipt['image_sha256'] or digest(frame.mask_path) != receipt['mask_sha256']:
                    raise RuntimeError(f'Input changed: {name}/{frame.frame_index}')
                owner = seen.setdefault(receipt['image_sha256'], name)
                if owner != name:
                    raise RuntimeError(f'Exact image shared across sequences: {owner}/{name}')
        references = {name: digest(args.reference_dir / f'full_{name}_C0.zip') for name in protocol['training_sequences']}
        baseline_hashes = {name: digest(args.source / 'cache' / name / 'native_frames.json') for name in protocol['validation_sequences']}
        native = {}
        for name in protocol['validation_sequences']:
            rows = json.loads((args.source / 'cache' / name / 'native_frames.json').read_text())
            prompt = prompt_index(videos[name], {'initialization': 'first_labeled_frame'})
            verify_baseline_coverage(rows, prompt, len(videos[name]))
            native[name] = {**propagation_summary(rows), **{key: float(np.mean([r['scores'][key] for r in rows if r['present']])) for key in ('iou', 'boundary_f')}}
        write(args.output_dir / 'setup.json', {'protocol': protocol, 'sources': hashes, 'sam2_revision': revision,
              'checkpoint_sha256': prior['checkpoint_sha256'], 'manifest_sha256': prior['manifest_sha256'],
              'source32_setup_sha256': digest(args.source / 'setup.json'), 'inputs_sha256': inputs_hash,
              'training_reference_sha256': references, 'validation_native_frames_sha256': baseline_hashes,
              'patient_independence_verified': False})
        samples = []
        for name in protocol['training_sequences']:
            samples += training_capture(args, name, videos[name], repo)
        write(args.output_dir / 'training_samples.json', samples)
        schedule = np.random.default_rng(20261031).choice(len(samples), size=24, replace=True)
        write(args.output_dir / 'schedule.json', schedule.tolist())
        for mode in protocol['conditions']:
            train(args, samples, repo, mode, schedule)
        results = {'native': native, 'conditions': {}, 'scope': protocol['limits']}
        for mode in protocol['conditions']:
            info = json.loads((args.output_dir / mode / 'training.json').read_text())
            results['conditions'][mode] = {}
            for name in protocol['validation_sequences']:
                rows, summary = validate(args, videos[name], repo, mode, name, args.output_dir / mode / 'head.pt', info['checkpoint_sha256'])
                write(args.output_dir / mode / f'{name}_frames.json', rows)
                results['conditions'][mode][name] = summary
        write(args.output_dir / 'results.json', results)
        (args.output_dir / 'results.md').write_text(report(results, protocol['validation_sequences']))
        if hashes != {name: digest(ROOT / name) for name in SOURCES}:
            raise RuntimeError('Source changed during pilot')
        if digest(inputs_path) != inputs_hash or any(digest(args.source / 'cache' / n / 'native_frames.json') != h for n, h in baseline_hashes.items()):
            raise RuntimeError('Baseline receipts changed during pilot')
        write(args.output_dir / 'completion.json', {'complete': True, 'training_samples': len(samples), 'validation_sequences': protocol['validation_sequences']})
        write(args.output_dir / 'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

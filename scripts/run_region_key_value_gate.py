"""Train and evaluate a region-conditioned value gate against native SAM2."""
import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.native_region_kv_gate import NativeRegionalKVGate
from modeling.paired_readout_training import clip_loss
from modeling.region_key_value_gate import RegionKeyValueGateHeads
from modeling.regional_refinement import refinement_loss
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = 'protocols/region_key_value_gate_polypgen_v1.json'
BASELINE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/81_query_conditioned_region_bias_polypgen_v1')
SOURCES = [PROTOCOL, 'scripts/run_region_key_value_gate.py', 'modeling/region_key_value_gate.py',
           'modeling/native_region_readout.py', 'modeling/native_region_kv_gate.py',
           'modeling/paired_readout_training.py',
           'modeling/cfbi_training.py', 'modeling/decoder_training_mask.py',
           'modeling/regional_refinement.py', 'modeling/region_definitions.py',
           'modeling/region_tokens.py', 'modeling/cfbi_region_gate.py',
           'scripts/verify_on_demand_clip.py', 'scripts/run_author_baseline.py',
           'scripts/run_cfbi_region_pilot.py', 'scripts/bank_integrity.py',
           'scripts/false_alarm_statistics.py', 'scripts/memory_token_io.py',
           'evaluation/segmentation.py', 'evaluation/davis2017/metrics.py',
           'datasets/sam2_clip_sampler.py', 'datasets/video.py', 'datasets/io.py',
           'scripts/verify_region_key_value_gate.py', 'tests/test_region_key_value_gate.py', 'external/RMNet/models/rmnet.py',
           'external/RMNet/LICENSE']


def make_report(result, names):
    groups = {'native': result['native'], 'regional_kv_gain': result['regional_kv_gain']}
    lines = ['# Region-conditioned key/value-gate pilot', '', '| Sequence | Native Dice | Regional Dice | Difference |',
             '|---|---:|---:|---:|']
    deltas = []
    for name in names:
        a = groups['native'][name]['present_dice']
        b = groups['regional_kv_gain'][name]['present_dice']
        deltas.append(b-a)
        lines.append(f'| {name} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    a = float(np.mean([groups['native'][name]['present_dice'] for name in names]))
    b = float(np.mean([groups['regional_kv_gain'][name]['present_dice'] for name in names]))
    lines.append(f'| **Equal-sequence mean ({len(names)})** | **{a:.6f}** | **{b:.6f}** | **{b-a:+.6f}** |')
    lines += ['', f'Sequence wins: {sum(x > 0 for x in deltas)}/{len(deltas)}; ties: {sum(x == 0 for x in deltas)}. ',
              'No shuffled condition was run. One-seed PolypGen result is exploratory.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--engineering-gate', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=BASELINE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        gate = json.loads(args.engineering_gate.read_text())
        baseline_setup = json.loads((args.baseline_dir / 'setup.json').read_text())
        baseline_completion = json.loads((args.baseline_dir / 'completion.json').read_text())
        if not gate.get('passed') or not baseline_completion.get('complete'):
            raise RuntimeError('Engineering gate or native baseline is incomplete')
        if digest(args.checkpoint) != gate['checkpoint_sha256'] or digest(args.checkpoint) != baseline_setup['checkpoint_sha256']:
            raise RuntimeError('Checkpoint does not match both gates and archived baseline')
        manifest = ROOT / protocol['data']['manifest']
        if digest(manifest) != gate['manifest_sha256'] or digest(manifest) != baseline_setup['manifest_sha256']:
            raise RuntimeError('Manifest does not match both gates and archived baseline')
        git_rev = subprocess.check_output(['git', '-C', str(ROOT / 'external/RMNet'), 'rev-parse', 'HEAD'], text=True).strip()
        if git_rev != protocol['prior_work']['RMNet_revision']:
            raise RuntimeError('RMNet source revision mismatch')
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        gate_sources = gate.get('sources', {})
        if not gate_sources or any(hashes.get(name) != value for name, value in gate_sources.items()):
            raise RuntimeError('Region-value engineering gate source hashes do not match')
        repo, revision = activate_author_repo()
        if revision != baseline_setup['sam2_revision']:
            raise RuntimeError('SAM2 revision differs from native baseline')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        training = protocol['data']['training_sequences']
        validation = protocol['data']['evaluation_sequences']
        if set(training) & set(validation):
            raise RuntimeError('Training and evaluation sequences overlap')
        args.method, args.device = 'sam2', 'cpu'
        args.model_config = protocol['model']['configuration']
        schedule = [(name, protocol['data']['seed'] + epoch * len(training) + i)
                    for epoch in range(protocol['data']['epochs']) for i, name in enumerate(training)]
        write(args.output_dir / 'setup.json', {'protocol': protocol, 'protocol_sha256': digest(ROOT / PROTOCOL),
              'sources': hashes, 'checkpoint_sha256': digest(args.checkpoint),
              'manifest_sha256': digest(manifest), 'sam2_revision': revision,
              'rmnet_revision': git_rev, 'native_baseline': str(args.baseline_dir),
              'native_baseline_setup_sha256': digest(args.baseline_dir / 'setup.json'),
              'engineering_gate_sha256': digest(args.engineering_gate), 'schedule': schedule})

        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        heads = RegionKeyValueGateHeads(channels=model.mem_dim).train()
        optimizer = torch.optim.Adam(heads.parameters(), lr=protocol['training']['learning_rate'])
        updates, seen_layers = [], set()
        for step, (name, seed) in enumerate(schedule):
            frames = sample_clip(videos[name], protocol['data']['clip_length'], seed)
            samples = capture_clip(model, frames, args.output_dir)
            memory_hashes = [fingerprint(sample['output_dict'])['sha256'] for sample in samples]
            optimizer.zero_grad()
            loss, calls = clip_loss(model, heads, samples, 'regional_kv_gain', name,
                                    check_native=(step == 0), return_trace=True)
            if len(calls) != 4 * len(samples) or [c['layer'] for c in calls] != [0, 1, 2, 3] * len(samples):
                raise RuntimeError('Training clip did not read all four layers for every target')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            grads = []
            for layer, head in enumerate(heads.layers):
                grad = head.gain_logits.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError(f'Missing/nonfinite region-gate gradient at layer {layer}')
                grads.append(float(grad.abs().sum()))
                if grads[-1] > 0:
                    seen_layers.add(layer)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen SAM2 acquired gradients')
            if memory_hashes != [fingerprint(sample['output_dict'])['sha256'] for sample in samples]:
                raise RuntimeError('Training changed captured native memory')
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in heads.parameters()):
                raise RuntimeError('Nonfinite trained adapter')
            updates.append({'sequence': name, 'seed': seed,
                            'frames': [frame.frame_index for frame in frames],
                            'loss': float(loss.detach()), 'gradient_l1': grads})
            write(args.output_dir / 'training.json', updates)
            write(args.output_dir / 'progress.json', {'phase': 'training', 'step': step + 1,
                  'total': len(schedule), 'sequence': name})
            del samples
        if seen_layers != {0, 1, 2, 3}:
            raise RuntimeError('One or more attention layers received no training gradient')
        head_path = args.output_dir / 'regional_kv_gain.pt'
        save_tensor(heads.state_dict(), head_path)
        write(args.output_dir / 'regional_kv_gain_training.json', {
            'updates': len(updates), 'schedule': schedule, 'checkpoint_sha256': digest(head_path),
            'all_layers_received_gradient': sorted(seen_layers)})
        del model, heads, optimizer
        gc.collect()

        native_groups, regional_groups = {}, {}
        for name in validation:
            native_rows = json.loads((args.baseline_dir / 'native' / f'{name}_frames.json').read_text())
            prompt = prompt_index(videos[name], {'initialization': 'first_labeled_frame'})
            frames_by_index = {frame.frame_index: frame for frame in videos[name]}
            if [row['frame'] for row in native_rows] != list(range(prompt + 1, len(videos[name]))):
                raise RuntimeError(f'Native baseline frame coverage mismatch: {name}')
            native_groups[name] = propagation_summary(native_rows)
            present = [r for r in native_rows if r['present']]
            native_groups[name].update({key: float(np.mean([r['scores'][key] for r in present]))
                                        for key in ('iou', 'boundary_f')})

            model = build_author_predictor(args, repo)
            heads = RegionKeyValueGateHeads(channels=model.mem_dim).eval()
            receipt = json.loads((args.output_dir / 'regional_kv_gain_training.json').read_text())
            heads.load_state_dict(load(head_path, receipt['checkpoint_sha256']), strict=True)
            rows = []
            with tempfile.TemporaryDirectory(prefix=f'{name}_', dir=args.output_dir) as temp:
                state, native_prompt = prepare(model, videos[name], Path(temp))
                if native_prompt != prompt:
                    raise RuntimeError(f'Prompt mismatch: {name}')
                started = time.monotonic()
                with NativeRegionalKVGate(model, heads, name) as observer, torch.inference_mode():
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1]:
                            raise RuntimeError(f'Unexpected object IDs: {name}/{frame}')
                        if frame == prompt:
                            continue
                        native_row = next((r for r in native_rows if r['frame'] == frame), None)
                        if native_row is None:
                            raise RuntimeError(f'No matching native output: {name}/{frame}')
                        read = observer.observer.trace[-1]
                        calls = observer.calls[-4:]
                        if read['frame'] != frame or [c['layer'] for c in calls] != [0, 1, 2, 3]:
                            raise RuntimeError(f'Incomplete four-layer regional read: {name}/{frame}')
                        gt = load_label_mask(frames_by_index[frame].mask_path, [1]) == 1
                        pred = logits[0, 0].detach().cpu().numpy() > 0
                        scores = {**calculate_scores(pred, gt), 'boundary_f': float(db_eval_boundary(pred, gt))}
                        rows.append({'frame': frame, 'present': bool(gt.any()), 'scores': scores,
                                     'absent_fp': bool(pred.any()) if not gt.any() else None,
                                     'prediction_sha256': tensor_hash(logits),
                                     'native_prediction_sha256': native_row['prediction_sha256'],
                                     'read': read, 'attention_layers': calls})
                        write(args.output_dir / 'progress.json', {'phase': 'validation',
                              'sequence': name, 'frame': frame})
                elapsed = time.monotonic() - started
                model.reset_state(state)
            if [r['frame'] for r in rows] != list(range(prompt + 1, len(videos[name]))):
                raise RuntimeError(f'Regional trajectory incomplete: {name}')
            regional_groups[name] = propagation_summary(rows)
            present = [r for r in rows if r['present']]
            regional_groups[name].update({key: float(np.mean([r['scores'][key] for r in present]))
                                          for key in ('iou', 'boundary_f')})
            regional_groups[name]['inference_wall_seconds'] = elapsed
            regional_groups[name]['mean_spatial_memory_keys'] = float(np.mean(
                [sum(c['spatial_keys'] for c in r['attention_layers']) / 4 for r in rows]))
            regional_groups[name]['mean_pointer_tokens'] = float(np.mean(
                [sum(c['pointer_keys'] for c in r['attention_layers']) / 4 for r in rows]))
            write(args.output_dir / f'{name}_regional_frames.json', rows)
            model.reset_state(state)
            del model, heads
            gc.collect()

        if hashes != {name: digest(ROOT / name) for name in SOURCES}:
            raise RuntimeError('Source changed during experiment')
        result = {'complete': True, 'native': native_groups, 'regional_kv_gain': regional_groups,
                  'evaluation_sequences': validation, 'scope': protocol['data']['scope']}
        write(args.output_dir / 'results.json', result)
        (args.output_dir / 'results.md').write_text(make_report(result, validation))
        write(args.output_dir / 'completion.json', {'complete': True, 'updates': len(updates),
              'validation_sequences': validation, 'conditions': ['native', 'regional_kv_gain'],
              'shuffled_evaluation': False})
        write(args.output_dir / 'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

"""Train a Cutie regional branch with a native object-presence anchor."""
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
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.cutie_region_projection import CutieRegionProjectionHeads
from modeling.native_region_readout import NativeRegionalReadout
from modeling.region_presence_anchor import clip_loss_with_presence_anchor
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.capture_presence_clip import capture_presence_clip

PROTOCOL = 'protocols/cutie_region_presence_anchor_polypgen_v1.json'
BASELINE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/81_query_conditioned_region_bias_polypgen_v1')
GATE_DEFAULT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/93_cutie_presence_anchor_engineering_gate_v1')
SOURCES = [PROTOCOL, 'scripts/run_cutie_presence_anchor.py',
           'scripts/verify_cutie_presence_anchor.py', 'scripts/capture_presence_clip.py',
           'modeling/region_presence_anchor.py', 'scripts/run_cutie_region_projection.py',
           'scripts/verify_cutie_region_projection.py', 'modeling/cutie_region_projection.py',
           'modeling/cutie_region_prototypes.py',
           'modeling/cutie_memory.py', 'modeling/cutie_region_queries.py',
           'modeling/paired_readout_training.py', 'modeling/native_region_readout.py',
           'modeling/cfbi_region_gate.py', 'modeling/region_definitions.py',
           'modeling/region_tokens.py', 'modeling/cfbi_training.py',
           'modeling/decoder_training_mask.py', 'modeling/regional_refinement.py',
           'scripts/verify_on_demand_clip.py',
           'scripts/run_author_baseline.py', 'scripts/run_cfbi_region_pilot.py',
           'scripts/bank_integrity.py', 'scripts/false_alarm_statistics.py',
           'scripts/memory_token_io.py', 'scripts/run_regional_refinement.py',
           'evaluation/segmentation.py', 'evaluation/davis2017/metrics.py',
           'datasets/sam2_clip_sampler.py', 'datasets/video.py', 'datasets/io.py',
           'tests/test_cutie_region_prototypes.py', 'tests/test_cutie_region_projection.py',
           'external/Cutie/cutie/model/transformer/object_summarizer.py',
           'external/Cutie/cutie/model/transformer/transformer_layers.py',
           'external/Cutie/cutie/model/transformer/positional_encoding.py']


def report(native, regional, names):
    lines = ['# Presence-anchored Cutie regional readout on SAM2 memory', '',
             '| Sequence | Native SAM2 Dice | Presence-anchored regional Dice | Difference |',
             '|---|---:|---:|---:|']
    differences = []
    for name in names:
        a, b = native[name]['present_dice'], regional[name]['present_dice']
        differences.append(b - a)
        lines.append(f'| {name} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    mean_native = float(np.mean([native[n]['present_dice'] for n in names]))
    mean_regional = float(np.mean([regional[n]['present_dice'] for n in names]))
    lines.append(f'| **Equal-sequence mean ({len(names)})** | **{mean_native:.6f}** | **{mean_regional:.6f}** | **{mean_regional-mean_native:+.6f}** |')
    lines.extend(['', f'Sequence wins: {sum(d > 0 for d in differences)}/{len(differences)}; '
                  f'losses: {sum(d < 0 for d in differences)}/{len(differences)}; '
                  f'ties: {sum(d == 0 for d in differences)}.',
                  'No shuffled condition was run. This one-seed PolypGen result is exploratory.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--engineering-gate', type=Path, default=GATE_DEFAULT)
    parser.add_argument('--baseline-dir', type=Path, default=BASELINE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol_path = ROOT / PROTOCOL
        protocol = json.loads(protocol_path.read_text())
        gate = json.loads((args.engineering_gate / 'completion.json').read_text())
        baseline_setup = json.loads((args.baseline_dir / 'setup.json').read_text())
        baseline_completion = json.loads((args.baseline_dir / 'completion.json').read_text())
        if not gate.get('passed') or not baseline_completion.get('complete'):
            raise RuntimeError('Real-data engineering gate or native baseline is incomplete')
        if digest(args.checkpoint) != gate['checkpoint_sha256'] or digest(args.checkpoint) != baseline_setup['checkpoint_sha256']:
            raise RuntimeError('Checkpoint does not match the engineering gate and native baseline')
        manifest = ROOT / protocol['data']['manifest']
        if digest(manifest) != gate['manifest_sha256'] or digest(manifest) != baseline_setup['manifest_sha256']:
            raise RuntimeError('Manifest does not match the engineering gate and native baseline')
        cutie_revision = subprocess.check_output(
            ['git', '-C', str(ROOT / 'external/Cutie'), 'rev-parse', 'HEAD'], text=True).strip()
        if cutie_revision != protocol['prior_work']['revision']:
            raise RuntimeError('Cutie source revision does not match the frozen protocol')
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for path, expected in gate['sources'].items():
            if hashes.get(path) != expected:
                raise RuntimeError(f'Engineering gate source changed: {path}')
        repo, sam2_revision = activate_author_repo()
        if sam2_revision != baseline_setup['sam2_revision']:
            raise RuntimeError('SAM2 revision differs from the native baseline')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        training = protocol['data']['training_sequences']
        validation = protocol['data']['evaluation_sequences']
        if set(training) & set(validation):
            raise RuntimeError('Training and evaluation sequences overlap')
        args.method, args.device = 'sam2', 'cpu'
        args.model_config = protocol['model']['sam2_configuration']
        schedule = [(name, protocol['data']['seed'] + epoch * len(training) + i)
                    for epoch in range(protocol['data']['epochs']) for i, name in enumerate(training)]
        write(args.output_dir / 'setup.json', {
            'protocol': protocol, 'protocol_sha256': digest(protocol_path), 'sources': hashes,
            'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(manifest),
            'sam2_revision': sam2_revision, 'cutie_revision': cutie_revision,
            'native_baseline': str(args.baseline_dir),
            'native_baseline_setup_sha256': digest(args.baseline_dir / 'setup.json'),
            'engineering_gate_sha256': digest(args.engineering_gate / 'completion.json'),
            'schedule': schedule})

        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        heads = CutieRegionProjectionHeads().train()
        optimizer = torch.optim.AdamW(heads.parameters(),
                                      lr=protocol['training']['learning_rate'],
                                      weight_decay=protocol['training']['weight_decay'])
        updates, seen_layers = [], set()
        for step, (name, seed) in enumerate(schedule):
            frames = sample_clip(videos[name], protocol['data']['clip_length'], seed)
            samples = capture_presence_clip(model, frames, args.output_dir)
            memory_hashes = [fingerprint(sample['output_dict'])['sha256'] for sample in samples]
            optimizer.zero_grad(set_to_none=True)
            loss, calls, anchor_loss = clip_loss_with_presence_anchor(
                model, heads, samples, name, anchor_weight=protocol['training']['presence_anchor_weight'],
                check_native=(step == 0), return_trace=True)
            expected = [layer for _ in samples for layer in range(4)]
            if [call['layer'] for call in calls] != expected:
                raise RuntimeError(f'Not all four attention layers were observed: {name}')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite regional-summary training loss')
            loss.backward()
            projection_gradients, reader_gradient_by_layer = [], []
            for layer, head in enumerate(heads.layers):
                gradient = head.output_projection.weight.grad
                if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                    raise RuntimeError(f'Invalid output-projection gradient at layer {layer}')
                projection_gradients.append(float(gradient.abs().sum()))
                reader_grad = sum(float(p.grad.abs().sum()) for p in head.reader.parameters()
                                  if p.grad is not None)
                reader_gradient_by_layer.append(reader_grad)
                if step > 0 and reader_grad <= 0:
                    raise RuntimeError(f'Cutie reader received no training gradient at layer {layer}')
                seen_layers.add(layer)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen SAM2 acquired gradients')
            if memory_hashes != [fingerprint(sample['output_dict'])['sha256'] for sample in samples]:
                raise RuntimeError('Training modified captured native memory')
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                raise RuntimeError('Nonfinite regional-summary parameters')
            updates.append({'sequence': name, 'seed': seed,
                            'frames': [frame.frame_index for frame in frames],
                            'loss': float(loss.detach()), 'presence_anchor_loss': float(anchor_loss.detach()),
                            'projection_gradient_l1': projection_gradients,
                            'reader_gradient_l1': reader_gradient_by_layer})
            write(args.output_dir / 'training.json', updates)
            write(args.output_dir / 'progress.json', {'phase': 'training', 'step': step + 1,
                  'total': len(schedule), 'sequence': name})
            del samples
        if seen_layers != {0, 1, 2, 3}:
            raise RuntimeError('One or more attention layers received no training gradient')
        head_path = args.output_dir / 'regional_summary_heads.pt'
        save_tensor(heads.state_dict(), head_path)
        write(args.output_dir / 'training_summary.json', {
            'updates': len(updates), 'schedule': schedule,
            'checkpoint_sha256': digest(head_path), 'layers_received_gradient': sorted(seen_layers),
            'output_projection_rms': [float(head.output_projection.weight.detach().square().mean().sqrt())
                                      for head in heads.layers]})
        del model, heads, optimizer
        gc.collect()

        native_groups, regional_groups = {}, {}
        for name in validation:
            native_rows = json.loads((args.baseline_dir / 'native' / f'{name}_frames.json').read_text())
            prompt = prompt_index(videos[name], {'initialization': 'first_labeled_frame'})
            if [row['frame'] for row in native_rows] != list(range(prompt + 1, len(videos[name]))):
                raise RuntimeError(f'Native baseline frame coverage mismatch: {name}')
            native_groups[name] = propagation_summary(native_rows)
            native_present = [row for row in native_rows if row['present']]
            native_groups[name].update({key: float(np.mean([row['scores'][key] for row in native_present]))
                                        for key in ('iou', 'boundary_f')})

            model = build_author_predictor(args, repo)
            heads = CutieRegionProjectionHeads().eval()
            training_receipt = json.loads((args.output_dir / 'training_summary.json').read_text())
            heads.load_state_dict(load(head_path, training_receipt['checkpoint_sha256']), strict=True)
            rows = []
            with tempfile.TemporaryDirectory(prefix=f'{name}_', dir=args.output_dir) as temp:
                state, actual_prompt = prepare(model, videos[name], Path(temp))
                if actual_prompt != prompt:
                    raise RuntimeError(f'Prompt differs from native baseline for {name}')
                started = time.monotonic()
                with NativeRegionalReadout(model, heads, 'regional', name) as observer, torch.inference_mode():
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1]:
                            raise RuntimeError(f'Unexpected object IDs: {name}/{frame}')
                        if frame == prompt:
                            continue
                        native_row = next((row for row in native_rows if row['frame'] == frame), None)
                        if native_row is None:
                            raise RuntimeError(f'Missing native prediction: {name}/{frame}')
                        calls = observer.calls[-4:]
                        trace = observer.observer.trace[-1]
                        if trace['frame'] != frame or [call['layer'] for call in calls] != [0, 1, 2, 3]:
                            raise RuntimeError(f'Incomplete regional read: {name}/{frame}')
                        if any(call['spatial_keys'] <= 0 or call['pointer_keys'] < 0
                               for call in calls):
                            raise RuntimeError(f'Invalid memory token accounting: {name}/{frame}')
                        gt = load_label_mask(videos[name][frame].mask_path, [1]) == 1
                        prediction = logits[0, 0].detach().cpu().numpy() > 0
                        scores = {**calculate_scores(prediction, gt),
                                  'boundary_f': float(db_eval_boundary(prediction, gt))}
                        rows.append({'frame': frame, 'present': bool(gt.any()), 'scores': scores,
                                     'absent_fp': bool(prediction.any()) if not gt.any() else None,
                                     'prediction_sha256': tensor_hash(logits),
                                     'native_prediction_sha256': native_row['prediction_sha256'],
                                     'read': trace, 'attention_layers': calls})
                        write(args.output_dir / 'progress.json', {'phase': 'validation',
                              'sequence': name, 'frame': frame})
                elapsed = time.monotonic() - started
                model.reset_state(state)
            if [row['frame'] for row in rows] != list(range(prompt + 1, len(videos[name]))):
                raise RuntimeError(f'Regional trajectory incomplete: {name}')
            regional_groups[name] = propagation_summary(rows)
            present = [row for row in rows if row['present']]
            regional_groups[name].update({key: float(np.mean([row['scores'][key] for row in present]))
                                          for key in ('iou', 'boundary_f')})
            regional_groups[name]['inference_wall_seconds'] = elapsed
            regional_groups[name]['mean_spatial_memory_keys'] = float(np.mean(
                [sum(call['spatial_keys'] for call in row['attention_layers']) / 4 for row in rows]))
            regional_groups[name]['mean_pointer_tokens'] = float(np.mean(
                [sum(call['pointer_keys'] for call in row['attention_layers']) / 4 for row in rows]))
            write(args.output_dir / f'{name}_regional_frames.json', rows)
            del model, heads
            gc.collect()

        if hashes != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError('Source code or protocol changed during the run')
        result = {'complete': True, 'native': native_groups, 'regional_presence_anchored': regional_groups,
                  'evaluation_sequences': validation, 'scope': protocol['data']['scope']}
        write(args.output_dir / 'results.json', result)
        (args.output_dir / 'results.md').write_text(report(native_groups, regional_groups, validation))
        write(args.output_dir / 'completion.json', {
            'complete': True, 'updates': len(updates), 'evaluation_sequences': validation,
            'conditions': ['native', 'regional_presence_anchored'], 'shuffled_evaluation': False,
            'meaningful_screen': protocol['evaluation'].get(
                'meaningful_screen', protocol['evaluation'].get('exploratory_meaningful_screen'))})
        write(args.output_dir / 'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

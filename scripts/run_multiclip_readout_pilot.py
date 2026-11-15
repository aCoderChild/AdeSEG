"""Train matched readouts on sampled clips and evaluate full video propagation."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.native_region_readout import RegionalReadoutHeads
from modeling.cutie_region_queries import CUTIE_REVISION, CutieRegionQueryHeads
from modeling.query_conditioned_region_readout import QueryConditionedRegionHeads
from modeling.paired_readout_training import clip_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_regional_readout_present_mask_pilot import PilotReadout
from scripts.verify_on_demand_clip import capture_clip
import scripts.run_cfbi_region_pilot as pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--engineering-updates', type=int)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--adapter', choices=('linear', 'cutie-query', 'query-region'), default='linear')
    parser.add_argument('--adapter-gate', type=Path)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        protocol_path = (args.protocol or ROOT / (
            'protocols/cutie_region_query_pilot.json' if args.adapter == 'cutie-query'
            else 'protocols/regional_readout_multiclip_pilot.json')).resolve()
        protocol = json.loads(protocol_path.read_text())
        data_cfg = protocol.get('data', protocol)
        names = data_cfg.get('training_sequences', protocol.get('training_sequences'))
        val_names = data_cfg.get('evaluation_sequences',
                                 data_cfg.get('validation_sequences', protocol.get('validation_sequences')))
        clip_length = data_cfg.get('clip_length', protocol.get('clip_length'))
        epochs = data_cfg.get('epochs', protocol.get('epochs'))
        seed_value = data_cfg.get('seed', protocol.get('seed'))
        if not names or not val_names or not clip_length or not epochs or seed_value is None:
            raise ValueError('Protocol is missing sequence, clip, epoch, or seed settings')
        if args.adapter == 'query-region':
            if args.adapter_gate is None:
                raise ValueError('Query-region adapter requires its real-clip gate receipt')
            adapter_gate = json.loads(args.adapter_gate.read_text())
            if not adapter_gate.get('passed'):
                raise RuntimeError('Query-region real-clip gate did not pass')
        gate = json.loads(args.gate.read_text())
        if not gate['passed'] or digest(args.checkpoint) != gate['checkpoint_sha256']:
            raise ValueError('Optimizer gate or checkpoint mismatch')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        args.method, args.device = 'sam2', 'cpu'
        args.model_config = protocol.get('model_config',
            protocol.get('model', {}).get('configuration', 'configs/sam2.1/sam2.1_hiera_b+.yaml'))
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        if digest(manifest) != gate['manifest_sha256']:
            raise ValueError('Manifest mismatch')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        if set(names) & set(val_names):
            raise ValueError('Training and validation sequences overlap')
        schedule = protocol.get('schedule') or [(name, seed_value + epoch * len(names) + i)
                    for epoch in range(epochs) for i, name in enumerate(names)]
        if args.engineering_updates is not None:
            if not 1 <= args.engineering_updates <= len(schedule):
                raise ValueError('Invalid engineering update count')
            schedule = schedule[:args.engineering_updates]
        source_files = [protocol_path, Path(__file__), ROOT / 'scripts/verify_on_demand_clip.py',
                        ROOT / 'scripts/run_author_baseline.py',
                        ROOT / 'scripts/run_cfbi_region_pilot.py',
                        ROOT / 'scripts/run_regional_readout_present_mask_pilot.py',
                        ROOT / 'scripts/bank_integrity.py',
                        ROOT / 'scripts/memory_token_io.py',
                        ROOT / 'scripts/run_regional_refinement.py',
                        ROOT / 'scripts/false_alarm_statistics.py',
                        ROOT / 'modeling/paired_readout_training.py', ROOT / 'modeling/native_region_readout.py',
                        ROOT / 'modeling/cfbi_region_gate.py',
                        ROOT / 'modeling/cfbi_training.py',
                        ROOT / 'modeling/decoder_training_mask.py',
                        ROOT / 'modeling/regional_refinement.py',
                        ROOT / 'modeling/region_definitions.py',
                        ROOT / 'modeling/region_tokens.py',
                        ROOT / 'datasets/sam2_clip_sampler.py',
                        ROOT / 'datasets/video.py', ROOT / 'datasets/io.py']
        head_factory = RegionalReadoutHeads
        if args.adapter == 'query-region':
            source_files.extend([ROOT / 'modeling/query_conditioned_region_readout.py',
                                 ROOT / 'modeling/region_mass_attention.py',
                                 ROOT / 'scripts/verify_query_conditioned_region_readout.py',
                                 ROOT / 'tests/test_query_conditioned_region_readout.py',
                                 ROOT / 'tests/test_region_mass_attention.py'])
            head_factory = lambda: QueryConditionedRegionHeads(
                heads=1, head_dim=256)
        if args.adapter == 'cutie-query':
            cutie_checkpoint = ROOT / 'external/Cutie/weights/cutie-base-mega.pth'
            def head_factory():
                return CutieRegionQueryHeads(pretrained_checkpoint=cutie_checkpoint)
            source_files.extend([
                ROOT / 'modeling/cutie_region_queries.py',
                ROOT / 'external/Cutie/cutie/model/transformer/object_transformer.py',
                ROOT / 'external/Cutie/cutie/model/transformer/transformer_layers.py',
                ROOT / 'external/Cutie/cutie/model/transformer/positional_encoding.py',
                cutie_checkpoint,
            ])
            if CUTIE_REVISION not in subprocess.check_output(
                    ['git', '-C', str(ROOT / 'external/Cutie'), 'rev-parse', 'HEAD'], text=True):
                raise RuntimeError('Cutie checkout revision mismatch')
            if digest(cutie_checkpoint) != '9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023':
                raise RuntimeError('Cutie pretrained weight digest mismatch')
        hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
        if args.adapter == 'query-region':
            for name, value in adapter_gate.get('sources', {}).items():
                if hashes.get(name) != value:
                    raise RuntimeError(f'Query-region engineering gate source mismatch: {name}')
            if not adapter_gate.get('sources'):
                raise RuntimeError('Query-region engineering gate has no source hashes')
        repo, revision = pilot.activate_author_repo()
        write(args.output_dir / 'setup.json', {'protocol': protocol, 'adapter': args.adapter,
              'cutie_revision': CUTIE_REVISION if args.adapter == 'cutie-query' else None,
              'protocol_sha256': digest(protocol_path),
              'adapter_gate_sha256': digest(args.adapter_gate) if args.adapter_gate else None,
              'cutie_checkpoint_sha256': digest(cutie_checkpoint) if args.adapter == 'cutie-query' else None,
              'schedule': schedule,
              'sources': hashes, 'checkpoint_sha256': digest(args.checkpoint),
              'manifest_sha256': digest(manifest), 'sam2_revision': revision,
              'engineering': args.engineering_updates is not None})
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        modes = tuple(protocol.get('training', {}).get(
            'arms', ('regional', 'shuffled', 'generic')))
        heads = {mode: head_factory().train() for mode in modes}
        if args.adapter == 'query-region':
            optimizers = {mode: torch.optim.Adam([
                {'params': [layer.query_gate for layer in heads[mode].layers],
                 'lr': protocol['training']['learning_rate']},
            ]) for mode in modes}
        else:
            optimizers = {mode: torch.optim.Adam(heads[mode].parameters(), lr=protocol['learning_rate']) for mode in modes}
        records = []
        for step, item in enumerate(schedule):
            name, seed = item[:2]
            candidates = videos[name] if len(item) == 2 else videos[name][item[2]:item[2] + clip_length]
            frames = sample_clip(candidates, clip_length, seed)
            if len(item) == 3:
                from datasets.io import load_label_mask
                present = [bool((load_label_mask(f.mask_path, [1]) == 1).any()) for f in frames]
                if not any(not a and b for a, b in zip(present, present[1:])):
                    raise RuntimeError('Scheduled clip does not contain an empty-to-present transition')
            samples = capture_clip(model, frames, args.output_dir)
            row = {'sequence': name, 'seed': seed, 'frames': [frame.frame_index for frame in frames], 'arms': {}}
            for mode in modes:
                optimizers[mode].zero_grad()
                loss = clip_loss(model, heads[mode], samples, mode, name,
                                 check_native=step == 0, shuffle_seed=seed_value)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                if args.adapter == 'cutie-query':
                    grads = [{'scale': float(layer.scale.grad.abs()) if layer.scale.grad is not None else 0.0,
                              'region_queries': float(layer.region_queries.grad.abs().sum())
                                  if layer.region_queries.grad is not None else 0.0}
                             for layer in heads[mode].layers]
                    bridge_grad = sum(float(parameter.grad.abs().sum())
                                      for parameter in heads[mode].core.parameters()
                                      if parameter.requires_grad and parameter.grad is not None)
                    valid_grads = all(torch.isfinite(torch.tensor(value))
                                      for row in grads for value in row.values())
                    valid_grads = valid_grads and all(row['scale'] > 0 for row in grads)
                    valid_grads = valid_grads and bridge_grad >= 0
                    if step > 0:
                        valid_grads = valid_grads and bridge_grad > 0 and all(
                            row['region_queries'] > 0 for row in grads)
                elif args.adapter == 'query-region':
                    grads = [{'query_gate': float(layer.query_gate.grad.abs().sum())
                                  if layer.query_gate.grad is not None else 0.0}
                             for layer in heads[mode].layers]
                    valid_grads = all(torch.isfinite(torch.tensor(value))
                                      for row in grads for value in row.values())
                    valid_grads = valid_grads and any(row['query_gate'] > 0 for row in grads)
                else:
                    grads = [float(layer.mix.weight.grad.abs().sum()) for layer in heads[mode].layers]
                    valid_grads = all(torch.isfinite(torch.tensor(g)) for g in grads)
                if not valid_grads or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Invalid gradients or unfrozen backbone')
                optimizers[mode].step()
                if not all(torch.isfinite(parameter).all() for parameter in heads[mode].parameters()):
                    raise RuntimeError('Nonfinite weights')
                row['arms'][mode] = {'loss': float(loss.detach()), 'gradient_l1': grads,
                                     'shared_bridge_gradient_l1': bridge_grad if args.adapter == 'cutie-query' else None}
            records.append(row)
            write(args.output_dir / 'training.json', records)
            write(args.output_dir / 'progress.json', {'phase': 'training', 'step': step + 1, 'total': len(schedule)})
            del samples
        if args.adapter == 'query-region':
            for mode in modes:
                if not all(any(row['arms'][mode]['gradient_l1'][layer]['query_gate'] > 0
                               for row in records) for layer in range(4)):
                    raise RuntimeError(f'{mode} query gate missed a layer')
        if hashes != {str(path.relative_to(ROOT)): digest(path) for path in source_files}:
            raise RuntimeError('Source changed during training')
        for mode in modes:
            folder = args.output_dir / mode
            folder.mkdir()
            save_tensor(heads[mode].state_dict(), folder / 'head.pt')
            write(folder / 'training.json', {'checkpoint_sha256': digest(folder / 'head.pt')})
        del model, heads, optimizers
        class SeededPilotReadout(PilotReadout):
            def __init__(self, model, head, mode, sequence):
                super().__init__(model, head, mode, sequence, seed_value)
        pilot.RegionGate, pilot.NativeCFBIGate = head_factory, SeededPilotReadout
        baseline = args.output_dir / 'native'
        baseline.mkdir()
        save_tensor(head_factory().state_dict(), baseline / 'head.pt')
        groups = {}
        for mode in ('native', *modes):
            path = args.output_dir / mode / 'head.pt'
            groups[mode] = {}
            for name in (val_names if args.engineering_updates is None else ['seq3']):
                inference_started = time.monotonic()
                rows, summary = pilot.validate(args, videos[name], repo, 'regional' if mode == 'native' else mode,
                                               name, path, digest(path))
                elapsed = time.monotonic() - inference_started
                summary['inference_wall_seconds'] = elapsed
                summary['inference_wall_seconds_per_frame'] = elapsed / max(1, len(rows))
                summary['mean_spatial_memory_keys'] = sum(
                    row['read']['spatial_keys'] for row in rows) / max(1, len(rows))
                summary['mean_pointer_tokens'] = sum(
                    row['read']['pointer_keys'] for row in rows) / max(1, len(rows))
                write(args.output_dir / mode / f'{name}_frames.json', rows)
                groups[mode][name] = summary
        if hashes != {str(path.relative_to(ROOT)): digest(path) for path in source_files}:
            raise RuntimeError('Scientific source changed during evaluation')
        write(args.output_dir / 'results.json', {'native': groups.pop('native'), 'conditions': groups,
              'scope': protocol.get('scope', protocol.get('data', {}).get('scope'))})
        write(args.output_dir / 'completion.json', {'complete': True, 'updates': len(schedule),
              'engineering': args.engineering_updates is not None})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

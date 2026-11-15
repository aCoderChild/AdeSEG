"""Exploratory regional readout training through predicted memory pairs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.native_region_readout import RegionalReadoutHeads
from modeling.paired_readout_training import pair_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import load, save_tensor, write
import scripts.run_cfbi_region_pilot as pilot
from scripts.run_regional_readout_present_mask_pilot import PilotReadout

SOURCES = ['protocols/regional_readout_pair_training_pilot.json', 'scripts/run_paired_readout_pilot.py',
           'modeling/paired_readout_training.py', 'modeling/native_region_readout.py',
           'modeling/region_readout.py', 'modeling/cfbi_region_gate.py', 'modeling/cfbi_training.py',
           'modeling/decoder_training_mask.py', 'scripts/run_cfbi_region_pilot.py']


def stability_schedule(seed):
    generator = torch.Generator().manual_seed(seed)
    return [int(i) for _ in range(8) for i in torch.randperm(3, generator=generator)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pairs', 'gate', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stability-seed', type=int)
    a = p.parse_args()
    a.output_dir = validate_output(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        a.method, a.device = 'sam2', 'cpu'
        protocol = json.loads((ROOT / SOURCES[0]).read_text())
        if a.stability_seed is not None:
            stability_path = ROOT / 'protocols/regional_readout_stability_pilot.json'
            stability = json.loads(stability_path.read_text())
            if a.stability_seed not in stability['replicates']:
                raise ValueError('Seed is not in the stability protocol')
            SOURCES.append(str(stability_path.relative_to(ROOT)))
            schedule = stability_schedule(a.stability_seed)
        else:
            stability = None
            schedule = [i % 3 for i in range(24)]
        shuffle_seed = a.stability_seed if a.stability_seed is not None else 20261031
        data = json.loads(a.pairs.read_text())
        gate = json.loads(a.gate.read_text())
        if not gate['passed'] or set(gate['modes']) != {'regional', 'shuffled', 'generic'}:
            raise RuntimeError('Matched optimizer gate did not pass')
        if digest(a.checkpoint) != data['checkpoint_sha256'] or gate['checkpoint_sha256'] != data['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        if digest(manifest) != data['manifest_sha256']:
            raise RuntimeError('Manifest mismatch')
        if [p['sequence'] for p in data['pairs']] != protocol['training_sequences']:
            raise RuntimeError('Training sequence mismatch')
        sources = {name: digest(ROOT / name) for name in SOURCES}
        pairs = [[load(row['path'], row['sha256']) for row in pair['samples']] for pair in data['pairs']]
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        names = protocol['validation_sequences']
        inputs = {name: [{'image': digest(f.image_path), 'mask': digest(f.mask_path)} for f in videos[name]] for name in names}
        repo, revision = pilot.activate_author_repo()
        write(a.output_dir / 'setup.json', {'protocol': protocol, 'sources': sources, 'pairs': data,
              'stability_protocol': stability, 'stability_seed': a.stability_seed, 'schedule': schedule,
              'checkpoint_sha256': data['checkpoint_sha256'], 'manifest_sha256': data['manifest_sha256'],
              'validation_inputs': inputs, 'sam2_revision': revision})
        model = build_author_predictor(a, repo)
        model.requires_grad_(False)
        for mode in ('regional', 'shuffled', 'generic'):
            torch.manual_seed(20261031)
            heads = RegionalReadoutHeads().train()
            optimizer = torch.optim.Adam(heads.parameters(), lr=0.001)
            losses, gradients = [], []
            for step in range(24):
                idx = schedule[step]
                optimizer.zero_grad()
                loss = pair_loss(model, heads, pairs[idx], mode, data['pairs'][idx]['sequence'], check_native=step == 0, shuffle_seed=shuffle_seed)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                norms = [float(h.mix.weight.grad.abs().sum()) for h in heads.layers]
                if not all(np.isfinite(x) for x in norms) or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Nonfinite gradient or unfrozen backbone')
                optimizer.step()
                losses.append(float(loss.detach()))
                gradients.append(norms)
                write(a.output_dir / 'progress.json', {'phase': 'training', 'mode': mode, 'step': step + 1, 'total': 24})
            folder = a.output_dir / mode
            folder.mkdir()
            save_tensor(heads.state_dict(), folder / 'head.pt')
            write(folder / 'training.json', {'schedule': schedule, 'shuffle_seed': shuffle_seed, 'losses': losses, 'gradient_l1': gradients,
                  'checkpoint_sha256': digest(folder / 'head.pt')})
        del model, pairs
        class SeededPilotReadout(PilotReadout):
            def __init__(self, model, heads, mode, sequence):
                super().__init__(model, heads, mode, sequence, shuffle_seed)
        pilot.RegionGate, pilot.NativeCFBIGate = RegionalReadoutHeads, SeededPilotReadout
        baseline = a.output_dir / 'native'
        baseline.mkdir()
        save_tensor(RegionalReadoutHeads().state_dict(), baseline / 'head.pt')
        groups = {}
        for mode in ('native', 'regional', 'shuffled', 'generic'):
            path = a.output_dir / mode / 'head.pt'
            groups[mode] = {}
            for name in names:
                rows, summary = pilot.validate(a, videos[name], repo, 'regional' if mode == 'native' else mode,
                                                name, path, digest(path))
                write(a.output_dir / mode / f'{name}_frames.json', rows)
                groups[mode][name] = summary
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise RuntimeError('Source changed during pilot')
        if inputs != {name: [{'image': digest(f.image_path), 'mask': digest(f.mask_path)} for f in videos[name]] for name in names}:
            raise RuntimeError('Validation input changed during pilot')
        write(a.output_dir / 'results.json', {'native': groups.pop('native'), 'conditions': groups, 'scope': protocol['limitations']})
        write(a.output_dir / 'completion.json', {'complete': True, 'validation_sequences': names})
    except BaseException as error:
        write(a.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

"""Run one matched optimizer update on a captured native training pair."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.native_region_readout import RegionalReadoutHeads
from modeling.paired_readout_training import pair_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pairs', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--shuffle-seed', type=int, default=20261031)
    a = p.parse_args()
    out = validate_output(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = json.loads(a.pairs.read_text())
    if digest(a.checkpoint) != data['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    pair = data['pairs'][0]
    samples = [load(row['path'], row['sha256']) for row in pair['samples']]
    repo, revision = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=a.checkpoint), repo)
    model.requires_grad_(False)
    results = {}
    for mode in ('regional', 'shuffled', 'generic'):
        torch.manual_seed(20261031)
        heads = RegionalReadoutHeads().train()
        optimizer = torch.optim.Adam(heads.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = pair_loss(model, heads, samples, mode, pair['sequence'], check_native=True, shuffle_seed=a.shuffle_seed)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        gradients = [float(h.mix.weight.grad.abs().sum()) for h in heads.layers]
        if not all(torch.isfinite(torch.tensor(g)) and g > 0 for g in gradients):
            raise RuntimeError('Invalid head gradients')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Backbone was not frozen')
        optimizer.step()
        if not all(torch.isfinite(h.mix.weight).all() and h.mix.weight.count_nonzero() > 0 for h in heads.layers):
            raise RuntimeError('Invalid optimizer update')
        results[mode] = {'loss': float(loss.detach()), 'gradient_l1': gradients}
    write(out / 'completion.json', {'passed': True, 'sequence': pair['sequence'],
          'frames': [s['frame'] for s in samples], 'sam2_revision': revision,
          'checkpoint_sha256': data['checkpoint_sha256'], 'shuffle_seed': a.shuffle_seed, 'modes': results,
          'scope': 'one-update engineering gate, not an accuracy result'})


if __name__ == '__main__':
    main()

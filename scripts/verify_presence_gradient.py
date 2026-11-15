"""Check presence-loss gradients on a saved native SAM2 memory read."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.cfbi_training import object_presence_loss, replay_training_read
from modeling.cutie_memory import CutieRegionGate
from scripts.bank_integrity import fingerprint
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import load, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261031)
    row = json.loads(args.samples.read_text())[args.index]
    sample = load(row['path'], row['sha256'])
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(checkpoint=args.checkpoint, method='sam2', device='cpu'), repo)
    gate = CutieRegionGate().train()
    before = fingerprint(sample['output_dict'])['sha256']
    masks, score, trace = replay_training_read(model, gate, sample, 'regional', row['sequence'], return_object_score=True)
    if not torch.equal(masks.detach(), sample['native_logits']):
        raise RuntimeError('Zero-gate logits differ from native')
    target = sample['target']
    loss = object_presence_loss(score, target)
    loss.backward(retain_graph=True)
    grad = gate.gate.IA.weight.grad
    magnitude = float(grad.abs().sum()) if grad is not None else 0.0
    if not torch.isfinite(loss) or not magnitude > 0 or before != fingerprint(sample['output_dict'])['sha256']:
        raise RuntimeError('Presence gradient or memory integrity check failed')
    gate.zero_grad()
    counterfactual = torch.ones_like(target) if not bool(target.gt(0).any()) else torch.zeros_like(target)
    object_presence_loss(score, counterfactual).backward()
    counter_grad = gate.gate.IA.weight.grad
    counter_magnitude = float(counter_grad.abs().sum()) if counter_grad is not None else 0.0
    if not counter_magnitude > 0:
        raise RuntimeError('Counterfactual presence gradient missing')
    write(args.output, {'passed': True, 'sequence': row['sequence'], 'frame': row['frame'],
          'target_present': bool(target.gt(0).any()), 'object_score': float(score.detach().item()),
          'presence_loss': float(loss.detach()), 'gate_gradient_l1': magnitude,
          'counterfactual_target_present': bool(counterfactual.gt(0).any()),
          'counterfactual_gate_gradient_l1': counter_magnitude,
          'layers': trace['layers'], 'native_logits_identical': True, 'bank_unchanged': True})


if __name__ == '__main__':
    main()

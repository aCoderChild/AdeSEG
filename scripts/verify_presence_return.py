"""Verify presence-loss gradients on a previously reproduced missed return."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.cfbi_region_gate import RegionGate
from modeling.cfbi_training import object_presence_loss, replay_training_read
from scripts.bank_integrity import fingerprint
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'replay-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    receipt = json.loads((args.replay_dir / 'results.json').read_text())
    completion = json.loads((args.replay_dir / 'completion.json').read_text())
    if not completion['regional_replay_bit_identical'] or not completion['bank_unchanged']:
        raise RuntimeError('Existing trajectory replay did not pass')
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(checkpoint=args.checkpoint, method='sam2', device='cpu'), repo)
    sample = torch.load(args.replay_dir / 'sample.pt', map_location='cpu', weights_only=False)
    info = json.loads((args.pilot_dir / 'regional' / 'training.json').read_text())
    gate = RegionGate().eval()
    gate.load_state_dict(load(args.pilot_dir / 'regional' / 'head.pt', info['checkpoint_sha256']))
    frames = {**load_manifest('data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest('data/polypgen_sequence.jsonl', 'test')}['seq13']
    target = torch.from_numpy((load_label_mask(frames[66].mask_path, [1]) == 1).copy())[None, None].float()
    before = fingerprint(sample['output_dict'])['sha256']
    masks, score, trace = replay_training_read(model, gate, sample, 'regional', 'seq13', return_object_score=True)
    if not target.any() or score.item() >= 0 or (masks > 0).any():
        raise RuntimeError('Saved return is not a missed positive object')
    if abs(score.item() - receipt['stored_object_score']) > 1e-6:
        raise RuntimeError('Object score differs from verified trajectory')
    loss = object_presence_loss(score, target)
    loss.backward()
    grad = gate.gate.IA.weight.grad
    magnitude = float(grad.abs().sum()) if grad is not None else 0.0
    if not magnitude > 0 or before != fingerprint(sample['output_dict'])['sha256']:
        raise RuntimeError('Gradient or memory integrity check failed')
    write(args.output, {'passed': True, 'sequence': 'seq13', 'frame': 66, 'target_present': True,
          'object_score': float(score.detach().item()), 'presence_loss': float(loss.detach()),
          'gate_gradient_l1': magnitude, 'layers': trace['layers'], 'bank_unchanged': True,
          'scope': 'engineering diagnostic on a previously inspected evaluation sequence'})


if __name__ == '__main__':
    main()

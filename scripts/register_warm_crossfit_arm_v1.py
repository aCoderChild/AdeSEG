#!/usr/bin/env python3
"""Bind an individually saved arm to its already frozen shared fold receipt."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.warm_crossfit_fold_training_v1 import load_preflight, require


def verify_arm(root, preflight_sha, fold, arm):
    root = Path(root).resolve(strict=True)
    group_path = root / 'train' / f'fold{fold}' / 'group.json'
    group_sha = phase_artifact(root, f'train_shared_f{fold}', group_path)
    group = json.loads(group_path.read_text())
    require(digest(group_path) == group_sha and group.get('passed') is True and
            group.get('fold') == fold and group.get('preflight_sha256') == preflight_sha and
            set(group.get('arms', {})) == set(ARMS), 'Shared fold receipt differs')
    for name in ARMS:
        entry = group['arms'][name]
        path = root / 'train' / f'fold{fold}' / name / 'completion.json'
        checkpoint = path.parent / 'checkpoint.pt'
        require(entry.get('path') == str(path) and entry.get('sha256') == digest(path) and
                entry.get('checkpoint_sha256') == digest(checkpoint), 'Shared arm artifacts differ')
        receipt = json.loads(path.read_text())
        require(digest(path) == entry['sha256'] and receipt.get('passed') is True and
                receipt.get('fold') == fold and receipt.get('arm') == name and
                receipt.get('preflight_sha256') == preflight_sha and
                receipt.get('checkpoint_sha256') == entry['checkpoint_sha256'] and
                receipt.get('training_mode') == 'shared_inputs_separate_models_optimizers_and_rng',
                'Shared arm receipt differs')
    require(arm in ARMS and digest(group_path) == group_sha, 'Shared fold changed')
    return group['arms'][arm]['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    _, sha = load_preflight(args.preflight)
    print(json.dumps({'passed': True, 'arm_completion_sha256': verify_arm(args.root, sha, args.fold, args.arm)}))


if __name__ == '__main__': main()

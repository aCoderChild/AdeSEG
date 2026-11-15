#!/usr/bin/env python3
"""Require both seed cohorts to finish label-free replay before any scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.source_group_boundary_multiseed_shared_v1 import ARMS, SEEDS, dependencies, read


def collect(run):
    run = validate_output(run).resolve()
    manifest = run/'manifest/manifest.json'
    manifest_sha = digest(manifest)
    if read(run/'manifest/completion.json').get('manifest_sha256') != manifest_sha:
        raise ValueError('Manifest receipt differs')
    train_paths, blocked_paths = mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    cohorts = {}
    for seed in SEEDS:
        base = run/str(seed)
        plan_path, parity_path = base/'plan/plan.json', base/'parity/parity.json'
        verification_path, done_path = base/'verification/verification.json', base/'verification/completion.json'
        plan, parity, verification, done = map(read, (plan_path, parity_path, verification_path, done_path))
        if (not plan.get('passed') or plan.get('seed') != seed or plan.get('arms') != list(ARMS)
                or plan.get('python_hash_seed') != '0' or plan.get('study_seed_environment') != str(seed)
                or plan.get('manifest_sha256') != manifest_sha or plan.get('dependency_sha256') != dependencies()
                or not parity.get('passed') or parity.get('manifest_sha256') != manifest_sha
                or parity.get('plan_sha256') != digest(plan_path)
                or not verification.get('passed') or not done.get('passed')
                or done.get('verification_sha256') != digest(verification_path)
                or verification.get('manifest_sha256') != manifest_sha
                or verification.get('plan_sha256') != digest(plan_path)
                or verification.get('parity_sha256') != digest(parity_path)
                or verification.get('dependency_sha256') != dependencies()
                or verification.get('runner_sha256') != digest(ROOT/'scripts/verify_source_group_boundary_multiseed_v1.py')
                or verification.get('label_guard', {}).get('open_attempts') != 0
                or verification.get('label_guard', {}).get('blocked_paths') != len(train_paths | blocked_paths)
                or set(verification.get('runs', {})) != set(ARMS)):
            raise ValueError(f'{seed}: plan, parity or replay evidence differs')
        arms = {}
        for arm in ARMS:
            folder = base/'runs'/arm
            ev = verification['runs'][arm]
            completion, checkpoint, frozen = folder/'completion.json', folder/'checkpoint.pt', folder/'frozen_predictions.json'
            if (Path(ev['completion_path']).resolve() != completion.resolve()
                    or ev['completion_sha256'] != digest(completion)
                    or ev['checkpoint_sha256'] != digest(checkpoint)
                    or ev['frozen_sha256'] != digest(frozen)):
                raise ValueError(f'{seed}/{arm}: replayed artifacts differ')
            arms[arm] = {'completion_sha256': digest(completion), 'checkpoint_sha256': digest(checkpoint),
                         'frozen_sha256': digest(frozen)}
        cohorts[str(seed)] = {'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path),
                              'verification_sha256': digest(verification_path), 'arms': arms,
                              'training_order': plan['training_order']}
    if cohorts[str(SEEDS[0])]['training_order'] == cohorts[str(SEEDS[1])]['training_order']:
        raise ValueError('Seed training orders unexpectedly identical')
    return {'manifest_sha256': manifest_sha, 'cohorts': cohorts}


def check_joint_gate(path, run):
    path = Path(path).resolve()
    run = validate_output(run).resolve()
    if path != run/'joint_gate/joint.json':
        raise ValueError('Joint gate path differs')
    saved = read(path)
    done = read(path.parent/'completion.json')
    if (not saved.get('passed') or not done.get('passed') or done.get('joint_sha256') != digest(path)
            or saved.get('runner_sha256') != digest(Path(__file__))
            or saved.get('dependency_sha256') != dependencies()
            or saved.get('evidence') != collect(run)):
        raise ValueError('Joint replay gate differs')
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    guard = install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    evidence = collect(args.run)
    if guard['open_attempts'] != 0:
        raise ValueError('Label mask opened during joint gate')
    result = {'passed': True, 'runner_sha256': digest(Path(__file__)), 'dependency_sha256': dependencies(),
              'label_guard': dict(guard), 'evidence': evidence}
    output = validate_output(args.run)/'joint_gate'
    output.mkdir(parents=True, exist_ok=False)
    path = output/'joint.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'passed': True, 'joint_sha256': digest(path)}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'seeds': list(SEEDS), 'label_open_attempts': guard['open_attempts']}))


if __name__ == '__main__':
    main()

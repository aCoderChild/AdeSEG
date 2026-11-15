#!/usr/bin/env python3
"""Reconstruct a mixed registry in a separate, label-blocked process.

This verifier does not authorize or start training. Its receipt is one of
several required inputs to the separate bounded shared-worker gate.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_bounded_crossfit_registry_v1 as registry
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def sources():
    values = registry.sources()
    for name in local_import_closure(ROOT, ('scripts/verify_bounded_crossfit_registry_v1.py',)):
        value = digest(ROOT / name)
        require(name not in values or values[name] == value, f'Conflicting source identity: {name}')
        values[name] = value
    return values


def verify(folder):
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Registry outside Drive experiment root')
    saved, registry_sha = read_bound_json(folder / 'registry.json')
    done, completion_sha = read_bound_json(folder / 'completion.json')
    require(saved.get('passed') is True and saved.get('training_authorized') is False and
            saved.get('format') == 'bounded_crossfit_registry_v1' and
            saved.get('label_guard', {}).get('open_attempts') == 0 and
            done.get('passed') is True and done.get('training_authorized') is False and
            done.get('registry_sha256') == registry_sha and
            done.get('label_guard', {}).get('open_attempts') == 0,
            'Mixed registry or non-training completion differs')
    rebuilt = registry.build(saved['preflight_path'])
    require({k: v for k, v in saved.items() if k != 'label_guard'} == rebuilt,
            'Mixed registry reconstruction differs')
    require(digest(folder / 'registry.json') == registry_sha and
            digest(folder / 'completion.json') == completion_sha, 'Mixed registry changed during verification')
    return {'passed': True, 'training_authorized': False, 'format': 'bounded_registry_verification_v1',
            'registry_path': str(folder), 'registry_sha256': registry_sha,
            'registry_completion_sha256': completion_sha, 'preflight_sha256': saved['preflight_sha256'],
            'preflight_completion_sha256': saved['preflight_completion_sha256'],
            'handoff_sha256': saved['handoff_sha256'], 'seq6_proof_sha256': saved['seq6_proof_sha256'],
            'seq11_proof_sha256': saved['seq11_proof_sha256'], 'source_sha256': sources(),
            'scope': 'Independent label-blocked reconstruction; no training or MPS probe.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA)
    frozen = sources(); result = verify(args.registry)
    require(frozen == result['source_sha256'] == sources() and guard['open_attempts'] == 0,
            'Sources changed or target label opened')
    result['label_guard'] = dict(guard)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'training_authorized': False, 'output': str(out)}))


if __name__ == '__main__':
    main()

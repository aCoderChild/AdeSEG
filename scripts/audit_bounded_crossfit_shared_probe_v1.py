#!/usr/bin/env python3
"""CPU audit of frozen bounded shared-probe receipts and checkpoints.

This checks persisted source/provenance, all twelve update records, nonzero
saved head updates, finite weights and strict checkpoint loading. It does
not compare CPU and MPS inference or independently repeat the MPS optimizer.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_v1 as worker
from scripts import verify_cache_release_crossfit_registry_v1 as verifier
from scripts.channel_value_fixed_read_v1 import model_for
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def audit(registry_folder, verification_path, probe_path):
    folder = Path(registry_folder).resolve(strict=True)
    verification_path = Path(verification_path).resolve(strict=True)
    probe_path = Path(probe_path).resolve(strict=True)
    require(all(x.is_relative_to(BASE.resolve(strict=True)) for x in (folder, verification_path, probe_path)),
            'Probe evidence outside Drive experiment root')
    registry, registry_sha = read_bound_json(folder / 'registry.json')
    verification, verification_sha = read_bound_json(verification_path)
    rebuilt = verifier.verify(folder)
    require(verification.get('label_guard', {}).get('open_attempts') == 0 and
            {k: v for k, v in verification.items() if k != 'label_guard'} == rebuilt and
            rebuilt['registry_sha256'] == registry_sha, 'Registry verification differs')
    probe, probe_sha = read_bound_json(probe_path)
    identity = worker.sources()
    require(probe.get('passed') is True and probe.get('format') == 'bounded_shared_six_arm_probe_v1' and
            probe.get('device') == 'mps' and probe.get('registry_sha256') == registry_sha and
            probe.get('verification_sha256') == verification_sha and
            probe.get('preflight_sha256') == registry['preflight_sha256'] and
            probe.get('runner_sha256') == digest(ROOT / 'scripts/probe_bounded_crossfit_shared_v1.py') and
            probe.get('source_sha256') == identity and probe.get('environment') == registry['environment'] and
            probe.get('label_guard', {}).get('open_attempts') == 0 and
            type(probe.get('peak_process_rss_bytes')) is int and type(probe.get('max_process_rss_bytes')) is int and
            0 < probe['peak_process_rss_bytes'] <= probe['max_process_rss_bytes'] <= 6 * 1024**3,
            'Probe source, resource evidence or provenance differs')
    expected_keys = {(row['sequence'], row['frame']) for row in registry['samples']}
    checks = probe.get('checks'); require(isinstance(checks, list) and len(checks) == 2 * len(ARMS), 'Probe count differs')
    seen = set()
    for check in checks:
        pair = (check['sequence'], check['arm'])
        require(pair not in seen and pair[0] in ('seq6', 'seq23') and pair[1] in ARMS and
                (pair[0], check['frame']) in expected_keys and check.get('optimizer_updated') is True and
                (pair[0] != 'seq23' or check.get('video_shape') == [720, 834]) and
                check.get('initial_state_sha256') == registry['initial_state_sha256'][pair[1]] and
                isinstance(check.get('synthetic_loss'), (int, float)) and math.isfinite(check['synthetic_loss']) and
                all(isinstance(check.get(k), (int, float)) and math.isfinite(check[k]) and check[k] > 0
                    for k in ('gradient_l2', 'head_bias_gradient_abs')), 'Probe update record differs')
        path = Path(check['checkpoint_path']).resolve(strict=True)
        require(path.is_relative_to(probe_path.parent) and digest(path) == check['checkpoint_sha256'],
                'Probe checkpoint path or hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(digest(path) == check['checkpoint_sha256'] and payload.get('crossfit_fold') == 1 and
                payload.get('crossfit_arm') == pair[1] and payload.get('training_steps') == 1 and
                payload.get('bounded_registry_sha256') == registry_sha and
                payload.get('aggregate_sha256') == registry['handoff_sha256'] and
                payload.get('source_sha256_map') == identity, 'Probe checkpoint metadata differs')
        model = model_for(pair[1])
        model.load_trained_state(payload, backbone_sha256=registry['backbone_sha256'],
            feature_pipeline_sha256=registry['feature_pipeline_sha256'],
            train_manifest_sha256=registry['preflight_sha256'])
        require(bool((model.head[-1].bias.detach().abs() > 0).any()), 'Probe checkpoint has no head update')
        seen.add(pair)
    require(seen == {(sequence, arm) for sequence in ('seq6', 'seq23') for arm in ARMS} and
            digest(probe_path) == probe_sha and digest(verification_path) == verification_sha and
            identity == worker.sources(), 'Probe identities changed')
    return {'passed': True, 'format': 'bounded_shared_six_arm_probe_audit_v1',
        'scope': 'CPU provenance/weight audit only; MPS optimizer is not independently rerun.',
        'registry_sha256': registry_sha, 'verification_sha256': verification_sha,
        'probe_sha256': probe_sha, 'source_sha256': identity, 'auditor_sha256': digest(Path(__file__)),
        'peak_process_rss_bytes': probe['peak_process_rss_bytes'],
        'max_process_rss_bytes': probe['max_process_rss_bytes'],
        'audited_checks': len(seen), 'error_count': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'verification', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    require(output.is_relative_to(BASE.resolve(strict=True)) and not output.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA); result = audit(args.registry, args.verification, args.probe)
    require(guard['open_attempts'] == 0, 'Target label opened during probe audit')
    result['label_guard'] = dict(guard)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'audited_checks': result['audited_checks'], 'output': str(output)}))


if __name__ == '__main__':
    main()

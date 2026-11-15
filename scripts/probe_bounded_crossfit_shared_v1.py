#!/usr/bin/env python3
"""Label-free same-MPS probe of the bounded shared update/checkpoint path.

This is disabled pending complete real capture proofs and independent source
review. Synthetic targets are derived from logits; no annotation is opened.
"""
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_v1 as worker
from scripts.bounded_crossfit_supervision_v1 import assert_supervised
from scripts import verify_cache_release_crossfit_registry_v1 as verifier
from scripts.channel_value_fixed_read_v1 import model_for, require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA
from scripts.crossfit_training_contract_v1 import restore_view_mask
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_fold_training_v1 import inputs
from scripts.warm_crossfit_shared_fold_v1 import make_arms, step_arm


ENABLE_REVIEWED_PROBE = True


def checked_registry(folder, verification_path):
    folder = Path(folder).resolve(strict=True); verification_path = Path(verification_path).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)) and verification_path.is_relative_to(BASE.resolve(strict=True)),
            'Registry verification outside Drive experiment root')
    saved, registry_sha = read_bound_json(folder / 'registry.json')
    evidence, evidence_sha = read_bound_json(verification_path)
    reconstructed = verifier.verify(folder)
    require(evidence.get('label_guard', {}).get('open_attempts') == 0 and
            {k: v for k, v in evidence.items() if k != 'label_guard'} == reconstructed and
            evidence['registry_sha256'] == registry_sha and digest(verification_path) == evidence_sha,
            'Label-blocked registry verification differs')
    return saved, registry_sha, evidence_sha


def probe(registry, registry_sha, verification_sha, output, guard, *, batch_root, device='mps', max_rss_bytes=6 * 1024**3):
    require(ENABLE_REVIEWED_PROBE, 'Bounded MPS probe awaits independent review and real input proofs')
    require(device == 'mps', 'Probe must use MPS')
    require(type(max_rss_bytes) is int and 0 < max_rss_bytes <= 6 * 1024**3, 'Invalid probe RSS limit')
    manifest = assert_supervised(batch_root, 'probe', output)
    require(max_rss_bytes == manifest['max_process_rss_bytes'], 'Probe RSS cap differs from supervisor')
    identity = worker.sources(); preflight_sha = registry['preflight_sha256']
    rows = {(x['sequence'], x['frame']): x for x in registry['samples']}
    require(len(rows) == 357, 'Mixed sample coverage differs')
    keys = []
    for sequence in ('seq6', 'seq23'):
        candidates = sorted(key for key in rows if key[0] == sequence)
        require(candidates, f'Missing probe source: {sequence}')
        keys.append(candidates[0])
    output.mkdir(parents=True, exist_ok=False); checks = []
    authorization = {'preflight_sha256': preflight_sha, 'handoff_sha256': registry['handoff_sha256'],
                     'registry_sha256': registry_sha, 'source_sha256': identity}
    for key in keys:
        values, shape = inputs(rows[key], device)
        require(key[0] != 'seq23' or shape == (720, 834), 'Probe viewport differs')
        original_shape = (720, 1280) if key[0] == 'seq23' else shape
        require(restore_view_mask(np.zeros(shape, dtype=bool), key[0], original_shape).shape == original_shape,
                'Probe restored geometry differs')
        target = (to_video(values[2], shape) <= 0).float()
        models, optimizers, rng, initial = make_arms(registry, device)
        for arm in ARMS:
            model = models[arm]
            prediction = model(*values)
            require(torch.isfinite(prediction).all() and torch.equal(prediction, values[2]),
                    'Initial residual is not a finite no-op')
            del prediction
            loss, norm, rng[arm] = step_arm(model, optimizers[arm], rng[arm], values, target, shape, device)
            grad = model.head[-1].bias.grad
            require(grad is not None and torch.isfinite(grad).all() and bool((grad.abs() > 0).any()) and
                    norm > 0 and np.isfinite(loss) and bool((model.head[-1].bias.detach().abs() > 0).any()),
                    'Shared probe gradient or optimizer update differs')
            before_reload = model(*values).detach()
            require(torch.isfinite(before_reload).all(), 'Updated model is nonfinite')
            checkpoint = output / f'{key[0]}_{arm}.pt'
            torch.save(worker.save_payload(model, arm=arm, fold=1, registry=registry,
                                           authorization=authorization, steps=1), checkpoint)
            checkpoint_sha = digest(checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(digest(checkpoint) == checkpoint_sha, 'Probe checkpoint changed during load')
            restored = model_for(arm)
            restored.load_trained_state(payload, backbone_sha256=registry['backbone_sha256'],
                feature_pipeline_sha256=registry['feature_pipeline_sha256'], train_manifest_sha256=preflight_sha)
            restored.to(device).eval().requires_grad_(False)
            with torch.no_grad():
                after_reload = restored(*values)
                require(torch.isfinite(after_reload).all() and
                        torch.allclose(before_reload, after_reload, rtol=1e-5, atol=1e-6),
                        'Reloaded shared checkpoint output differs')
            checks.append({'arm': arm, 'sequence': key[0], 'frame': key[1], 'video_shape': list(shape),
                'head_bias_gradient_abs': float(grad.abs().max().detach().cpu()), 'gradient_l2': norm,
                'synthetic_loss': loss, 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                'initial_state_sha256': initial[arm], 'optimizer_updated': True})
            del restored, before_reload, after_reload
        del models, optimizers, rng, values, target
    require(guard['open_attempts'] == 0 and identity == worker.sources(), 'Annotation opened or source changed')
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    require(0 < rss_bytes <= max_rss_bytes, 'Probe recorded high-water RSS exceeded its limit')
    result = {'passed': True, 'format': 'bounded_shared_six_arm_probe_v1', 'device': device,
        'scope': 'Label-free MPS optimizer/checkpoint probe; no cross-device or efficacy claim.',
        'registry_sha256': registry_sha, 'verification_sha256': verification_sha,
        'preflight_sha256': preflight_sha, 'source_sha256': identity, 'runner_sha256': digest(Path(__file__)),
        'environment': registry['environment'], 'checks': checks, 'label_guard': dict(guard),
        'peak_process_rss_bytes': rss_bytes, 'max_process_rss_bytes': max_rss_bytes}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True, type=Path)
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-root', required=True, type=Path)
    parser.add_argument('--max-process-rss-bytes', type=int, default=6 * 1024**3)
    args = parser.parse_args(); output = args.output.resolve()
    require(output.is_relative_to(BASE.resolve(strict=True)) and not output.exists(), 'Invalid output')
    require(ENABLE_REVIEWED_PROBE, 'Bounded MPS probe awaits independent review and real input proofs')
    manifest = assert_supervised(args.batch_root, 'probe', output)
    require(args.max_process_rss_bytes == manifest['max_process_rss_bytes'], 'Probe RSS cap differs from supervisor')
    guard = install_mask_open_guard(DATA)
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    registry, registry_sha, verification_sha = checked_registry(args.registry, args.verification)
    require(environment_identity() == registry['environment'], 'MPS/runtime environment differs from capture')
    probe(registry, registry_sha, verification_sha, output, guard, batch_root=args.batch_root,
          max_rss_bytes=args.max_process_rss_bytes)
    print(json.dumps({'passed': True, 'output': str(output)}))


if __name__ == '__main__':
    main()

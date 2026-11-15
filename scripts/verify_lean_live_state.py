#!/usr/bin/env python3
"""Audit live token-window model tensor state during a verified MPS replay.

This is an engineering prerequisite, not a live-refiner authorization. It owns
the model construction, verifies the historical capture assets and refiner
checkpoint, compares CPU/MPS loaded parameters and buffers, and checks them
after the existing seq4 replay. It does not install the learned refiner.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_channel_attention import author_sha256
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.verify_refiner_capture_static import verify as verify_capture
import scripts.verify_lean_window_mps_replay as replay

FORK = ROOT / 'external/Lean-SAM2-token-window'


def fingerprint_digest(fingerprints):
    raw = json.dumps(fingerprints, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'split-preflight', 'static-capture', 'seq4-capture',
                 'checkpoint', 'detector', 'refiner-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0':
        raise ValueError('Explicitly disable MPS fallback and fast math')

    static = verify_capture(args.preflight, args.split_preflight, args.static_capture,
                            args.checkpoint, args.detector)
    manifest = json.loads((args.preflight / 'manifest.json').read_text())
    payload = torch.load(args.refiner_checkpoint, map_location='cpu', weights_only=True)
    if payload.get('model_source_sha256') != digest(ROOT / 'modeling/multilevel_contrast_refiner.py'):
        raise ValueError('Refiner source differs from its checkpoint')
    if payload.get('author_sha256') != author_sha256():
        raise ValueError('Refiner author-gate source differs')
    refiner = MultilevelContrastRefiner()
    refiner.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
                               feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
                               train_manifest_sha256=static['manifest_sha256'])
    refiner.eval()
    refiner_fingerprint = module_tensor_fingerprints(refiner)

    seq4 = json.loads((args.seq4_capture / 'channel_feature_receipt.json').read_text())
    if seq4['feature_pipeline_sha256'] != static['feature_pipeline_sha256']:
        raise ValueError('Sequence capture and refiner pipeline differ')
    sys.path.insert(0, str(FORK))
    reference_model = build_author_predictor(
        SimpleNamespace(method='lean', checkpoint=args.checkpoint, device='cpu'), FORK)
    reference = module_tensor_fingerprints(reference_model)
    del reference_model
    gc.collect()

    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    original_build = replay.build_author_predictor
    built = []

    def checked_build(*build_args, **build_kwargs):
        model = original_build(*build_args, **build_kwargs)
        if built:
            raise RuntimeError('Expected exactly one live predictor')
        before = module_tensor_fingerprints(model)
        if before != reference:
            mismatch = sorted(k for k in set(before) | set(reference) if before.get(k) != reference.get(k))
            raise ValueError(f'Loaded CPU/MPS model state differs: {mismatch[:8]}')
        built.append((model, before))
        return model

    replay.build_author_predictor = checked_build
    previous_args = sys.argv
    try:
        sys.argv = [str(replay.__file__), '--device', 'mps', '--checkpoint', str(args.checkpoint),
                    '--capture-reference', str(args.seq4_capture), '--preflight', str(args.split_preflight),
                    '--output', str(out / 'replay'), '--end-frame', '20']
        replay.main()
    finally:
        sys.argv = previous_args
        replay.build_author_predictor = original_build
    if len(built) != 1:
        raise RuntimeError('No live predictor was constructed')
    model, before = built[0]
    after = module_tensor_fingerprints(model)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed:
        raise RuntimeError(f'Live predictor parameters or buffers changed: {changed[:8]}')
    result = {
        'passed': True,
        'scope': 'Engineering state audit only; no learned-refiner video execution or authorization.',
        'static_capture': static,
        'checkpoint_sha256': digest(args.checkpoint),
        'refiner_checkpoint_sha256': digest(args.refiner_checkpoint),
        'refiner_model_source_sha256': digest(ROOT / 'modeling/multilevel_contrast_refiner.py'),
        'refiner_state_sha256': fingerprint_digest(refiner_fingerprint),
        'reference_tensor_count': len(reference),
        'reference_state_sha256': fingerprint_digest(reference),
        'live_before_state_sha256': fingerprint_digest(before),
        'live_after_state_sha256': fingerprint_digest(after),
        'state_unchanged': True,
        'replay_receipt_sha256': digest(out / 'replay/receipt.json'),
        'auditor_source_sha256': digest(Path(__file__)),
        'limitations': ['This check does not verify a trained refiner during propagation.',
                        'Hooks, mutable non-tensor state and the full cohort require separate verification.',
                        'CPU/MPS feature shift must be evaluated rather than assumed absent.'],
    }
    (out / 'cpu_reference_fingerprints.json').write_text(json.dumps(reference, indent=2) + '\n')
    (out / 'refiner_fingerprints.json').write_text(json.dumps(refiner_fingerprint, indent=2) + '\n')
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'reference_tensor_count', 'state_unchanged', 'scope')}))


if __name__ == '__main__':
    main()

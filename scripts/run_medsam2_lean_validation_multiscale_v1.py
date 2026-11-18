#!/usr/bin/env python3
"""Validate MedSAM2+Lean with multi-scale memory features hook.

Extends the three-arm validation to include a multi-scale memory attention arm:
early/mid FPN features are injected as additional memory key/value tokens,
bounded by Lean-SAM2 TAMP pruning (mem_keep_ratio=0.25).

Writes predictions and receipts, never labels or scores. Runs on predeclared
sequences including difficult targets (seq11, seq20, seq23) and preserved
sequences (seq8, seq14).
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.early_deep_prompt_tokens_v1 import FeatureTokenAdapter
from modeling.lean_indexed_spatial_rope_mps_dynamic_v1 import indexed_spatial_rope_mps_dynamic
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.medsam2_lean_memory_builder_v1 import build_medsam2_lean_memory, _validate_sources
from modeling.multi_scale_memory_features_v1 import MultiScaleMemoryFeatures, MultiScaleMemoryHook
from modeling.sam2_lazy_video_frames_v1 import install_lazy_video_frames
from modeling.seq23_viewport import restore_mask, stage_cropped_frames
from scripts.kvasir_prompt_adapter_v1 import _runtime
from scripts.memory_token_io import digest
from scripts.run_author_lean_prompt_batch_v1 import exclusive_lock
from scripts.run_native_default_auto_video_v1 import DRIVE, MANIFEST, model_hash, tensor_hash, resource_guard
from scripts.source_group_label_guard_v1 import install_mask_open_guard

FORMAT = 'medsam2_lean_validation_multiscale_v1'

# Difficult sequences to improve + good sequences to preserve
SEQUENCES = ('seq5', 'seq8', 'seq10', 'seq11', 'seq14', 'seq15', 'seq17', 'seq20', 'seq23')


def sources():
    paths = [Path(__file__),
             ROOT/'modeling/medsam2_lean_memory_builder_v1.py',
             ROOT/'modeling/multi_scale_memory_features_v1.py',
             ROOT/'modeling/lean_mps_blocking_pointer_position.py',
             ROOT/'modeling/sam2_lazy_video_frames_v1.py',
             ROOT/'modeling/seq23_viewport.py',
             ROOT/'scripts/run_native_default_auto_video_v1.py',
             ROOT/'scripts/run_author_lean_prompt_batch_v1.py',
             ROOT/'scripts/source_group_label_guard_v1.py',
             ROOT/'scripts/memory_token_io.py',
             ROOT/'datasets/video.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def _load_adapter(checkpoint_path, device='mps'):
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    adapter = FeatureTokenAdapter('both', grid_size=2).eval().requires_grad_(False)
    adapter.load_state_dict(raw['state_dict'], strict=True)
    adapter = adapter.to(device)
    return adapter


def run_sequence(model, adapter, sequence, frames, start, box_viewport,
                 output_dir, use_multiscale=True, device='mps'):
    """Run single-sequence validation with optional multi-scale memory hook."""
    from scripts.memory_token_io import digest as _digest

    guard = install_mask_open_guard(MANIFEST)

    output_dir = Path(output_dir)
    seq_dir = output_dir / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with exclusive_lock(DRIVE / '.matched_native_author_mps.lock'), _runtime(device):
        resource_guard()
        model, config = build_medsam2_lean_memory(device)
        before = model_hash(model)
        adapter_before = model_hash(adapter)

        aggregator = MultiScaleMemoryFeatures().to(device).eval().requires_grad_(False)
        hook = MultiScaleMemoryHook(model, aggregator)

        box = np.asarray(box_viewport, dtype=np.float32)

        with hook:
            hook.clear_multi_scale_tokens()
            _, ids, _ = model.add_new_points_or_box(
                None, frame_idx=start, obj_id=1, box=box)
            if ids != [1]:
                raise ValueError('unexpected object id')

            for i, ids_out, logits in model.propagate_in_video(
                    None, start_frame_idx=start, reverse=False):
                if ids_out != [1]:
                    raise ValueError('unexpected object id')

                mask = (logits[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                pred_path = seq_dir / f'{i:06d}.png'
                Image.fromarray((mask * 255).astype(np.uint8)).save(pred_path)

                records.append({
                    'frame': i,
                    'prediction_sha256': _digest(pred_path),
                    'image_sha256': _digest(frames[i].image_path),
                    'logits_sha256': tensor_hash(logits),
                })

            torch.mps.synchronize()
            torch.mps.empty_cache()

        if model_hash(model) != before:
            raise ValueError('frozen predictor parameters changed')
        if model_hash(adapter) != adapter_before:
            raise ValueError('adapter changed during inference')

    if guard['open_attempts']:
        raise ValueError('label guard triggered during inference')

    receipt = {
        'format': FORMAT + '_receipt',
        'passed': True,
        'sequence': sequence,
        'use_multiscale': use_multiscale,
        'image_size': 512,
        'prompt_frame': start,
        'tarr_enabled': False,
        'records': records,
        'model_state_sha256': model_hash(model),
        'adapter_state_sha256': adapter_before,
        'label_guard': dict(guard),
        'source_sha256': sources(),
        'limitations': [
            'Exploratory multi-scale memory attention arm.',
            'TARR disabled; no Tiny bypass checkpoint released.',
            'Not a 21-sequence independent confirmation.',
        ],
    }
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--adapter-checkpoint', required=True,
                     help='Path to trained adapter checkpoint (.pt)')
    run.add_argument('--output-root', required=True,
                     help='Output directory root on Drive')
    run.add_argument('--sequences', nargs='+', default=list(SEQUENCES),
                     help='Sequences to validate (default: all target sequences)')
    run.add_argument('--device', default='mps',
                     help='Device: mps (default) or cpu')
    args = p.parse_args()

    if args.command == 'run':
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        adapter = _load_adapter(args.adapter_checkpoint, device=args.device)

        dev, test = load_manifest(MANIFEST, 'dev'), load_manifest(MANIFEST, 'test')
        videos = {**dev, **test}

        for seq in args.sequences:
            if seq not in videos:
                raise ValueError(f'Sequence {seq} not in manifest')
            frames = list(videos[seq])
            if len(frames) < 2:
                print(f'  {seq}: insufficient frames ({len(frames)})')
                continue

            start = 0
            box_viewport = json.loads(
                (DRIVE / 'kvasir_prompt_polypgen_composite_20261116_v2' /
                 'baseline' / 'seq23' / 'receipt.json').read_text())['box_xyxy_viewport']
            box_viewport = [float(v) for v in box_viewport]

            print(f'  Validating {seq} (total {len(frames)} frames, start={start})...')
            receipt = run_sequence(
                model=None, adapter=adapter, sequence=seq,
                frames=frames, start=start, box_viewport=box_viewport,
                output_dir=output_root / 'predictions', use_multiscale=True,
                device=args.device)
            receipt_path = output_root / f'{seq}_receipt.json'
            with receipt_path.open('w') as f:
                json.dump(receipt, f, indent=2, sort_keys=True)
                f.write('\n')
            print(f'    receipt saved to {receipt_path}')

        print('\nValidation completed.')
        print(f'Results stored at: {output_root}')


if __name__ == '__main__':
    main()
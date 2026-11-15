#!/usr/bin/env python3
"""Automatic YOLO + early-feature prompt + Lean multiscale memory experiment."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest, stage_frames  # noqa: E402
from evaluation.segmentation import calculate_scores  # noqa: E402
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from modeling.lean_multiscale_anchor import install_multiscale_anchor  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import (  # noqa: E402
    MODEL_CONFIG, assert_parameter_integrity, build_author_predictor,
    parameter_fingerprints, source_identity,
)
from scripts.run_prompt_feature_refinement import detect_box  # noqa: E402

MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'
PROTOCOL = ROOT / 'protocols/lean_auto_multiscale_early_prompt_v1.json'
MULTISCALE_PROTOCOL = ROOT / 'protocols/lean_multiscale_anchor_v1.json'
NAMES = [f'seq{i}' for i in range(2, 24) if i != 7]


def load_gate(path, identity, mode):
    setup = json.loads((path / 'setup.json').read_text())
    done = json.loads((path / 'completion.json').read_text())
    if setup['signature'] != identity or setup['mode'] != mode or not done['passed'] or digest(path / 'records.json') != done['records_sha256']:
        raise ValueError(f'Gate integrity or identity mismatch: {path}')
    return json.loads((path / 'records.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--detector', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mode', required=True, choices=('gate', 'cohort'))
    p.add_argument('--sequence', action='append', required=True)
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    p.add_argument('--memory-gate-a', required=True, type=Path)
    p.add_argument('--memory-gate-b', required=True, type=Path)
    args = p.parse_args()
    if len(set(args.sequence)) != len(args.sequence) or not set(args.sequence).issubset(NAMES):
        raise ValueError('Invalid or duplicate sequence')
    repo, revision = source_identity('lean')
    auto_identity = {'revision': revision, 'checkpoint_sha256': digest(args.checkpoint),
        'detector_sha256': digest(args.detector), 'manifest_sha256': digest(MANIFEST),
        'protocol_sha256': digest(PROTOCOL), 'memory_protocol_sha256': digest(MULTISCALE_PROTOCOL),
        'early_module_sha256': digest(ROOT / 'modeling/early_feature_prompt.py'),
        'prompt_module_sha256': digest(ROOT / 'modeling/prompt_feature_refinement.py'),
        'memory_module_sha256': digest(ROOT / 'modeling/lean_multiscale_anchor.py'),
        'lean_runner_sha256': digest(ROOT / 'scripts/run_author_baseline.py'), 'runner_sha256': digest(Path(__file__)),
        'bypass_sha256': digest(repo / 'checkpoints/bypass_bottleneck_base.pth'),
        'lean_config_sha256': digest(repo / 'configs/lean_sam2/lean_sam2.yaml'),
        'torch': torch.__version__, 'ultralytics': version('ultralytics'),
        'numpy': np.__version__, 'python': sys.version, 'device': 'cpu float32; 2 threads'}
    memory_gates = []
    for path in (args.memory_gate_a, args.memory_gate_b):
        setup = json.loads((path / 'setup.json').read_text())
        receipt = json.loads((path / 'completion.json').read_text())
        sig = setup['signature']
        if (setup['mode'] != 'gate' or not receipt['passed'] or digest(path / 'records.json') != receipt['records_sha256']
            or sig['implementation_sha256'] != auto_identity['memory_module_sha256']
            or sig['lean']['checkpoint_sha256'] != auto_identity['checkpoint_sha256']):
            raise ValueError('Memory no-op gate is incomplete or differs')
        memory_gates.append(json.loads((path / 'records.json').read_text()))
    if memory_gates[0] != memory_gates[1] or not memory_gates[0]:
        raise ValueError('Memory gates differ')
    if args.mode == 'cohort':
        if args.gate_a is None or args.gate_b is None:
            raise ValueError('Two automatic-system gates are required')
        a = load_gate(args.gate_a, auto_identity, 'gate')
        b = load_gate(args.gate_b, auto_identity, 'gate')
        if not a or a != b or {r['sequence'] for r in a} != {'seq3', 'seq20'}:
            raise ValueError('Automatic system gates differ or lack required coverage')
    output = validate_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings_dir = output / 'ultralytics'
    settings_dir.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(settings_dir)
    os.environ['WANDB_MODE'] = 'disabled'
    from ultralytics import YOLO
    detector = YOLO(str(args.detector))
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    torch.set_num_threads(2)
    (output / 'setup.json').write_text(json.dumps({'signature': auto_identity, 'mode': args.mode,
        'sequences': args.sequence, 'initialization': 'first chronological YOLO detection; no GT prompt'}, indent=2) + '\n')
    records = []
    for name in args.sequence:
        frames = videos[name]
        folder = output / name
        folder.mkdir()
        # The detection path uses images only. Labels are first opened after all
        # predictions for the sequence have been saved.
        first = None
        box = confidence = None
        for i, frame in enumerate(frames):
            image = np.asarray(Image.open(frame.image_path).convert('RGB'))
            box, confidence = detect_box(detector, image)
            if box is not None:
                first = i
                break
        events, selections = [], []
        prompt_info = {'first_detection_index': first, 'box': box, 'confidence': confidence,
                       'point_status': 'no_detection', 'points': None, 'lean_mask_status': 'no_detection'}
        if first is not None:
            predictor = build_author_predictor(SimpleNamespace(method='lean', checkpoint=args.checkpoint, device='cpu'), repo)
            before = parameter_fingerprints(predictor)
            events = install_multiscale_anchor(predictor, .5, selection_events=selections)
            with tempfile.TemporaryDirectory(prefix='ordered_frames_', dir=folder) as temporary:
                stage_frames(frames, Path(temporary))
                staged = sorted(Path(temporary).glob('*.jpg'), key=lambda path: int(path.stem))
                if len(staged) != len(frames) or any(path.resolve() != frame.image_path.resolve() for path, frame in zip(staged, frames)):
                    raise RuntimeError('Staged image order differs from manifest')
                state = predictor.init_state(video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False)
                if state['num_frames'] != len(frames):
                    raise RuntimeError('Lean video loader frame count differs')
                try:
                    with torch.inference_mode():
                        _, ids, initial = predictor.add_new_points_or_box(state, frame_idx=first, obj_id=1,
                                                                         box=np.asarray(box, dtype=np.float32))
                        if ids != [1]:
                            raise RuntimeError('Unexpected object ID')
                        provisional = (initial[0, 0].detach().cpu().numpy() > 0)
                        Image.fromarray(provisional.astype(np.uint8) * 255).save(folder / 'prompt_provisional.png')
                        prompt_info['provisional_mask_sha256'] = digest(folder / 'prompt_provisional.png')
                        _, backbone_out, _, _, _ = predictor._get_image_feature(state, first, batch_size=1)
                        early = backbone_out['backbone_fpn'][0].detach().float().cpu().contiguous()
                        prompt_info['feature_shape'] = list(early.shape)
                        prompt_info['early_feature_sha256'] = hashlib.sha256(early.numpy().tobytes()).hexdigest()
                        points, status = early_feature_points(backbone_out, provisional, box, seed=20261114)
                        prompt_info['point_status'] = status
                        if points is not None:
                            prompt_info['points'] = points['feature_points']
                            _, _, initial = predictor.add_new_points_or_box(state, frame_idx=first, obj_id=1,
                                points=np.asarray(points['feature_points'], dtype=np.float32),
                                labels=np.array([1, 0], dtype=np.int32), clear_old_points=False)
                        # Released Lean initializes its anchor bank from mask
                        # prompts, not point/box prompts. Use the automatic
                        # refined mask, never a label, as its conditioning mask.
                        initial_mask = (initial[0, 0].detach().cpu().numpy() > 0)
                        Image.fromarray(initial_mask.astype(np.uint8) * 255).save(folder / 'prompt_refined.png')
                        prompt_info['refined_mask_sha256'] = digest(folder / 'prompt_refined.png')
                        if initial_mask.any():
                            predictor.add_new_mask(state, frame_idx=first, obj_id=1, mask=initial_mask)
                            prompt_info['lean_mask_status'] = 'automatic_mask'
                        else:
                            prompt_info['lean_mask_status'] = 'empty_prediction_box_only'
                        count = 0
                        for index, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=False):
                            if index != first + count or object_ids != [1]:
                                raise RuntimeError('Nonchronological propagation or unexpected object')
                            mask = logits[0, 0].detach().cpu().numpy() > 0
                            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f'{index:06d}.png')
                            count += 1
                        if count != len(frames) - first:
                            raise RuntimeError('Incomplete propagation')
                finally:
                    predictor.reset_state(state)
            assert_parameter_integrity(predictor, before, folder, 'after')
            del predictor
        for i in range(first if first is not None else len(frames)):
            with Image.open(frames[i].image_path) as image:
                w, h = image.size
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(folder / f'{i:06d}.png')
        (folder / 'prompt.json').write_text(json.dumps(prompt_info, indent=2) + '\n')
        (folder / 'memory_events.json').write_text(json.dumps(events, indent=2) + '\n')
        (folder / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
        rows = []
        for i, frame in enumerate(frames):
            path = folder / f'{i:06d}.png'
            pred = np.asarray(Image.open(path)) > 0
            truth = load_label_mask(frame.mask_path, [1]) == 1
            if pred.shape != truth.shape:
                raise ValueError(f'Shape mismatch: {name}/{i}')
            score = calculate_scores(pred, truth)
            rows.append({'index': i, 'source_frame_index': frame.frame_index, 'present': bool(truth.any()),
                         'dice': score['dice'], 'iou': score['iou'], 'false_alarm': bool(pred.any()) if not truth.any() else None,
                         'image_sha256': digest(frame.image_path), 'label_sha256': digest(frame.mask_path), 'mask_sha256': digest(path)})
        (folder / 'frames.json').write_text(json.dumps(rows, indent=2) + '\n')
        records.append({'sequence': name, 'frames': len(rows), 'first_detection_index': first,
            'point_status': prompt_info['point_status'], 'lean_mask_status': prompt_info['lean_mask_status'],
            'prompt_sha256': digest(folder / 'prompt.json'),
            'frames_sha256': digest(folder / 'frames.json'), 'events_sha256': digest(folder / 'memory_events.json'),
            'selections_sha256': digest(folder / 'selections.json'), 'memory_events': len(events), 'selection_events': len(selections)})
        (output / 'progress.json').write_text(json.dumps({'completed': [r['sequence'] for r in records], 'records': records}, indent=2) + '\n')
        print(f'{name}: {len(rows)} frames, first YOLO={first}, points={prompt_info["point_status"]}, memory events={len(events)}', flush=True)
    if args.mode == 'gate' and (not any(r['memory_events'] for r in records) or not any(r['point_status'] == 'available' for r in records)):
        raise ValueError('Automatic gate did not exercise both point and memory modifications')
    (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(output / 'records.json')}, indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Matched-frame detector-box and feature-point pilot; no video propagation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from evaluation.segmentation import calculate_scores  # noqa: E402
from modeling.prompt_feature_refinement import reference_prototypes, select_points  # noqa: E402
from scripts.memory_token_io import digest, prompt_index, validate_output  # noqa: E402

PROTOCOL = ROOT / 'protocols/prompt_feature_refinement_v1.json'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'
REPO = ROOT / 'external/sam2_diagnostic_reference'
REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
ARMS = ('yolo_box', 'box_random_points', 'box_feature_points', 'oracle_gt_box')


def gt_box(mask):
    y, x = np.nonzero(mask)
    return None if not len(x) else [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1 - x0) * max(0., y1 - y0)
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return float(intersection / union) if union else 0.0


def choose_cases(frames):
    prompt = prompt_index(frames, {'initialization': 'first_labeled_frame'})
    present, empty = [], []
    for i in range(prompt + 1, len(frames)):
        gt = load_label_mask(frames[i].mask_path, [1]) == 1
        (present if gt.any() else empty).append(i)
    indices = sorted(set(([present[0], present[len(present)//2], present[-1]] if present else []) +
                         ([empty[0]] if empty else [])))
    return prompt, indices


def embedding(predictor):
    z = predictor.get_image_embedding()[0].detach().float()
    if predictor.model.directly_add_no_mem_embed:
        z = z - predictor.model.no_mem_embed.detach().float().reshape(-1, 1, 1)
    return z


def predict_mask(predictor, shape, box, points=None):
    if box is None:
        return np.zeros(shape, dtype=bool)
    kwargs = {'box': np.asarray(box, dtype=np.float32), 'multimask_output': False}
    if points is not None:
        kwargs.update(point_coords=np.asarray(points, dtype=np.float32),
                      point_labels=np.array([1, 0], dtype=np.int32))
    masks, _, _ = predictor.predict(**kwargs)
    mask = np.asarray(masks[0], dtype=bool)
    if mask.shape != shape:
        raise RuntimeError('SAM2 mask dimensions changed')
    return mask


def detect_box(detector, image):
    h, w = image.shape[:2]
    result = detector.predict(source=np.ascontiguousarray(image[:, :, ::-1]),
                              conf=.25, iou=.7, imgsz=640, classes=[0], device='cpu',
                              save=False, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return None, None
    selected = int(torch.argmax(boxes.conf).item())
    raw = boxes.xyxy[selected].cpu().numpy().astype(float)
    raw[[0, 2]] = np.clip(raw[[0, 2]], 0, w)
    raw[[1, 3]] = np.clip(raw[[1, 3]], 0, h)
    if raw[0] >= raw[2] or raw[1] >= raw[3]:
        return None, None
    return raw.tolist(), float(boxes.conf[selected].item())


def automatic_reference(predictor, detector, image):
    """Construct a candidate reference using only the image and model outputs."""
    predictor.set_image(image)
    box, confidence = detect_box(detector, image)
    mask = predict_mask(predictor, image.shape[:2], box)
    positive = negative = None
    status = 'no_detection' if box is None else 'empty_mask'
    if mask.any():
        try:
            positive, negative = reference_prototypes(embedding(predictor), mask)
            status = 'available'
        except ValueError:
            status = 'degenerate_mask_or_features'
    return box, confidence, mask, positive, negative, status


def gate_records(path):
    meta = json.loads((path / 'setup.json').read_text())
    done = json.loads((path / 'completion.json').read_text())
    records = json.loads((path / 'records.json').read_text())
    if meta['mode'] != 'gate' or not done['passed'] or done['records_sha256'] != digest(path / 'records.json'):
        raise ValueError('Gate is incomplete or changed')
    return meta['signature'], records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('gate', 'pilot'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--detector', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if actual != REVISION or dirty:
        raise RuntimeError('SAM2 reference checkout differs from its pin')
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(output)
    os.environ['WANDB_MODE'] = 'disabled'
    import ultralytics
    from ultralytics import YOLO
    sys.path.insert(0, str(REPO))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    signature = {'sam2_revision': actual, 'sam2_checkpoint_sha256': digest(args.checkpoint),
                 'detector_sha256': digest(args.detector), 'manifest_sha256': digest(MANIFEST),
                 'protocol_sha256': digest(PROTOCOL), 'runner_sha256': digest(Path(__file__)),
                 'feature_code_sha256': digest(ROOT / 'modeling/prompt_feature_refinement.py'),
                 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                 'runtime': 'CPU FP32, no autocast, 2 torch threads',
                 'sam2_config': protocol['sam2']['config']}
    gates = None
    if args.mode == 'pilot':
        if args.gate_a is None or args.gate_b is None:
            p.error('pilot requires --gate-a and --gate-b')
        siga, first = gate_records(args.gate_a)
        sigb, second = gate_records(args.gate_b)
        if siga != signature or sigb != signature or first != second:
            raise ValueError('Independent gates differ or source/runtime changed')
        gates = {(r['sequence'], r['source_frame_index']): r for r in first}
    (output / 'setup.json').write_text(json.dumps({'mode': args.mode, 'signature': signature,
        'protocol': protocol, 'warning': 'Detector training split/overlap unknown; exploratory.'}, indent=2) + '\n')
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    names = protocol['gate_sequences'] if args.mode == 'gate' else [f'seq{i}' for i in range(2, 24) if i != 7]
    plan = []
    for name in names:
        prompt, indices = choose_cases(data[name])
        for i in indices:
            plan.append({'sequence': name, 'index': i, 'source_frame_index': data[name][i].frame_index,
                         'prompt_index': prompt})
    (output / 'cases.json').write_text(json.dumps(plan, indent=2) + '\n')
    detector = YOLO(str(args.detector))
    if detector.names != {0: 'polyp'}:
        raise ValueError(f'Unexpected detector classes: {detector.names}')
    model = build_sam2(protocol['sam2']['config'], str(args.checkpoint), device='cpu',
                       mode='eval', apply_postprocessing=False)
    model.eval().requires_grad_(False)
    predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    records = []
    for name in names:
        frames = data[name]
        # Search only past frames, never target GT, for the first usable automatic reference.
        scan_index = 0
        positive = negative = None
        reference_index = reference_box = reference_confidence = None
        reference_image_sha = reference_prediction_sha = None
        folder = output / name
        folder.mkdir(exist_ok=True)
        reference_path = folder / 'reference_prediction.png'
        for case in (x for x in plan if x['sequence'] == name):
            i = case['index']
            while scan_index < i and positive is None:
                reference_image = np.array(Image.open(frames[scan_index].image_path).convert('RGB'), copy=True)
                ref_box, ref_conf, ref_mask, ref_pos, ref_neg, ref_status = automatic_reference(predictor, detector, reference_image)
                if ref_status == 'available':
                    reference_index, reference_box, reference_confidence = scan_index, ref_box, ref_conf
                    positive, negative = ref_pos, ref_neg
                    Image.fromarray(ref_mask.astype(np.uint8) * 255).save(reference_path)
                    reference_prediction_sha = digest(reference_path)
                    reference_image_sha = digest(frames[scan_index].image_path)
                scan_index += 1
            image = np.array(Image.open(frames[i].image_path).convert('RGB'), copy=True)
            h, w = image.shape[:2]
            predictor.set_image(image)
            box, confidence = detect_box(detector, image)
            seed = int.from_bytes(hashlib.sha256(f'20261114|{name}|{case["source_frame_index"]}'.encode()).digest()[:8], 'little')
            points = select_points(embedding(predictor), positive, negative, box, (h, w), seed) if box is not None and positive is not None else None
            generated = {
                'yolo_box': predict_mask(predictor, (h, w), box),
                'box_random_points': predict_mask(predictor, (h, w), box, points['random_points'] if points else None),
                'box_feature_points': predict_mask(predictor, (h, w), box, points['feature_points'] if points else None),
            }
            # Target GT is loaded only after all detector-based predictions are produced.
            truth = load_label_mask(frames[i].mask_path, [1]) == 1
            oracle = gt_box(truth)
            if oracle is not None:
                generated['oracle_gt_box'] = predict_mask(predictor, (h, w), oracle)
            row = {**case, 'present': bool(truth.any()), 'image_sha256': digest(frames[i].image_path),
                   'target_mask_sha256': digest(frames[i].mask_path),
                   'reference_index': reference_index,
                   'reference_frame': frames[reference_index].frame_index if reference_index is not None else None,
                   'reference_image_sha256': reference_image_sha,
                   'reference_box': reference_box, 'reference_confidence': reference_confidence,
                   'reference_status': 'available' if positive is not None else 'unavailable_prior',
                   'reference_prediction_sha256': reference_prediction_sha,
                   'box': box, 'confidence': confidence, 'oracle_box': oracle,
                   'detector_box_iou': box_iou(box, oracle) if oracle else None,
                   'point_selection': points, 'point_fallback': bool(box is not None and points is None),
                   'conditions': {}}
            folder = output / name
            folder.mkdir(exist_ok=True)
            for arm, prediction in generated.items():
                score = calculate_scores(prediction, truth)
                path = folder / f'{i:06d}_{arm}.png'
                Image.fromarray(prediction.astype(np.uint8) * 255).save(path)
                row['conditions'][arm] = {'dice': score['dice'], 'iou': score['iou'],
                    'false_alarm': bool(prediction.any()) if not truth.any() else None,
                    'mask_sha256': digest(path)}
            if gates is not None and name in protocol['gate_sequences']:
                if gates.get((name, case['source_frame_index'])) != row:
                    raise RuntimeError(f'Pilot did not replay the gate exactly: {name}/{i}')
            records.append(row)
            print(f'{name}: target {i}, detection={box is not None}', flush=True)
        predictor.reset_predictor()
    if gates is not None and len(gates) != sum(r['sequence'] in protocol['gate_sequences'] for r in records):
        raise RuntimeError('Gate coverage differs')
    (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    results = {'sequences': len(names), 'targets': len(records),
               'present_targets': sum(r['present'] for r in records),
               'empty_targets': sum(not r['present'] for r in records),
               'detector_no_box': sum(r['box'] is None for r in records),
               'point_fallbacks': sum(r['point_fallback'] for r in records),
               'reference_available_sequences': len({r['sequence'] for r in records if r['reference_status'] == 'available'}),
               'detector_recall_iou_0_5': float(np.mean([r['detector_box_iou'] >= .5 for r in records if r['present']])),
               'mean_detector_box_iou_present': float(np.mean([r['detector_box_iou'] for r in records if r['present']])),
               'empty_frame_detections': sum(r['box'] is not None for r in records if not r['present']),
               'arms': {}}
    for arm in ARMS:
        by_seq = {name: [r['conditions'][arm]['dice'] for r in records if r['sequence'] == name and r['present'] and arm in r['conditions']] for name in names}
        means = {k: float(np.mean(v)) for k, v in by_seq.items() if v}
        results['arms'][arm] = {'mean_present_dice_equal_sequence': float(np.mean(list(means.values()))),
            'per_sequence_dice': means, 'empty_false_alarms': sum(r['conditions'][arm]['false_alarm'] is True for r in records if not r['present'] and arm in r['conditions']) if arm != 'oracle_gt_box' else None}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(output / 'records.json'),
        'results_sha256': digest(output / 'results.json'), 'cases': len(records),
        'interpretation': 'matched-frame exploratory pilot, not propagation or independent validation'}, indent=2) + '\n')


if __name__ == '__main__':
    main()

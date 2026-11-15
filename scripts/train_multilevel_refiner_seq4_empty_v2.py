#!/usr/bin/env python3
"""17-train/10-validation development pilot, not a propagation experiment.

Train only on saved SAM2 predictions/features from seq2/3/4. Freeze the resulting
weights before evaluating previously inspected seq9/10 frames. Ground truth is
used solely as a training target and for evaluation, never as a model input.
"""
import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask, save_binary_mask
from modeling.lean_channel_attention import author_sha256
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.memory_token_io import digest
from scripts.preflight_refiner_seq4_empty_v2 import validate_samples

DRIVE_OUTPUT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs').resolve()
TRAIN_SEQUENCES = {'seq2', 'seq3', 'seq4'}
VALIDATION_SEQUENCES = {'seq9', 'seq10'}


def check_manifest(folder):
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    manifest_sha = digest(manifest_path)
    if not manifest.get('passed') or not completion.get('passed') or completion.get('manifest_sha256') != manifest_sha:
        raise ValueError('Pilot manifest or completion receipt differs')
    if manifest.get('source_sha256') != digest(ROOT / 'scripts/preflight_refiner_seq4_empty_v2.py'):
        raise ValueError('Pilot preflight source differs')
    samples = manifest['samples']
    validate_samples(samples)
    if (digest(manifest['original_manifest_path']) != manifest['original_manifest_sha256']
            or digest(manifest['extra_receipt_path']) != manifest['extra_receipt_sha256']):
        raise ValueError('Original manifest or extra-capture receipt changed')
    return manifest, manifest_sha


def read_capture(item):
    for key in ('feature', 'snapshot'):
        if digest(item[key]) != item[key + '_sha256']:
            raise ValueError(f'Changed {key} capture')
    feature = torch.load(item['feature'], map_location='cpu', weights_only=True)
    snapshot = torch.load(item['snapshot'], map_location='cpu', weights_only=True)
    early = feature['feature'].float().unsqueeze(0)
    deep = snapshot['decoder_inputs']['backbone_features'].float()
    logits = snapshot['decoder_outputs'][3].float()
    high = snapshot['decoder_outputs'][4].float()
    if feature['frame'] != item['frame'] or snapshot['frame'] != item['frame'] or early.shape != (1, 32, 256, 256) or deep.shape != (1, 256, 64, 64) or logits.shape != (1, 1, 256, 256) or high.shape != (1, 1, 1024, 1024):
        raise ValueError('Capture frame or feature geometry differs')
    if not all(torch.isfinite(x).all() for x in (early, deep, logits, high)):
        raise ValueError('Nonfinite capture')
    if not torch.allclose(F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False), high, atol=1e-5, rtol=1e-6):
        raise ValueError('Captured decoder outputs do not match the native resize')
    shape = (int(snapshot['video_height']), int(snapshot['video_width']))
    if min(shape) < 1:
        raise ValueError('Invalid original video shape')
    return early, deep, logits, shape


def to_video(logits, shape):
    # _run_single_frame_inference returns low-resolution logits; the video
    # predictor resizes those directly, not through the memory-encoder mask.
    return F.interpolate(logits, size=shape, mode='bilinear', align_corners=False) if logits.shape[-2:] != shape else logits


def load_target(item, shape):
    if digest(item['label']) != item['label_sha256']:
        raise ValueError('Label changed')
    mask = load_label_mask(Path(item['label']), [1])
    if mask.shape != shape:
        raise ValueError('Label and original video geometry differ')
    return torch.from_numpy((mask == 1).astype(np.float32))[None, None]


def loss_fn(prediction, target):
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prob = prediction.sigmoid()
    dice_loss = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice_loss


def input_features(early, deep, arm):
    if arm == 'mask_only':
        return torch.zeros_like(early), torch.zeros_like(deep)
    if arm != 'multilevel':
        raise ValueError('Unknown arm')
    return early, deep


def verify_parity(path, manifest_sha):
    parity = json.loads(path.read_text())
    if (not parity.get('passed') or parity.get('manifest_sha256') != manifest_sha or
            parity.get('runner_sha256') != digest(Path(__file__)) or
            parity.get('verifier_sha256') != digest(ROOT / 'scripts/verify_refiner_seq4_empty_mps_v2.py') or
            parity.get('model_source_sha256') != digest(ROOT / 'modeling/multilevel_contrast_refiner.py') or
            parity.get('author_gate_sha256') != author_sha256() or
            parity.get('platform') != platform.platform() or
            parity.get('machine') != platform.machine() or
            parity.get('torch_version') != str(torch.__version__)):
        raise ValueError('MPS parity receipt does not match this data, runner or environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-preflight', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('multilevel', 'mask_only'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    parser.add_argument('--parity-receipt', type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        raise ValueError('Epochs must be between 1 and 100')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; do not silently fall back to CPU')
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('MPS fallback must be disabled for this verification pilot')
    device = torch.device(args.device)
    out = args.output.resolve()
    if not out.is_relative_to(DRIVE_OUTPUT) or out == DRIVE_OUTPUT:
        raise ValueError('Pilot output must be a new folder under the specified Google Drive outputs')
    manifest, manifest_sha = check_manifest(args.data_preflight)
    if args.device == 'mps':
        if args.parity_receipt is None:
            raise ValueError('MPS training requires a completed CPU/MPS parity receipt')
        verify_parity(args.parity_receipt, manifest_sha)
    train = sorted((x for x in manifest['samples'] if x['split'] == 'train'), key=lambda x: (x['sequence'], x['frame']))
    val = sorted((x for x in manifest['samples'] if x['split'] == 'validation'), key=lambda x: (x['sequence'], x['frame']))
    # Verify all captures and images before starting; do not inspect validation labels.
    for item in train + val:
        if digest(item['image']) != item['image_sha256']:
            raise ValueError('Input image changed')
        read_capture(item)
    for item in train:
        if digest(item['label']) != item['label_sha256']:
            raise ValueError('Training label changed')
    out.mkdir(parents=True, exist_ok=False)
    seed = 20261114
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'mps':
        torch.mps.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = MultilevelContrastRefiner().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    steps = 0
    history = []
    for epoch in range(args.epochs):
        epoch_loss = []
        order = torch.randperm(len(train)).tolist()
        for index in order:
            item = train[index]
            early, deep, logits, shape = read_capture(item)
            target = load_target(item, shape)
            early, deep = input_features(early, deep, args.arm)
            early, deep, logits, target = (x.to(device) for x in (early, deep, logits, target))
            optimizer.zero_grad(set_to_none=True)
            prediction = to_video(model(early, deep, logits), shape)
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite training gradient')
            optimizer.step(); steps += 1
            epoch_loss.append(loss.item())
        history.append({'epoch': epoch + 1, 'mean_training_loss': float(np.mean(epoch_loss))})
        (out / 'progress.json').write_text(json.dumps({'epoch': epoch + 1, 'steps': steps, 'history': history}, indent=2) + '\n')
        print(f'epoch {epoch + 1}/{args.epochs}: loss={history[-1]["mean_training_loss"]:.6f}', flush=True)
    payload = {'format': 'multilevel_contrast_refiner_v1', 'research_only': True, 'training_steps': steps,
               'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'backbone_sha256': manifest['checkpoint_sha256'], 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
               'train_manifest_sha256': manifest_sha, 'author_sha256': author_sha256(),
               'model_source_sha256': digest(ROOT / 'modeling/multilevel_contrast_refiner.py'),
               'runner_source_sha256': digest(Path(__file__)), 'arm': args.arm, 'seed': seed,
               'training_sequences': sorted(TRAIN_SEQUENCES), 'validation_sequences': sorted(VALIDATION_SEQUENCES),
               'epochs': args.epochs, 'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
               'device': args.device, 'torch_version': str(torch.__version__),
               'parity_receipt_sha256': digest(args.parity_receipt) if args.parity_receipt else None,
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'loss': 'full-video-resolution BCE-with-logits plus soft Dice loss',
               'scope': 'Exploratory 17/10 fixed-read development pilot; seq4 empty frames selected by label, validation previously inspected, patient independence unknown; not full-video inference.'}
    checkpoint = out / 'checkpoint.pt'
    torch.save(payload, checkpoint)
    # Re-load through the inference provenance gate, freeze weights, and persist
    # every validation prediction before opening any validation label.
    evaluated = MultilevelContrastRefiner()
    evaluated.load_trained_state(torch.load(checkpoint, map_location='cpu', weights_only=True), backbone_sha256=manifest['checkpoint_sha256'], feature_pipeline_sha256=manifest['feature_pipeline_sha256'], train_manifest_sha256=manifest_sha)
    evaluated.to(device).eval()
    pred_dir = out / 'validation_predictions'; pred_dir.mkdir()
    pred_rows = []
    with torch.no_grad():
        for item in val:
            early, deep, logits, shape = read_capture(item)
            early, deep = input_features(early, deep, args.arm)
            early, deep, logits = (x.to(device) for x in (early, deep, logits))
            prediction = (to_video(evaluated(early, deep, logits), shape) > 0).cpu()
            baseline = (to_video(logits, shape) > 0).cpu()
            stem = f"{item['sequence']}_{item['frame']:06d}"
            pred_path = pred_dir / f'{stem}_refined.png'; base_path = pred_dir / f'{stem}_baseline.png'
            save_binary_mask(prediction[0, 0].numpy(), pred_path)
            save_binary_mask(baseline[0, 0].numpy(), base_path)
            pred_rows.append({'sequence': item['sequence'], 'frame': item['frame'], 'shape': list(shape), 'refined': str(pred_path), 'baseline': str(base_path), 'refined_sha256': digest(pred_path), 'baseline_sha256': digest(base_path)})
    (out / 'frozen_predictions.json').write_text(json.dumps({'checkpoint_sha256': digest(checkpoint), 'rows': pred_rows}, indent=2) + '\n')
    from PIL import Image
    rows = []
    for item, row in zip(val, pred_rows):
        target = load_target(item, tuple(row['shape']))[0, 0].numpy().astype(bool)
        metrics = {}
        for arm in ('baseline', 'refined'):
            pred = np.asarray(Image.open(row[arm]).convert('L')) > 0
            intersection = int((pred & target).sum()); denominator = int(pred.sum() + target.sum())
            metrics[arm] = {'dice': (2 * intersection / denominator) if denominator else 1.0, 'predicted_pixels': int(pred.sum())}
        rows.append({'sequence': item['sequence'], 'frame': item['frame'], 'labeled_present': bool(target.any()), **metrics})
    present = [x for x in rows if x['labeled_present']]
    empty = [x for x in rows if not x['labeled_present']]
    summary = {arm: {'mean_present_dice': float(np.mean([x[arm]['dice'] for x in present])) if present else None,
                     'empty_frame_false_alarms': sum(x[arm]['predicted_pixels'] > 0 for x in empty), 'empty_frames': len(empty)} for arm in ('baseline', 'refined')}
    receipt = {'passed': True, 'scope': payload['scope'], 'manifest_sha256': manifest_sha, 'checkpoint_sha256': digest(checkpoint), 'frozen_predictions_sha256': digest(out / 'frozen_predictions.json'), 'history': history, 'rows': rows, 'summary': summary}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

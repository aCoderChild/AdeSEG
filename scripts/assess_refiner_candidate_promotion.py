#!/usr/bin/env python3
"""Independently rescore frozen fixed-read candidate and control predictions.

This conservative development-screening rule does not authorize video
inference, create a locked test, or establish statistical significance. The
existing labeled validation frames are already inspected. Frozen predictions
are regenerated on their original device and checked before labels are read.
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
from modeling.lean_channel_attention import author_sha256
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import (
    check_manifest, input_features, load_target, read_capture, to_video,
)

SHARED_KEYS = ('backbone_sha256', 'feature_pipeline_sha256', 'train_manifest_sha256',
               'author_sha256', 'model_source_sha256', 'runner_source_sha256', 'seed',
               'training_sequences', 'validation_sequences', 'training_steps', 'epochs',
               'optimizer', 'learning_rate', 'weight_decay', 'loss', 'device', 'torch_version')


def screen(metrics):
    """Conservative development rule; compares video means and empty burden.

    Missing present or empty frames is insufficient evidence, not a pass.
    """
    present = [x for x in metrics if x['labeled_present']]
    empty = [x for x in metrics if not x['labeled_present']]
    if not present or not empty:
        return {'eligible': False, 'reason': 'Validation lacks present or labeled-empty frames'}
    videos = sorted({x['sequence'] for x in present})
    means = {k: float(np.mean([np.mean([x[k]['dice'] for x in present if x['sequence'] == video])
                               for video in videos]))
             for k in ('baseline', 'candidate', 'control')}
    false_alarms = {k: sum(x[k]['predicted_pixels'] > 0 for x in empty)
                    for k in ('baseline', 'candidate', 'control')}
    empty_pixels = {k: sum(x[k]['predicted_pixels'] for x in empty)
                    for k in ('baseline', 'candidate', 'control')}
    empty_videos = sorted({x['sequence'] for x in empty})
    per_video_empty = {}
    for video in empty_videos:
        frames = [x for x in empty if x['sequence'] == video]
        per_video_empty[video] = {
            name: {'false_alarm_rate': sum(x[name]['predicted_pixels'] > 0 for x in frames) / len(frames),
                   'mean_predicted_pixels': float(np.mean([x[name]['predicted_pixels'] for x in frames]))}
            for name in ('baseline', 'candidate', 'control')}
    checks = {'beats_baseline_present_dice': means['candidate'] > means['baseline'],
              'beats_control_present_dice': means['candidate'] > means['control'],
              'no_worse_empty_false_alarm_rate_per_video': all(
                  x['candidate']['false_alarm_rate'] <= x['baseline']['false_alarm_rate']
                  for x in per_video_empty.values()),
              'no_more_empty_predicted_pixels_per_video': all(
                  x['candidate']['mean_predicted_pixels'] <= x['baseline']['mean_predicted_pixels']
                  for x in per_video_empty.values())}
    return {'eligible': all(checks.values()), 'checks': checks, 'present_frames': len(present),
            'present_videos': len(videos), 'empty_frames': len(empty), 'mean_per_video_present_dice': means,
            'empty_false_alarms': false_alarms, 'empty_predicted_pixels': empty_pixels,
            'per_video_empty': per_video_empty}


def read_run(folder, manifest_sha, runner_source, expected_arm):
    folder = folder.resolve()
    completion_path = folder / 'completion.json'
    frozen_path = folder / 'frozen_predictions.json'
    checkpoint_path = folder / 'checkpoint.pt'
    completion = json.loads(completion_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if (not completion.get('passed') or completion.get('manifest_sha256') != manifest_sha or
            completion.get('checkpoint_sha256') != digest(checkpoint_path) or
            completion.get('frozen_predictions_sha256') != digest(frozen_path) or
            frozen.get('checkpoint_sha256') != digest(checkpoint_path)):
        raise ValueError('Run, frozen predictions or checkpoint changed')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (payload.get('arm') != expected_arm or payload.get('train_manifest_sha256') != manifest_sha or
            payload.get('runner_source_sha256') != digest(runner_source) or
            payload.get('model_source_sha256') != digest(ROOT / 'modeling/multilevel_contrast_refiner.py') or
            payload.get('author_sha256') != author_sha256()):
        raise ValueError('Checkpoint arm, manifest or source differs')
    for row in frozen['rows']:
        for name in ('refined', 'baseline'):
            path = Path(row[name]).resolve()
            if not path.is_relative_to(folder) or digest(path) != row[name + '_sha256']:
                raise ValueError('Frozen prediction path or content differs')
    return {'folder': folder, 'completion_path': completion_path, 'completion': completion,
            'frozen_path': frozen_path, 'frozen': frozen, 'checkpoint_path': checkpoint_path,
            'payload': payload}


def verify(args):
    manifest, manifest_sha = check_manifest(args.data_preflight)
    candidate = read_run(args.candidate_run, manifest_sha, args.runner_source, 'multilevel')
    control = read_run(args.control_run, manifest_sha, args.runner_source, 'mask_only')
    c, b = candidate['payload'], control['payload']
    if any(c.get(key) != b.get(key) for key in SHARED_KEYS):
        raise ValueError('Candidate and control training conditions differ')
    if ({k: tuple(v.shape) for k,v in c['state_dict'].items()} !=
            {k: tuple(v.shape) for k,v in b['state_dict'].items()}):
        raise ValueError('Candidate and control parameter capacities differ')
    device = c['device']
    if device != 'mps' or not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise ValueError('Original MPS inference device required with fallback disabled')
    models = {}
    for name, run in (('candidate', candidate), ('control', control)):
        model = MultilevelContrastRefiner()
        model.load_trained_state(run['payload'], backbone_sha256=manifest['checkpoint_sha256'],
                                 feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
                                 train_manifest_sha256=manifest_sha)
        models[name] = model.to(device).eval()
    samples = sorted((x for x in manifest['samples'] if x['split'] == 'validation'),
                     key=lambda x: (x['sequence'], x['frame']))
    maps = []
    for run in (candidate, control):
        rows = run['frozen']['rows']
        mapping = {(x['sequence'], x['frame']): x for x in rows}
        if len(mapping) != len(rows) or set(mapping) != {(x['sequence'], x['frame']) for x in samples}:
            raise ValueError('Frozen validation frame coverage differs')
        maps.append(mapping)

    # First reproduce every frozen prediction, then open any validation label.
    frozen_masks = []
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        for item in samples:
            key = (item['sequence'], item['frame'])
            early, deep, logits, shape = read_capture(item)
            tensors = [early.to(device), deep.to(device), logits.to(device)]
            baseline = (to_video(tensors[2], shape) > 0).cpu().numpy()[0, 0]
            masks = {'baseline': baseline}
            for name, run, mapping in (('candidate', candidate, maps[0]), ('control', control, maps[1])):
                arm = run['payload']['arm']
                e, d = input_features(tensors[0], tensors[1], arm)
                actual = (to_video(models[name](e, d, tensors[2]), shape) > 0).cpu().numpy()[0, 0]
                row = mapping[key]
                for label, expected in (('baseline', baseline), ('refined', actual)):
                    saved = np.asarray(Image.open(row[label]).convert('L')) > 0
                    if saved.shape != shape or not np.array_equal(saved, expected):
                        raise ValueError(f'{name} frozen {label} prediction does not reproduce')
                masks[name] = actual
            frozen_masks.append((item, shape, masks))
    metrics = []
    for item, shape, masks in frozen_masks:
        truth = load_target(item, shape).numpy()[0, 0].astype(bool)
        row = {'sequence': item['sequence'], 'frame': item['frame'], 'labeled_present': bool(truth.any())}
        for name, mask in masks.items():
            denom = int(mask.sum() + truth.sum())
            row[name] = {'dice': 2 * int((mask & truth).sum()) / denom if denom else 1.0,
                         'predicted_pixels': int(mask.sum())}
        metrics.append(row)
    decision = screen(metrics)
    return {'audit_passed': True, 'decision': decision, 'rows': metrics,
            'candidate_checkpoint_sha256': digest(candidate['checkpoint_path']),
            'control_checkpoint_sha256': digest(control['checkpoint_path']),
            'candidate_completion_sha256': digest(candidate['completion_path']),
            'control_completion_sha256': digest(control['completion_path']),
            'candidate_frozen_sha256': digest(candidate['frozen_path']),
            'control_frozen_sha256': digest(control['frozen_path']),
            'manifest_sha256': manifest_sha, 'runner_source_sha256': digest(args.runner_source),
            'auditor_source_sha256': digest(Path(__file__)),
            'scope': 'Development screen specified after inspection of earlier pilot outcomes; not preregistered, not a live-runtime authorization or independent final test.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-preflight', 'candidate-run', 'control-run', 'runner-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    out = validate_output(args.output)
    if out.exists():
        raise FileExistsError('Output must be new')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['decision'], indent=2))


if __name__ == '__main__':
    main()

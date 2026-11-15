#!/usr/bin/env python3
"""Independently replay and rescore the existing fixed-read pilot.

This read-only audit checks saved artifacts, not a real-video runtime. It
reproduces masks with the historical captured features and compares the stored
metrics with independently recomputed pixel counts. Outputs are printed only.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.memory_token_io import digest
from scripts.train_multilevel_refiner_fixed_read_pilot import (
    check_manifest, input_features, load_target, read_capture, to_video,
)


def verify(base: Path):
    manifest, manifest_hash = check_manifest(base / 'pilot_data_preflight_attempt01')
    samples = sorted((x for x in manifest['samples'] if x['split'] == 'validation'),
                     key=lambda x: (x['sequence'], x['frame']))
    torch.set_num_threads(2)
    outputs = {}
    for arm in ('multilevel', 'mask_only'):
        folder = base / f'fixed_read_{arm}_attempt01'
        receipt = json.loads((folder / 'completion.json').read_text())
        frozen = json.loads((folder / 'frozen_predictions.json').read_text())
        checkpoint = folder / 'checkpoint.pt'
        if (not receipt['passed'] or receipt['manifest_sha256'] != manifest_hash or
                digest(checkpoint) != receipt['checkpoint_sha256'] or
                frozen['checkpoint_sha256'] != receipt['checkpoint_sha256'] or
                digest(folder / 'frozen_predictions.json') != receipt['frozen_predictions_sha256']):
            raise ValueError('Receipt, checkpoint or frozen predictions differ')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (payload['arm'] != arm or
                payload['model_source_sha256'] != digest(ROOT / 'modeling/multilevel_contrast_refiner.py') or
                payload['runner_source_sha256'] != digest(ROOT / 'scripts/train_multilevel_refiner_fixed_read_pilot.py')):
            raise ValueError('Checkpoint implementation or arm differs')
        model = MultilevelContrastRefiner()
        model.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
                                 feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
                                 train_manifest_sha256=manifest_hash)
        model.eval()
        recomputed = []
        for item, row, recorded in zip(samples, frozen['rows'], receipt['rows'], strict=True):
            key = (item['sequence'], item['frame'])
            if key != (row['sequence'], row['frame']) or key != (recorded['sequence'], recorded['frame']):
                raise ValueError('Validation ordering differs')
            early, deep, logits, shape = read_capture(item)
            early, deep = input_features(early, deep, arm)
            with torch.no_grad():
                predictions = {'baseline': (to_video(logits, shape) > 0)[0, 0].numpy(),
                               'refined': (to_video(model(early, deep, logits), shape) > 0)[0, 0].numpy()}
            target = load_target(item, shape)[0, 0].numpy().astype(bool)
            if bool(target.any()) != recorded['labeled_present']:
                raise ValueError('Recorded foreground presence differs')
            result = {'labeled_present': bool(target.any())}
            for name, prediction in predictions.items():
                if digest(row[name]) != row[name + '_sha256']:
                    raise ValueError('Frozen prediction hash differs')
                saved = np.asarray(Image.open(row[name]).convert('L')) > 0
                if not np.array_equal(saved, prediction):
                    raise ValueError('Replayed prediction differs')
                n = int(prediction.sum() + target.sum())
                dice = 2 * int((prediction & target).sum()) / n if n else 1.0
                if dice != recorded[name]['dice'] or int(prediction.sum()) != recorded[name]['predicted_pixels']:
                    raise ValueError('Independently recomputed metric differs')
                result[name] = {'dice': dice, 'predicted_pixels': int(prediction.sum())}
            recomputed.append(result)
        summary = {}
        for name in ('baseline', 'refined'):
            present = [x[name]['dice'] for x in recomputed if x['labeled_present']]
            empty = [x[name]['predicted_pixels'] for x in recomputed if not x['labeled_present']]
            summary[name] = {'mean_present_dice': float(np.mean(present)) if present else None,
                             'empty_frame_false_alarms': sum(x > 0 for x in empty),
                             'empty_frames': len(empty)}
        if summary != receipt['summary']:
            raise ValueError('Recorded summary differs')
        outputs[arm] = {'frames_verified': len(recomputed), 'summary': summary}
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.experiment), indent=2))

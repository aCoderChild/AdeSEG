#!/usr/bin/env python3
"""Render chosen audited PolypGen frames and saved prediction/label contours."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402

# These examples were selected retrospectively to illustrate observed intervals.
SELECTED = {'seq8': [0, 20, 55], 'seq11': [2, 50, 100],
            'seq20': [13, 20, 45], 'seq23': [0, 8, 35]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--interval-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    done = json.loads((a.merged / 'completion.json').read_text())
    interval_done = json.loads((a.interval_audit / 'completion.json').read_text())
    if (not done['passed'] or not interval_done['passed'] or
        digest(a.merged / 'results.json') != done['results_sha256'] or
        digest(a.interval_audit / 'results.json') != interval_done['results_sha256']):
        raise ValueError('Audited reference changed')
    locations = json.loads((a.merged / 'locations.json').read_text())
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, indices in SELECTED.items():
        folder = Path(locations[name]) / name
        rows = json.loads((folder / 'frames.json').read_text())
        fig, axes = plt.subplots(len(indices), 3, figsize=(12, 9), constrained_layout=True)
        details = []
        for axrow, index in zip(axes, indices):
            frame, row = videos[name][index], rows[index]
            mask_file = folder / f'{index:06d}.png'
            if (digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['label_sha256']
                or digest(mask_file) != row['mask_sha256']):
                raise ValueError('Input or prediction digest differs')
            image = np.asarray(Image.open(frame.image_path).convert('RGB'))
            truth = load_label_mask(frame.mask_path, [1]) == 1
            pred = np.asarray(Image.open(mask_file)) > 0
            if image.shape[:2] != truth.shape or truth.shape != pred.shape:
                raise ValueError('Shape mismatch')
            overlay = image.copy()
            # RGB contours: yellow = ground truth, cyan = predicted boundary.
            cv2.drawContours(overlay, cv2.findContours(truth.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 225, 0), 5)
            cv2.drawContours(overlay, cv2.findContours(pred.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 210, 255), 4)
            axrow[0].imshow(image); axrow[0].set_title(f'Frame {index}: image')
            axrow[1].imshow(overlay); axrow[1].set_title('Yellow: label; cyan: prediction')
            axrow[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            title = f"Predicted mask; Dice {row['dice']:.3f}" if row['present'] else 'Predicted mask; labeled empty'
            axrow[2].set_title(title)
            for ax in axrow: ax.axis('off')
            details.append({'index': index, 'present': row['present'], 'dice': row['dice'],
                            'image_sha256': row['image_sha256'], 'label_sha256': row['label_sha256'], 'mask_sha256': row['mask_sha256']})
        fig.suptitle(f'{name}: selected examples from the automatic Lean run', fontsize=14)
        path = out / f'{name}.png'
        fig.savefig(path, dpi=130); plt.close(fig)
        records[name] = {'examples': details, 'figure_sha256': digest(path)}
    (out / 'records.json').write_text(json.dumps({'scope': 'Retrospectively selected illustrative frames, not a representative sample.', 'cases': records}, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(out / 'records.json'),
                                                    'source_sha256': digest(Path(__file__))}, indent=2) + '\n')


if __name__ == '__main__': main()

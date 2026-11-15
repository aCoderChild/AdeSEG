#!/usr/bin/env python3
"""Post-hoc sensitivity and reference-label checks for a completed prompt pilot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402
from scripts.run_prompt_feature_refinement import MANIFEST  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    audit = json.loads((run / 'independent_audit.json').read_text())
    if not audit['passed'] or digest(run / 'records.json') != audit['records_sha256']:
        raise ValueError('An independent audit of the current records is required')
    rows = json.loads((run / 'records.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    base = results['arms']['yolo_box']['per_sequence_dice']
    feature = results['arms']['box_feature_points']['per_sequence_dice']
    delta = {name: feature[name] - score for name, score in base.items()}
    refs = {}
    for row in rows:
        name, index = row['sequence'], row['reference_index']
        if index is None or name in refs:
            continue
        frame = data[name][index]
        truth = load_label_mask(frame.mask_path, [1]) == 1
        prediction = np.asarray(Image.open(run / name / 'reference_prediction.png')) > 0
        denominator = int(truth.sum() + prediction.sum())
        refs[name] = {
            'reference_index': index,
            'reference_source_frame': frame.frame_index,
            'ground_truth_present': bool(truth.any()),
            'reference_mask_dice': 2 * int(np.count_nonzero(truth & prediction)) / denominator if denominator else 1.0,
        }
    empty = [name for name, item in refs.items() if not item['ground_truth_present']]
    diagnostic = {
        'description': 'Post-hoc checks; reference ground truth used only for evaluation, never inference.',
        'records_sha256': audit['records_sha256'],
        'sequence_dice_feature_minus_box': delta,
        'feature_minus_box_excluding_seq10': float(np.mean([v for k, v in delta.items() if k != 'seq10'])),
        'reference_ground_truth_empty_sequences': empty,
        'reference_assessments': refs,
    }
    (run / 'posthoc_diagnostics.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    print(f"Excluded seq10: {diagnostic['feature_minus_box_excluding_seq10']:+.5f}; "
          f'{len(empty)} / {len(refs)} automatic references are labeled empty: {", ".join(empty)}')


if __name__ == '__main__':
    main()

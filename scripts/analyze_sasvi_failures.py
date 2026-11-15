#!/usr/bin/env python3
"""Measure present-frame false positives and misses on two audited sequences."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--cmsa-run', required=True, type=Path)
    args = p.parse_args()
    done = json.loads((args.run / 'completion.json').read_text())
    if not done['passed'] or digest(args.run / 'results.json') != done['results_sha256']:
        raise ValueError('Cohort incomplete or changed')
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    data = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    locations = json.loads((args.run / 'locations.json').read_text()) if (args.run / 'locations.json').exists() else {}
    report = {}
    for name in ('seq11', 'seq20'):
        folder = Path(locations[name]) if locations else args.run / name
        audit = json.loads((folder / 'audit_results.json').read_text())
        rows = json.loads((folder / 'records.json').read_text())
        start = audit['prompt_index_for_evaluation_only'] + 1
        totals = {arm: {'tp': 0, 'fp': 0, 'fn': 0, 'present_nonempty': 0} for arm in ('cmsa', 'sasvi')}
        sources = []
        for i in range(start, len(rows)):
            frame = data[name][i]
            gt = load_label_mask(frame.mask_path, [1]) == 1
            if not gt.any():
                continue
            source = args.cmsa_run / 'predictions' / frame.image_path.resolve().relative_to((ROOT / 'data').resolve()).with_suffix('.png')
            output = folder / 'predictions' / name / f'{i:06d}.png'
            if digest(source) != rows[i]['cmsa_mask_sha256'] or digest(output) != rows[i]['mask_sha256']:
                raise ValueError('Prediction changed')
            masks = {'cmsa': np.asarray(Image.open(source).convert('L')) >= 128,
                     'sasvi': np.asarray(Image.open(output)) > 0}
            for arm, mask in masks.items():
                totals[arm]['tp'] += int(np.count_nonzero(mask & gt))
                totals[arm]['fp'] += int(np.count_nonzero(mask & ~gt))
                totals[arm]['fn'] += int(np.count_nonzero(~mask & gt))
                totals[arm]['present_nonempty'] += bool(mask.any())
            sources.append({'index': i, 'ground_truth_sha256': digest(frame.mask_path)})
        for stats in totals.values():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            stats['precision'] = tp / (tp + fp) if tp + fp else None
            stats['recall'] = tp / (tp + fn) if tp + fn else None
        report[name] = {'present_targets': len(sources), 'totals': totals, 'source_label_hashes': sources}
    (args.run / 'failure_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in val.items() if kk != 'source_label_hashes'} for k, val in report.items()}, indent=2))


if __name__ == '__main__':
    main()

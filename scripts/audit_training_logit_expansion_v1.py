#!/usr/bin/env python3
"""Training-only coverage of continuous automatic-logit expansions.

No labels control the expansions. Labels are opened only afterwards to
measure representativeness; only declared training masks are accessible.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.training_foreground_preservation_v1 import expand_logits
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_boundary_shared_v1 import load_prepared
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target, to_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard = install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl')
    torch.set_num_threads(2)
    manifest, manifest_sha = load_prepared(args.manifest)
    rows = []
    for item in manifest['samples']:
        if item['split'] != 'train':
            continue
        if digest(item['snapshot']) != item['snapshot_sha256']:
            raise ValueError('Snapshot changed')
        snapshot = torch.load(item['snapshot'], map_location='cpu', weights_only=True)
        logits = snapshot['decoder_outputs'][3].float()
        if snapshot['frame'] != item['frame'] or logits.shape != (1, 1, 256, 256) or not torch.isfinite(logits).all():
            raise ValueError('Snapshot logits or frame differ')
        shape = (int(snapshot['video_height']), int(snapshot['video_width']))
        # Compute all candidate inputs before the training label is loaded.
        predictions = {str(w): (to_video(expand_logits(logits, w), shape)>0).numpy()[0, 0]
                       for w in (1, 33, 65)}
        target = load_target(item, shape).numpy()[0, 0] > 0
        gt = int(target.sum()); stats = {}
        for width, pred in predictions.items():
            area = int(pred.sum()); intersection = int((pred & target).sum())
            stats[width] = {'prediction_pixels': area, 'intersection_pixels': intersection,
                            'recall': intersection/gt if gt else None,
                            'precision': intersection/area if area else None}
        rows.append({'sequence': item['sequence'], 'frame': item['frame'], 'target_pixels': gt,
                     'label_sha256': item['label_sha256'], 'snapshot_sha256': item['snapshot_sha256'], 'stats': stats})
    if len(rows) != 164 or guard['blocked_open_attempts']:
        raise ValueError('Training coverage or label guard differs')
    summaries = {}
    for width in ('1', '33', '65'):
        present = [x for x in rows if x['target_pixels']]
        empty = [x for x in rows if not x['target_pixels']]
        high = [x for x in present if x['stats'][width]['recall'] >= .8 and x['stats'][width]['precision'] is not None]
        summaries[width] = {'high_recall_precision_lt_0_2': sum(x['stats'][width]['precision'] < .2 for x in high),
                            'high_recall_precision_lt_0_5': sum(x['stats'][width]['precision'] < .5 for x in high),
                            'high_recall_low_precision_sequences': sorted({x['sequence'] for x in high if x['stats'][width]['precision'] < .2}),
                            'empty_false_alarms': sum(x['stats'][width]['prediction_pixels'] > 0 for x in empty),
                            'zero_overlap': sum(x['stats'][width]['intersection_pixels'] == 0 for x in present)}
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    result = {'passed': True, 'scope': 'Exploratory training-only augmentation coverage; not video or validation evidence.',
              'manifest_sha256': manifest_sha, 'source_sha256': digest(Path(__file__)),
              'expansion_sha256': digest(ROOT/'modeling/training_foreground_preservation_v1.py'),
              'label_guard': dict(guard), 'widths_in_256_logits': [1, 33, 65], 'summaries': summaries, 'rows': rows}
    (out/'coverage.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Check the frozen native comparison with the project's other label decoder."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.score_samurai_automatic_priority import score, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = a.experiment_root
    root = base/'native_sam2_automatic_priority_v1'
    report = root/'matched_three_system_report_attempt01/results.json'
    result = json.loads(report.read_text())
    if result['report_sha256'] != digest(report.parent/'report.md'):
        raise ValueError('Report changed')
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    pre = json.loads((base/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03/manifest.json').read_text())
    evidence = {}
    for seq in ('seq11', 'seq20', 'seq23'):
        frames = videos[seq]
        run = root/f'{seq}_full_mps_attempt01'
        receipt = json.loads((run/'receipt.json').read_text())
        if digest(run/'receipt.json') != result['evidence'][seq]['native_receipt_sha256']:
            raise ValueError('Native receipt changed after primary scoring')
        if len(receipt['records']) != len(frames) or len(result['rows'][seq]) != len(frames):
            raise ValueError('Frame count differs')
        labels = json.loads((Path(pre['sequences'][seq]['source'])/seq/'frames.json').read_text())
        if len(labels) != len(frames):
            raise ValueError('Label metadata coverage differs')
        rows = []
        for i, frame in enumerate(frames):
            prediction = run/f'{i:06d}.png'
            if digest(prediction) != receipt['records'][i]['mask_sha256']:
                raise ValueError('Native prediction changed after primary scoring')
            if digest(frame.mask_path) != labels[i]['label_sha256']:
                raise ValueError('Label changed after primary scoring')
            truth = load_label_mask(frame.mask_path, [1]) == 1
            with Image.open(prediction) as im:
                mask = np.asarray(im.convert('L')) > 0
            metric = score(mask, truth)
            if metric != result['rows'][seq][i]['native']:
                raise ValueError(f'Alternative per-frame score differs: {seq} {i}')
            rows.append({'native': metric, 'post_prompt': i > receipt['prompt_frame']})
        for post in (False, True):
            key = 'post_prompt' if post else 'all'
            if summary(rows, 'native', post_only=post) != result['summaries'][seq]['native'][key]:
                raise ValueError(f'Alternative summary differs: {seq} {key}')
        evidence[seq] = {'frames': len(frames), 'native_receipt_sha256': digest(run/'receipt.json')}
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    record = {'passed': True, 'scope': 'Alternative label decoder and scoring-function audit of frozen native masks; not an independent dataset or inference run.',
              'result_sha256': digest(report), 'auditor_sha256': digest(Path(__file__)), 'sequences': evidence}
    (out/'audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

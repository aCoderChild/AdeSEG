#!/usr/bin/env python3
"""Audit archived training-video predictions to plan development captures.

Uses labels from seq2/3/4 only after the predictions were archived. Findings
are training-data diagnostics, not estimates of held-out performance.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('preflight', 'captures', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.preflight / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads((args.preflight / 'completion.json').read_text())
    if not completion['passed'] or completion['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Split preflight changed')
    if manifest['groups']['train'] != ['seq2', 'seq3', 'seq4']:
        raise ValueError('Unexpected training cohort')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    rows, summaries = [], {}
    receipts = {}
    for seq in manifest['groups']['train']:
        meta = manifest['sequences'][seq]
        source = Path(meta['source']) / seq
        frames_path = source / 'frames.json'
        if digest(frames_path) != meta['frames_metadata_sha256']:
            raise ValueError(f'{seq}: source metadata changed')
        source_rows = json.loads(frames_path.read_text())
        folder = args.captures / ('gate_attempt01' if seq == 'seq3' else f'{seq}_capture_attempt01')
        capture = json.loads((folder / 'utility_capture_receipt.json').read_text())
        done = json.loads((folder / 'completion.json').read_text())
        records_path = folder / 'records.json'
        records = json.loads(records_path.read_text())
        if (not capture['passed'] or capture['sequence'] != seq or not done['passed']
                or done['records_sha256'] != digest(records_path)
                or capture['feature_receipt_sha256'] != digest(folder / 'channel_feature_receipt.json')
                or len(source_rows) != len(videos[seq]) or records['frames'] != len(source_rows)):
            raise ValueError(f'{seq}: capture provenance or coverage differs')
        receipts[seq] = digest(folder / 'utility_capture_receipt.json')
        seq_rows = []
        for i, (frame, source_row) in enumerate(zip(videos[seq], source_rows, strict=True)):
            pred_path = folder / f'{i:06d}.png'
            if digest(pred_path) != records['mask_sha256'][i] or digest(frame.mask_path) != source_row['label_sha256']:
                raise ValueError(f'{seq}/{i}: prediction or label changed')
            prediction = np.asarray(Image.open(pred_path).convert('L')) > 0
            target = load_label_mask(frame.mask_path, [1]) == 1
            if prediction.shape != target.shape:
                raise ValueError(f'{seq}/{i}: mask geometry differs')
            n_pred, n_truth = int(prediction.sum()), int(target.sum())
            intersection = int((prediction & target).sum())
            den = n_pred + n_truth
            row = {'sequence': seq, 'frame': i, 'after_initialization': i > meta['first_prompt'],
                   'present': bool(n_truth), 'predicted_pixels': n_pred, 'target_pixels': n_truth,
                   'dice': 2 * intersection / den if den else 1.0,
                   'captured_in_original_preflight': i in meta['targets']}
            rows.append(row)
            seq_rows.append(row)
        eligible = [x for x in seq_rows if x['after_initialization']]
        present = [x for x in eligible if x['present']]
        empty = [x for x in eligible if not x['present']]
        summaries[seq] = {'post_prompt_frames': len(eligible), 'present': len(present),
                          'empty': len(empty), 'empty_false_alarms': sum(x['predicted_pixels'] > 0 for x in empty),
                          'present_dice_under_half': sum(x['dice'] < .5 for x in present),
                          'empty_false_alarms_not_in_original_targets': [x['frame'] for x in empty if x['predicted_pixels'] > 0 and not x['captured_in_original_preflight']]}
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'Label-inspected training-video coverage audit; capture flags refer only to the original split preflight, not later extra captures. Not held-out evaluation.',
              'source_sha256': digest(Path(__file__)), 'preflight_sha256': digest(manifest_path),
              'capture_receipts': receipts, 'summary': summaries, 'rows': rows}
    (out / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'coverage_sha256': digest(out / 'coverage.json')}, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()

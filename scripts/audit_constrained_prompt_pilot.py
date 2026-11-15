#!/usr/bin/env python3
"""Independently recompute the paired prompt pilot from saved masks."""
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


def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--merged', type=Path, required=True)
    a = p.parse_args()
    done = read(a.run / 'completion.json'); setup = read(a.run / 'setup.json')
    if not done['passed'] or digest(a.run / 'results.json') != done['results_sha256'] or digest(a.run / 'predictions.json') != done['predictions_sha256']:
        raise ValueError('Pilot incomplete or changed')
    merged = read(a.merged / 'completion.json')
    if not merged['passed'] or digest(a.merged / 'results.json') != merged['results_sha256'] or merged['results_sha256'] != setup['reference_results_sha256']:
        raise ValueError('Reference changed')
    locations = read(a.merged / 'locations.json')
    predictions = read(a.run / 'predictions.json')
    results = read(a.run / 'results.json')
    if [r['sequence'] for r in predictions] != setup['sequences'] or len(predictions) != len(locations):
        raise ValueError('Pilot coverage differs')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'), **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    recomputed = []
    for row, saved in zip(predictions, results['sequences']):
        if any(saved[k] != v for k, v in row.items()): raise ValueError('Prediction receipt differs')
        name, index = row['sequence'], row['frame']
        source = Path(locations[name]) / name
        prompt = read(source / 'prompt.json')
        if digest(source / 'prompt.json') != row['old_prompt_sha256'] or prompt['first_detection_index'] != index:
            raise ValueError('Prior prompt differs')
        frame = videos[name][index]
        if digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['label_sha256']:
            raise ValueError('Inputs differ')
        gt = load_label_mask(frame.mask_path, [1]) == 1
        files = {'box_only': source / 'prompt_provisional.png', 'original_points': source / 'prompt_refined.png',
                 'constrained_points': a.run / name / 'constrained.png'}
        if digest(files['box_only']) != prompt['provisional_mask_sha256'] or digest(files['original_points']) != prompt['refined_mask_sha256'] or digest(files['constrained_points']) != row['mask_sha256']:
            raise ValueError('Prediction mask differs')
        scores = {}
        for arm, path in files.items():
            pred = np.asarray(Image.open(path)) > 0
            if pred.shape != gt.shape: raise ValueError('Mask shape differs')
            denominator = int(pred.sum()+gt.sum())
            value = 2 * int((pred & gt).sum())/denominator if gt.any() and denominator else None
            alarm = bool(pred.any()) if not gt.any() else None
            observed = saved['scores'][arm]
            if (value is None) != (observed['dice'] is None) or (value is not None and not np.isclose(value, observed['dice'], atol=1e-12, rtol=0)) or alarm != observed['foreground_on_labeled_empty']:
                raise ValueError('Scoring differs')
            scores[arm] = value
        recomputed.append((bool(gt.any()), scores))
    present = [scores for flag, scores in recomputed if flag]
    mean = {arm: float(np.mean([x[arm] for x in present])) for arm in ('box_only','original_points','constrained_points')}
    if results['present_frames'] != len(present) or any(not np.isclose(mean[k], results['mean_present_dice'][k], atol=1e-12, rtol=0) for k in mean):
        raise ValueError('Aggregate differs')
    differences = np.array([x['constrained_points'] - x['original_points'] for x in present])
    audit = {'passed': True, 'results_sha256': done['results_sha256'], 'frames': len(predictions),
             'present_frames': len(present), 'mean_present_dice': mean,
             'constrained_minus_original': float(differences.mean()),
             'wins': int((differences > 1e-12).sum()), 'ties': int((np.abs(differences) <= 1e-12).sum()),
             'losses': int((differences < -1e-12).sum()), 'audit_source_sha256': digest(Path(__file__))}
    target = a.run / 'independent_audit.json'
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Score six frozen fixed-read arms after a separate no-label replay audit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_refiner_shared_v3 import ARMS, dependencies, ensure_target, load_prepared, read
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.train_source_group_refiner_v3 import require_parity, require_plan
from scripts.verify_source_group_refiner_v3 import check_frozen_arm


def metric(pred, truth):
    if pred.shape != truth.shape: raise ValueError('Prediction/label shape differs')
    tp = int((pred & truth).sum()); pp = int(pred.sum()); gp = int(truth.sum()); denominator = pp+gp
    return {'dice': 2*tp/denominator if denominator else 1.0,
            'precision': tp/pp if pp else None, 'recall': tp/gp if gp else None,
            'predicted_pixels': pp, 'false_alarm': bool(pp) if not gp else None}


def summary(rows, arm):
    present = [r[arm] for r in rows if r['present']]; empty = [r[arm] for r in rows if not r['present']]
    precision = [r['precision'] for r in present if r['precision'] is not None]
    return {'present_frames': len(present), 'mean_present_dice': float(np.mean([r['dice'] for r in present])) if present else None,
            'mean_present_precision_when_defined': float(np.mean(precision)) if precision else None,
            'precision_defined_frames': len(precision),
            'mean_present_recall': float(np.mean([r['recall'] for r in present])) if present else None,
            'empty_frames': len(empty), 'empty_false_alarms': sum(r['false_alarm'] for r in empty),
            'mean_empty_predicted_pixels': float(np.mean([r['predicted_pixels'] for r in empty])) if empty else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('manifest', 'plan', 'parity', 'verification', 'run-root', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args()
    manifest, manifest_sha = load_prepared(a.manifest); plan = require_plan(a.plan, manifest, manifest_sha)
    require_parity(a.parity, plan, a.plan, manifest_sha)
    verified = read(a.verification/'verification.json'); done = read(a.verification/'completion.json')
    train_paths, blocked_paths = mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    if (not done.get('passed') or done.get('verification_sha256') != digest(a.verification/'verification.json')
            or not verified.get('passed') or verified.get('manifest_sha256') != manifest_sha
            or verified.get('plan_sha256') != digest(a.plan) or verified.get('parity_sha256') != digest(a.parity)
            or verified.get('dependency_sha256') != dependencies() or verified.get('environment') != plan['environment']
            or verified.get('runner_sha256') != digest(ROOT/'scripts/verify_source_group_refiner_v3.py')
            or verified.get('label_guard', {}).get('open_attempts') != 0
            or verified.get('label_guard', {}).get('blocked_paths') != len(train_paths | blocked_paths)
            or set(verified.get('runs', {})) != set(ARMS)):
        raise ValueError('Frozen no-label replay verification differs')
    runs = {}
    # Recheck all six completions, checkpoints and every frozen output before
    # opening the first validation label. The separate verifier already replayed
    # the corresponding masks on the capture device with a mask-open guard.
    for arm in ARMS:
        folder = a.run_root/arm; ev = verified['runs'][arm]
        if (Path(ev['completion_path']).resolve() != (folder/'completion.json').resolve()
                or ev['completion_sha256'] != digest(folder/'completion.json')
                or ev['checkpoint_sha256'] != digest(folder/'checkpoint.pt')
                or ev['frozen_sha256'] != digest(folder/'frozen_predictions.json')):
            raise ValueError(f'{arm}: replay-verified artifacts changed')
        runs[arm] = check_frozen_arm(folder, arm, manifest, manifest_sha, a.plan, a.parity, plan)
    targets = sorted((x for x in manifest['samples'] if x['split']=='validation'), key=lambda x: (x['sequence'], x['frame']))
    rows = []
    for item in targets:
        key = (item['sequence'], item['frame']); row = {'sequence': key[0], 'frame': key[1]}
        shape = tuple(runs[ARMS[0]]['mapping'][key]['shape'])
        truth = ensure_target(item, shape)[0,0].numpy().astype(bool)
        row['present'] = bool(truth.any())
        for arm, run in runs.items():
            saved = run['mapping'][key]
            for kind in ('baseline', 'refined'):
                name = 'baseline' if kind == 'baseline' else arm
                if name in row: continue
                with Image.open(saved[kind]) as img: pred = np.asarray(img.convert('L'))>0
                row[name] = metric(pred, truth)
        rows.append(row)
    names = ('baseline', *ARMS)
    results = {name: {'pooled_frames': summary(rows, name),
                      'per_video': {seq: summary([r for r in rows if r['sequence']==seq], name)
                                    for seq in sorted({r['sequence'] for r in rows})}} for name in names}
    for value in results.values():
        video_means = [x['mean_present_dice'] for x in value['per_video'].values() if x['mean_present_dice'] is not None]
        value['equal_video_mean_present_dice'] = float(np.mean(video_means)) if video_means else None
    lines = ['# Source-group fixed-read exploratory validation', '',
             'Previously inspected validation videos; these are fixed-read results, not full-video or patient-independent evidence.', '',
             '| Arm | Present-frame Dice | Labeled-empty false alarms |', '|---|---:|---:|']
    for name in names:
        s = results[name]['pooled_frames']; dice = 'n/a' if s['mean_present_dice'] is None else f"{s['mean_present_dice']:.4f}"
        lines.append(f"| {name} | {dice} | {s['empty_false_alarms']}/{s['empty_frames']} |")
    lines += ['', 'The same source-group capture, checkpoint, seed, training order, optimizer, loss and parameter count are used for each fitted arm. The RGB control replaces only the contrast-map input and still uses SAM features; spatial shuffle applies a fixed bijection to aligned feature positions. Empty-frame labels may be incomplete; zero empty frames means false-alarm behavior is unmeasured.', '',
              'All validation masks for every arm were frozen and replay-verified before this scoring process opened any validation label. The CPython mask-open guards do not intercept arbitrary native-code system calls. A single seed and two previously inspected validation videos cannot establish generalization.']
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    report = out/'report.md'; report.write_text('\n'.join(lines)+'\n')
    result = {'passed': True, 'scope': 'Exploratory fixed-read validation, not live inference.',
              'manifest_sha256': manifest_sha, 'plan_sha256': digest(a.plan),
              'verification_sha256': digest(a.verification/'verification.json'),
              'scorer_sha256': digest(Path(__file__)), 'report_sha256': digest(report), 'summary': results, 'rows': rows}
    file = out/'scores.json'; file.write_text(json.dumps(result, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'scores_sha256': digest(file)}, indent=2)+'\n')
    print(report.read_text())


if __name__ == '__main__':
    main()

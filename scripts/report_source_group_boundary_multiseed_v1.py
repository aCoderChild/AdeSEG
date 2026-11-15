#!/usr/bin/env python3
"""Summarize the original and both independently audited seed results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output

BASE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1')
PRIOR = BASE/'source_group_boundary_factorial_v1_bvqxd475'
PRIOR_AUDIT = BASE/'source_group_boundary_factorial_v1_bvqxd475_independent_audit_attempt01/audit.json'
ARMS = tuple(f'{m}_{o}' for m in ('region', 'no_contrast', 'mask_only') for o in ('bce', 'boundary'))
SEEDS = (20261125, 20261135)


def read(path):
    return json.loads(Path(path).read_text())


def identities(manifest):
    keys = ('split', 'sequence', 'frame', 'image_sha256', 'label_sha256', 'feature_sha256', 'snapshot_sha256')
    rows = [tuple(item[k] for k in keys) for item in manifest['samples']]
    if len(rows) != 214 or len(set((x[1], x[2]) for x in rows)) != 214:
        raise ValueError('Sample identity coverage differs')
    return sorted(rows)


def collect(run):
    run = validate_output(run).resolve()
    original = read(PRIOR/'scores/scores.json')
    prior_audit = read(PRIOR_AUDIT)
    prior_manifest = PRIOR/'manifest/manifest.json'
    current_manifest = run/'manifest/manifest.json'
    before, current = read(prior_manifest), read(current_manifest)
    bound = ('checkpoint_sha256', 'feature_pipeline_sha256', 'source_freeze_sha256',
             'preflight_sha256', 'short_gate_sha256')
    if (not original.get('passed') or not prior_audit.get('passed')
            or prior_audit.get('scores_sha256') != digest(PRIOR/'scores/scores.json')
            or prior_audit.get('actual_manifest_sha256') != digest(prior_manifest)
            or any(not before.get(k) or before.get(k) != current.get(k) for k in bound)
            or identities(before) != identities(current)):
        raise ValueError('Original factorial or exact sample identities differ')
    outcomes = {'20261115': original['summary']}
    for seed in SEEDS:
        base = run/str(seed)
        score, audit = base/'scores/scores.json', base/'independent_audit/audit.json'
        result, check = read(score), read(audit)
        done = read(audit.parent/'completion.json')
        if (not result.get('passed') or not check.get('passed') or not done.get('passed')
                or done.get('audit_sha256') != digest(audit) or check.get('seed') != seed
                or check.get('scores_sha256') != digest(score) or check.get('error_count') != 0
                or check.get('actual_manifest_sha256') != digest(current_manifest)
                or check.get('mask_files_checked') != 600 or check.get('validation_frames') != 50):
            raise ValueError(f'{seed}: independent audit differs')
        outcomes[str(seed)] = result['summary']
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = collect(args.run)
    lines = ['# Exploratory boundary-factorial outcomes by training seed', '',
             'The additional seeds were chosen after the first seed and development videos were inspected. '
             'All use the same fixed reads; they are not independent patient or video evaluations. '
             'The prior run did not record PYTHONHASHSEED, whereas both added seeds explicitly use 0.', '',
             '| Seed | Arm | Present Dice | Equal-video Dice | Boundary F, 5 px | Empty false alarms |',
             '|---:|---|---:|---:|---:|---:|']
    for seed, summary in results.items():
        for name in ('baseline', *ARMS):
            s = summary[name]['pooled_frames']
            lines.append(f"| {seed} | {name} | {s['mean_present_dice']:.4f} | "
                         f"{summary[name]['equal_video_mean_present_dice']:.4f} | {s['mean_present_boundary_f_5px']:.4f} | "
                         f"{s['empty_false_alarms']}/{s['empty_frames']} |")
    lines += ['', 'Each additional seed completed all six arms and label-free replay before either seed was scored. '
              'The fixed boundary objective changes both weighted BCE and IoU. Empty labels may be incomplete; '
              'three empty frames cannot establish false-alarm behavior. Do not select only the best seed.']
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = out/'report.md'; report.write_text('\n'.join(lines)+'\n')
    receipt = out/'summary.json'
    receipt.write_text(json.dumps({'passed': True, 'outcomes': results,
        'report_sha256': digest(report), 'runner_sha256': digest(Path(__file__)),
        'prior_audit_sha256': digest(PRIOR_AUDIT),
        'audit_sha256': {str(s): digest(Path(args.run)/str(s)/'independent_audit/audit.json') for s in SEEDS}}, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'summary_sha256': digest(receipt)}, indent=2)+'\n')
    print(report.read_text())


if __name__ == '__main__':
    main()

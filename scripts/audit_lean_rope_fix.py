#!/usr/bin/env python3
"""Independently validate and score the corrected automatic Lean cohort."""
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


def read(path): return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', type=Path, action='append', required=True)
    for arg in ('merged', 'historical', 'verification', 'fixed-gate', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    a = parser.parse_args()
    protocol = read(ROOT/'protocols/lean_cpu_rope_repeat_fix_v1.json')
    verification = read(a.verification)
    merge = read(a.merged/'completion.json')
    if (not verification['passed'] or not merge['passed']
        or digest(a.merged/'results.json') != merge['results_sha256']
        or digest(a.historical/'comparison.json') != merge['historical_comparison_sha256']):
        raise ValueError('Verification or reference comparison invalid')
    old_all = read(a.merged/'results.json')['sequences']
    locations = read(a.merged/'locations.json')
    history = {x['sequence']: x for x in read(a.historical/'comparison.json')['sequences']}
    videos = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'test')}
    gate = read(a.fixed_gate/'records.json')
    if digest(a.fixed_gate/'records.json') != verification['fixed_records_sha256']:
        raise ValueError('Fixed gate changed')
    results = {}
    for part in a.part:
        setup, done = read(part/'setup.json'), read(part/'completion.json')
        if (not done['passed'] or done['sequences'] != setup['sequences']
            or setup['verification_sha256'] != digest(a.verification)
            or setup['runner_sha256'] != verification['identity']['runner_sha256']):
            raise ValueError('Partition invalid')
        for name in setup['sequences']:
            if name in results: raise ValueError('Overlapping partitions')
            run = part/name
            record, receipt, meta = read(run/'records.json'), read(run/'completion.json'), read(run/'setup.json')
            if (not receipt['passed'] or digest(run/'records.json') != receipt['records_sha256']
                or meta['mode'] != 'fixed' or meta['sequence'] != name or meta['identity'] != verification['identity']
                or not read(run/'parameter_integrity_after.json')['unchanged']):
                raise ValueError(f'{name}: invalid run')
            for key, filename in [('frames_sha256','frames.json'), ('frequency_trace_sha256','frequency_trace.json'),
                                  ('selections_sha256','selections.json'), ('memory_events_sha256','memory_events.json')]:
                if digest(run/filename) != record[key]: raise ValueError(f'{name}: {filename} changed')
            selections = read(run/'selections.json')
            if any(x['candidates'] != 4096 or x['kept'] != 1024 for x in selections):
                raise ValueError(f'{name}: token budget changed')
            trace = read(run/'frequency_trace.json')
            if not trace or len(trace) != record['affected_calls'] or min(x['frame'] for x in trace) != record['first_affected_frame']:
                raise ValueError(f'{name}: frequency trace invalid')
            original = Path(locations[name]); old_receipt = read(original/'completion.json'); old_audit = read(original/'independent_audit.json')
            if (not old_receipt['passed'] or not old_audit['passed'] or digest(original/'records.json') != old_receipt['records_sha256']
                or old_audit['records_sha256'] != old_receipt['records_sha256']
                or record['original_records_sha256'] != old_receipt['records_sha256']):
                raise ValueError(f'{name}: original run changed')
            old_record = read(original/'records.json')[0]
            old_folder = original/name
            if digest(old_folder/'frames.json') != old_record['frames_sha256']:
                raise ValueError('Original frame record changed')
            old_rows, rows, frames = read(old_folder/'frames.json'), read(run/'frames.json'), videos[name]
            if not (len(rows) == len(old_rows) == len(frames) == record['frames'] == len(record['mask_sha256'])):
                raise ValueError('Frame count mismatch')
            values = []
            for i, (row, old, frame) in enumerate(zip(rows, old_rows, frames)):
                path = run/f'{i:06d}.png'
                if (row['frame'] != i or digest(path) != row['mask_sha256'] or row['mask_sha256'] != record['mask_sha256'][i]
                    or digest(frame.mask_path) != old['label_sha256'] or digest(frame.image_path) != old['image_sha256']
                    or (i < record['first_affected_frame'] and row['mask_sha256'] != old['mask_sha256'])):
                    raise ValueError(f'{name}/{i}: input or output changed')
                pred = np.asarray(Image.open(path)) > 0
                truth = load_label_mask(frame.mask_path, [1]) == 1
                if pred.shape != truth.shape: raise ValueError('Mask shape differs')
                denom = int(pred.sum() + truth.sum()); present = bool(truth.any())
                dice = 2*int((pred & truth).sum())/denom if denom else 1.
                alarm = bool(pred.any()) if not present else None
                if row['present'] != present or old['present'] != present or not np.isclose(row['dice'], dice, atol=1e-12, rtol=0) or row['false_alarm'] != alarm:
                    raise ValueError(f'{name}/{i}: score mismatch')
                values.append((present, dice, alarm))
            targets = [json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()]
            if len(targets) != history[name]['targets'] or any(values[x['frame']][0] != x['present'] for x in targets):
                raise ValueError('Historical targets differ')
            present = [x[1] for x in values if x[0]]; empty = [x[2] for x in values if not x[0]]
            hp = [values[x['frame']][1] for x in targets if x['present']]
            he = [values[x['frame']][2] for x in targets if not x['present']]
            results[name] = {'frames':len(values), 'all_present':len(present), 'all_empty':len(empty),
                'corrected_all_dice':float(np.mean(present)), 'original_all_dice':old_all[name]['all_present_dice'],
                'corrected_all_false_alarms':sum(empty), 'original_all_false_alarms':old_all[name]['all_false_alarms'],
                'historical_present':len(hp), 'historical_empty':len(he),
                'corrected_historical_dice':float(np.mean(hp)), 'original_historical_dice':old_all[name]['historical_present_dice'],
                'corrected_historical_false_alarms':sum(he), 'original_historical_false_alarms':old_all[name]['historical_false_alarms'],
                'native_gt_prompt_dice':history[name]['native_dice'], 'lean_gt_prompt_dice':history[name]['lean_dice'],
                'changed_frame_masks':sum(row['mask_sha256'] != old['mask_sha256'] for row,old in zip(rows,old_rows)),
                'selection_events':len(selections)}
            if name == 'seq3':
                for key in ('mask_sha256','selections_sha256','frequency_trace_sha256','memory_events_sha256'):
                    if record[key] != gate[key]: raise ValueError('Full seq3 differs from fixed engineering run')
    if set(results) != set(protocol['cohort']): raise ValueError('Cohort incomplete')
    rows = list(results.values())
    deltas = np.array([r['corrected_historical_dice'] - r['original_historical_dice'] for r in rows])
    rng = np.random.default_rng(20261114)
    boot = rng.choice(deltas, (10000, len(deltas)), replace=True).mean(axis=1)
    summary = {'scope':'Same automatic prompt, retained token count, pooled anchor and video; only CPU key-frequency repetition differs. Historical native and Lean use GT initial masks and are descriptive references.',
        'sequences':results,
        'original_all_dice':float(np.mean([r['original_all_dice'] for r in rows])),
        'corrected_all_dice':float(np.mean([r['corrected_all_dice'] for r in rows])),
        'original_historical_dice':float(np.mean([r['original_historical_dice'] for r in rows])),
        'corrected_historical_dice':float(np.mean([r['corrected_historical_dice'] for r in rows])),
        'paired_historical_delta_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'original_all_false_alarms':sum(r['original_all_false_alarms'] for r in rows),
        'corrected_all_false_alarms':sum(r['corrected_all_false_alarms'] for r in rows),
        'all_empty':sum(r['all_empty'] for r in rows),
        'original_historical_false_alarms':sum(r['original_historical_false_alarms'] for r in rows),
        'corrected_historical_false_alarms':sum(r['corrected_historical_false_alarms'] for r in rows),
        'historical_empty':sum(r['historical_empty'] for r in rows),
        'below_best_gt_prompt_reference':sum(r['corrected_historical_dice'] < max(r['native_gt_prompt_dice'],r['lean_gt_prompt_dice'])-1e-12 for r in rows)}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    (out/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# CPU rotary-frequency correction: independent cohort audit','',summary['scope'],'',
      '| Sequence | Native GT | Lean GT | Original automatic | Corrected automatic | Change | Empty-frame false alarms original / corrected |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        lines.append(f"| {name} | {r['native_gt_prompt_dice']:.3f} | {r['lean_gt_prompt_dice']:.3f} | {r['original_historical_dice']:.3f} | {r['corrected_historical_dice']:.3f} | {r['corrected_historical_dice']-r['original_historical_dice']:+.3f} | {r['original_all_false_alarms']} / {r['corrected_all_false_alarms']} ({r['all_empty']}) |")
    lines += ['',f"Equal-sequence all-present-frame Dice: {summary['original_all_dice']:.4f} original, {summary['corrected_all_dice']:.4f} corrected. False alarms: {summary['original_all_false_alarms']} / {summary['corrected_all_false_alarms']} out of {summary['all_empty']} labeled-empty frames.",
      f"Historical matched-target Dice: {summary['original_historical_dice']:.4f} original, {summary['corrected_historical_dice']:.4f} corrected. Paired sequence-bootstrap 95% interval for the change: {summary['paired_historical_delta_ci95']}. {summary['below_best_gt_prompt_reference']}/21 corrected sequences are below the retrospectively better GT-prompt reference.",
      'These exploratory results do not establish generalization. The original CPU helper differs from the CUDA branch; the correction does not establish whole-model CPU/GPU parity, nor fix separate retained-token positional indexing and query-grid fallback concerns. No labels enter inference. No new re-detection, threshold or adaptive model gate was introduced.','']
    (out/'report.md').write_text('\n'.join(lines))
    (out/'completion.json').write_text(json.dumps({'passed':True, 'results_sha256':digest(out/'results.json'), 'audit_sha256':digest(Path(__file__))}, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__': main()

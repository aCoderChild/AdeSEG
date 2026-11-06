"""Verify completed Run168 artifacts and report the frozen-history diagnostic.

No model inference. Reuses the project's GT decoding, segmentation metrics,
sequence statistics and author SAM2's bilinear output-resize convention.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from scripts.memory_token_io import digest
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_pilot import aggregate, atomic_json

ARMS = ('native', 'generic', 'regional')
METRICS = ('present_dice', 'all_frame_dice', 'present_iou', 'present_boundary_f',
           'gt_empty_false_alarm_rate')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b, message):
    require(bool(np.isclose(a, b, rtol=0, atol=1e-12)), message)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_sequence(folder, frames, inputs, native_reference):
    """Recompute scores and reconstruct each PNG from the saved low-res logits."""
    seq = folder.name
    summary = json.loads((folder / 'summary.json').read_text())
    require(summary['passed'], f'{seq}: sequence not complete')
    for name, sha in summary['receipt_hashes'].items():
        require(digest(folder / name) == sha, f'{seq}: changed receipt {name}')
    expected = list(range(inputs['prompt'] + 1, len(frames)))
    require(summary['targets'] == len(expected), f'{seq}: target count mismatch')
    integrity = jsonl(folder / 'integrity.jsonl')
    require([r['frame'] for r in integrity] == expected, f'{seq}: missing integrity targets')
    checks = ('native_exact', 'both_zero_controls_exact', 'native_restoration_exact', 'no_writeback')
    require(all(all(r[k] is True for k in checks) for r in integrity), f'{seq}: integrity failure')
    rows = {a: jsonl(folder / f'{a}_frames.jsonl') for a in ARMS}
    for arm in ARMS:
        require([r['frame'] for r in rows[arm]] == expected, f'{seq}/{arm}: incomplete order')
        for extension in ('png', 'pt'):
            require(sorted(p.stem for p in (folder / arm).glob(f'*.{extension}')) ==
                    [f'{f:06d}' for f in expected], f'{seq}/{arm}: tensor/mask coverage')
    receipts = {r['local_frame']: r for r in inputs['frames']}
    paired = []
    for index, frame in enumerate(expected):
        item = frames[frame]
        require(digest(item.mask_path) == receipts[frame]['mask_sha256'], f'{seq}/{frame}: GT changed')
        truth = load_label_mask(item.mask_path, [1]) == 1
        record = {'sequence': seq, 'frame': frame, 'source_frame_index': item.frame_index,
                  'present': bool(truth.any()), 'arms': {}}
        for arm in ARMS:
            row = rows[arm][index]
            prefix = f'{seq}/{frame}/{arm}'
            require(row['source_frame_index'] == item.frame_index == receipts[frame]['source_frame'], prefix + ': frame mismatch')
            require(row['present'] == record['present'], prefix + ': presence mismatch')
            mask_path = folder / arm / f'{frame:06d}.png'
            tensor_path = mask_path.with_suffix('.pt')
            require(digest(mask_path) == row['mask_sha256'], prefix + ': mask hash mismatch')
            require(digest(tensor_path) == row['tensor_sha256'], prefix + ': tensor hash mismatch')
            array = np.array(Image.open(mask_path))
            require(set(np.unique(array)).issubset({0, 255}), prefix + ': invalid mask labels')
            pred = array > 0
            tensors = torch.load(tensor_path, map_location='cpu', weights_only=True)
            require(all(bool(torch.isfinite(t).all()) for t in tensors.values()), prefix + ': nonfinite tensors')
            low = tensors['pred_masks']
            require(tuple(low.shape) == (1, 1, 256, 256), prefix + ': low-res geometry changed')
            full = F.interpolate(low, size=truth.shape, mode='bilinear', align_corners=False)
            require(tensor_hash(full) == row['prediction_sha256'], prefix + ': reconstructed logit mismatch')
            require(np.array_equal(full[0, 0].numpy() > 0, pred), prefix + ': PNG/logit mismatch')
            if arm == 'native' and native_reference is not None:
                require(row['prediction_sha256'] == native_reference[frame]['prediction_sha256'], prefix + ': Run167 native mismatch')
            scores = calculate_scores(pred, truth)
            scores['boundary_f'] = float(db_eval_boundary(pred, truth))
            for metric, value in scores.items():
                close(value, row['scores'][metric], prefix + ': score mismatch ' + metric)
            score = tensors['object_score_logits'].item()
            close(score, row['object_score_logit'], prefix + ': object score mismatch')
            if score <= 0:
                require(not pred.any(), prefix + ': native absence suppression mismatch')
            if not truth.any():
                require(row['absent_fp'] == bool(pred.any()), prefix + ': false alarm mismatch')
            record['arms'][arm] = {
                'dice': scores['dice'], 'boundary_f': scores['boundary_f'],
                'object_score_logit': score, 'native_absence_decision': score <= 0,
                'raw_candidate_positive_fraction': float((tensors['raw_decoder_logits'] > 0).float().mean()),
                'predicted_area_fraction': float(pred.mean()),
                'state': ('false_alarm' if pred.any() else 'correct_empty') if not truth.any()
                    else 'missed_empty' if not pred.any() else 'overlap' if (pred & truth).any() else 'nonempty_disjoint',
                'residual': row['residual']}
        paired.append(record)
    for arm in ARMS:
        present = [r for r in rows[arm] if r['present']]
        absent = [r for r in rows[arm] if not r['present']]
        stored = summary['arms'][arm]
        require((stored['target_frames'], stored['present_frames'], stored['gt_empty_frames']) ==
                (len(expected), len(present), len(absent)), f'{seq}/{arm}: denominators differ')
        for metric in ('dice', 'iou', 'boundary_f'):
            close(np.mean([r['scores'][metric] for r in present]), stored[f'present_{metric}'], f'{seq}/{arm}: mean {metric}')
        close(np.mean([r['scores']['dice'] for r in rows[arm]]), stored['all_frame_dice'], f'{seq}/{arm}: all Dice')
        require(sum(r['absent_fp'] for r in absent) == stored['gt_empty_false_alarm_count'], f'{seq}/{arm}: FA count')
        if absent:
            close(np.mean([r['absent_fp'] for r in absent]), stored['gt_empty_false_alarm_rate'], f'{seq}/{arm}: FA rate')
    return summary['arms'], paired


def summarize_diagnostics(rows):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for present in (True, False):
            subset = [r['arms'][arm] for r in rows if r['present'] == present]
            entry = {'frames': len(subset), 'native_absence_decisions': sum(r['native_absence_decision'] for r in subset),
                     'states': {s: sum(r['state'] == s for r in subset) for s in
                         ('overlap', 'missed_empty', 'nonempty_disjoint', 'false_alarm', 'correct_empty')}}
            for key in ('object_score_logit', 'raw_candidate_positive_fraction', 'predicted_area_fraction'):
                values = [r[key] for r in subset]
                entry[key] = dict(zip(('min', 'median', 'max'), map(float, np.quantile(values, [0, .5, 1]))))
            if arm != 'native':
                values = [r['residual']['rms'] / r['residual']['native_rms'] for r in subset]
                require(all(np.isfinite(values)), 'Nonfinite residual/native RMS ratio')
                entry['residual_over_native_rms'] = dict(zip(('min', 'median', 'max'), map(float, np.quantile(values, [0, .5, 1]))))
            result[arm]['present' if present else 'empty'] = entry
    result['decision_flips'] = {a: {str(present): {
        'native_absent_reader_present': sum(r['arms']['native']['native_absence_decision'] and not r['arms'][a]['native_absence_decision'] for r in rows if r['present'] == present),
        'native_present_reader_absent': sum(not r['arms']['native']['native_absence_decision'] and r['arms'][a]['native_absence_decision'] for r in rows if r['present'] == present)}
        for present in (True, False)} for a in ('generic', 'regional')}
    return result


def report(result, full, diagnostics):
    lines = ['# Run168: fixed-native-memory replay', '',
        'Post-hoc diagnostic of unchanged Run167 readers. Eight previously examined',
        'sequences, 372 post-prompt targets. No replay predictions enter later memory.',
        'Sequence is the unit. This is not a full-propagation benchmark.', '',
        '| Arm | Replay present Dice | Full-propagation Dice | Replay − full | Replay Boundary F | Replay false alarms |',
        '|---|---:|---:|---:|---:|---:|']
    seqs = list(result['per_sequence'])
    for arm in ARMS:
        values = [result['per_sequence'][s][arm] for s in seqs]
        score = result['equal_sequence'][arm]
        old = full['equal_sequence'][arm]
        fa = sum(r['gt_empty_false_alarm_count'] for r in values)
        absent = sum(r['gt_empty_frames'] for r in values)
        lines.append(f'| {arm} | {score:.6f} | {old:.6f} | {score-old:+.6f} | {np.mean([r["present_boundary_f"] for r in values]):.6f} | {fa}/{absent} |')
    lines += ['', '| Contrast on native memory | Dice Δ | 95% sequence-bootstrap CI | Wins / ties / losses | Holm p |',
              '|---|---:|---|---:|---:|']
    for name, c in result['primary_contrasts'].items():
        lo, hi = c['sequence_bootstrap_95_ci']
        lines.append(f'| {name} | {c["mean_present_dice_difference"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c["wins"]}/{c["ties"]}/{c["losses"]} | {c["holm_adjusted_p"]:.4f} |')
    lines += ['', '| Sequence | Native | Generic replay | Regional replay | Generic full | Regional full |', '|---|---:|---:|---:|---:|---:|']
    for s in seqs:
        v = result['per_sequence'][s]
        f = full['per_sequence'][s]
        lines.append(f'| {s} | {v["native"]["present_dice"]:.6f} | {v["generic"]["present_dice"]:.6f} | {v["regional"]["present_dice"]:.6f} | {f["generic"]["present_dice"]:.6f} | {f["regional"]["present_dice"]:.6f} |')
    lines += ['', '| Arm | GT-present suppressed by native object-score rule | GT-present empty masks | GT-empty false alarms |', '|---|---:|---:|---:|']
    for arm in ARMS:
        d = diagnostics[arm]
        lines.append(f'| {arm} | {d["present"]["native_absence_decisions"]}/{d["present"]["frames"]} | {d["present"]["states"]["missed_empty"]} | {d["empty"]["states"]["false_alarm"]}/{d["empty"]["frames"]} |')
    lines += ['', 'The object-score rule is unchanged author SAM2 behavior, not an added gate.',
        'Raw decoder candidate logits are saved before this suppression; candidate',
        'positivity alone does not establish a correct selected mask. Residual RMS',
        'ratios describe perturbation magnitude, not information quality.', '',
        'Replay-versus-full differences measure an intervention on the entire prior',
        'trajectory. They can include history-distribution shifts and interactions;',
        'they do not uniquely quantify error accumulation. Do not compute redetection',
        'from these independent replays or treat their Dice as deployable VOS accuracy.',
        'Run167 failed its efficacy continuation rule; this diagnostic does not override it.', '']
    return '\n'.join(lines)


def sync_docs(text, result):
    """Replace only this diagnostic's section, after complete artifact verification."""
    path = ROOT / 'docs/REGION_TOKENS_RESULTS.md'
    document = path.read_text()
    start = document.index('## Run 168: fixed-native-memory replay\n')
    end = document.index('## Published dataset references\n', start)
    section = text.replace('# Run168: fixed-native-memory replay', '## Run 168: fixed-native-memory replay', 1)
    section += ('\nVerified 73 engineering and 372 diagnostic targets; all 1,335 PNGs and\n'
                '1,335 tensor files passed offline hash/metric/reconstruction checks.\n'
                'Protocol commit `5342dfcb`; runner commit `da42eeed`; 33 focused tests passed.\n'
                'Drive `outputs/experiment/168_complementary_memory_replay_v1/report_attempt_01/`\n'
                'contains the report, paired frame diagnostics and verification receipt.\n\n')
    path.write_text(document[:start] + section + document[end:])
    path = ROOT / 'docs/REGION_TOKENS.md'
    document = path.read_text()
    start = document.index('**Next diagnostic (Run 168):**')
    end = document.index('\nEarlier regional ToMe', start)
    n, r = result['equal_sequence']['native'], result['equal_sequence']['regional']
    section = (f'**Run 168 diagnostic completed:** unchanged readers were evaluated on identical\n'
        f'native memory without prediction writeback. Regional replay Dice was {r:.6f}\n'
        f'versus native {n:.6f}. This is an immediate-read diagnostic, not a deployable\n'
        f'full-propagation score. All 73 engineering and 372 diagnostic targets passed\n'
        f'integrity checks and offline reconstruction. See the\n'
        f'[results tables](REGION_TOKENS_RESULTS.md#run-168-fixed-native-memory-replay).\n'
        f'Run167\'s failed efficacy decision remains unchanged; review these findings\n'
        f'before another architecture, training run or external efficacy evaluation.\n')
    path.write_text(document[:start] + section + document[end:])
    path = ROOT / 'protocols/LUNA_WORK_PLAN.md'
    document = path.read_text()
    anchor = '| Published-results comparison |'
    i = document.index(anchor)
    row = ('| Replay diagnostic/report | Complete; post-hoc only (' + datetime.now(timezone.utc).date().isoformat() +
        ') | Run168: 73 engineering + 372 diagnostic targets; 1,335 masks and 1,335 tensors independently verified. '
        'Drive `168_complementary_memory_replay_v1/report_attempt_01/` contains report and completion receipt. '
        'No new training, repeated seeds or external efficacy run. Review before choosing another method. |\n')
    document = document[:i] + row + document[i:]
    document = document.replace('| Replay implementation | Verified and running (2026-11-06) |',
                                '| Replay implementation | Verified; run subsequently completed |')
    path.write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--sync-docs', action='store_true', help='Update only Run168 results, method status and ledger after verification')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, out = args.run.resolve(), args.output_dir.resolve()
    require('GoogleDrive-' in str(out), 'Save the report in the Drive sync folder')
    require(not (run / 'failure.json').exists(), 'Experiment has a failure receipt')
    completion = json.loads((run / 'completion.json').read_text())
    require(completion['status'] == 'complete', 'Wait for experiment completion')
    require(digest(run / 'setup.json') == completion['setup_sha256'], 'Setup changed')
    require(digest(run / 'summary.json') == completion['summary_sha256'], 'Summary changed')
    setup = json.loads((run / 'setup.json').read_text())
    source = Path(setup['source_run'])
    require(digest(source / 'setup.json') == setup['source_run_setup_sha256'], 'Source setup changed')
    original = json.loads((source / 'setup.json').read_text())['setup_core']
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    require(digest(manifest) == original['manifest_sha256'], 'Manifest changed')
    for name, sha in setup['source_hashes'].items():
        require(digest(ROOT / name) == sha, f'Run168 source changed: {name}')
    videos = load_manifest(manifest, 'test') | load_manifest(manifest, 'dev')
    summaries = {a: {} for a in ARMS}
    rows, gate_count = [], 0
    for seq in setup['protocol']['engineering_sequences']:
        _, gate_rows = verify_sequence(run / 'gate' / seq, videos[seq], original['input_receipts'][seq], None)
        gate_count += len(gate_rows)
    require(gate_count == completion['gate_targets'] == 73, 'Gate coverage mismatch')
    for seq in setup['protocol']['diagnostic_sequences']:
        references = {r['frame']: r for r in jsonl(source / 'evaluation/native' / seq / 'frames.jsonl')}
        values, paired = verify_sequence(run / 'full' / seq, videos[seq], original['input_receipts'][seq], references)
        for a in ARMS:
            summaries[a][seq] = values[a]
        rows.extend(paired)
        print(f'Verified {seq}: {len(paired)} targets, all masks/tensors/metrics', flush=True)
    require(len(rows) == completion['diagnostic_targets'] == 372, 'Diagnostic coverage mismatch')
    result = aggregate(summaries)
    stored = json.loads((run / 'summary.json').read_text())
    require(all(result[k] == stored[k] for k in result), 'Aggregate statistics changed')
    full = json.loads((source / 'seed_20261106_summary.json').read_text())
    require(result['equal_sequence']['native'] == full['equal_sequence']['native'], 'Native score changed')
    diagnostics = summarize_diagnostics(rows)
    out.mkdir(parents=True, exist_ok=False)
    result['diagnostics'] = diagnostics
    result['replay_minus_full_propagation'] = stored['replay_minus_full_propagation']
    result['equal_sequence_secondary'] = {a: {m: float(np.mean([summaries[a][s][m] for s in summaries[a]])) for m in METRICS} for a in ARMS}
    atomic_json(out / 'analysis.json', result)
    (out / 'paired_frames.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    text = report(result, full, diagnostics)
    (out / 'report.md').write_text(text)
    atomic_json(out / 'completion.json', {'status': 'complete', 'at': datetime.now(timezone.utc).isoformat(),
        'inference_run': False, 'verified_targets': len(rows), 'verified_gate_targets': gate_count,
        'verified_masks_and_tensor_files_each': 3 * (gate_count + len(rows)),
        'source_sha256': digest(Path(__file__)), 'run_completion_sha256': digest(run / 'completion.json'),
        'report_sha256': {n: digest(out / n) for n in ('analysis.json', 'paired_frames.jsonl', 'report.md')}})
    if args.sync_docs:
        sync_docs(text, result)
        atomic_json(out / 'docs_sync.json', {'at': datetime.now(timezone.utc).isoformat(), 'files':
            {p: digest(ROOT / p) for p in ('docs/REGION_TOKENS.md', 'docs/REGION_TOKENS_RESULTS.md', 'protocols/LUNA_WORK_PLAN.md')}})
    print(out / 'report.md', flush=True)


if __name__ == '__main__':
    main()

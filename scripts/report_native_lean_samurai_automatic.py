#!/usr/bin/env python3
"""Score three frozen automatic-mask comparators after validating every output.

Exploratory development sequences only. Source-image and mask checks for all
three sequences complete before any annotation pixels are opened.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.report_lean_mps_vs_samurai import independent_metric, independent_summary
from scripts.run_native_sam2_automatic_priority import identity, verify_gate
from scripts.run_samurai_automatic_priority import MANIFEST, prepare_archived_prompt
from scripts.score_lean_window_mps_vs_samurai import check_frozen


def read(path):
    return json.loads(Path(path).read_text())


def check_native(sequence, frames, native, gate_path, prompt, start, checkpoint):
    run = read(native / 'receipt.json')
    current = identity(checkpoint, 'mps')
    gate = verify_gate(gate_path, sequence, prompt, current)
    if (not run.get('passed') or not run.get('full_video') or run.get('sequence') != sequence
            or run.get('prompt_frame') != start or run.get('prompt') != prompt
            or run.get('identity') != current or run.get('device') != 'mps'
            or not run.get('deterministic') or not run.get('blocking_pointer_transfer')
            or run.get('state_sha256') != gate['state_sha256']
            or run.get('gate_sha256') != digest(gate_path) or run.get('rebased_start') != 0
            or run.get('frames') != len(frames) or run.get('post_prompt_frames') != len(frames)-start
            or len(run.get('records', [])) != len(frames)):
        raise ValueError(f'{sequence}: native receipt or gate differs')
    for i, frame in enumerate(frames):
        record = run['records'][i]
        file = native / f'{i:06d}.png'
        if (record['frame'] != i or record['source_frame_index'] != frame.frame_index
                or record['image_sha256'] != digest(frame.image_path)
                or record['mask_sha256'] != digest(file) or record['post_prompt'] != (i >= start)):
            raise ValueError(f'{sequence}: native image or prediction changed at frame {i}')
        with Image.open(file) as im, Image.open(frame.image_path) as source:
            if im.size != source.size or (i < start and np.asarray(im.convert('L')).any()):
                raise ValueError(f'{sequence}: native prediction geometry or pre-prompt frame differs')
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.experiment_root
    v = base / 'contrast_refinement_v1'
    sam_dir = base / 'samurai_automatic_priority_v1'
    nat_dir = base / 'native_sam2_automatic_priority_v1'
    preflight = base / 'lean_channel_attention_training_pilot_v1/split_preflight_attempt03'
    cr = base / 'lean_token_window_candidate_v1/seq23_attempt01'
    cb = base / 'lean_seq23_no_inset_v1/full_attempt02'
    checkpoint = base / 'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    verified = {}
    # Check all sequences and all masks before reading any label.
    for seq in ('seq11', 'seq20', 'seq23'):
        frames = videos[seq]
        lean = v / f'lean_mps_baseline_{seq}_full_attempt01'
        sam = sam_dir / f'{seq}_full_mps_attempt02'
        native = nat_dir / f'{seq}_full_mps_attempt01'
        lg = v / f'lean_mps_baseline_{seq}_gate_attempt02/gate.json'
        sg = sam_dir / f'{seq}_gate_verified_v3/gate.json'
        ng = nat_dir / f'{seq}_gate_verified_attempt01/gate.json'
        _, _, _, prompt = prepare_archived_prompt(seq, frames, preflight, cr, cb)
        start, l, s = check_frozen(seq, frames, lean, sam, lg, sg, prompt)
        n = check_native(seq, frames, native, ng, prompt, start, checkpoint)
        if n['identity']['checkpoint_sha256'] != l['identity']['checkpoint_sha256'] or n['identity']['torch'] != s['torch']:
            raise ValueError(f'{seq}: backend or checkpoint differs')
        verified[seq] = (frames, lean, sam, native, start, l, s, n)

    pre = read(preflight / 'manifest.json')
    summaries, evidence, per_frame = {}, {}, {}
    for seq, (frames, lean, sam, native, start, l, s, n) in verified.items():
        meta = read(Path(pre['sequences'][seq]['source']) / seq / 'frames.json')
        previous = read(v / f'lean_mps_vs_samurai_{seq}_score_attempt01/results.json')
        if len(meta) != len(frames) or len(previous['rows']) != len(frames):
            raise ValueError(f'{seq}: label or previous-score coverage differs')
        rows = []
        for i, frame in enumerate(frames):
            if digest(frame.mask_path) != meta[i]['label_sha256']:
                raise ValueError(f'{seq}: annotation bytes changed')
            with Image.open(frame.mask_path) as im:
                if im.format != 'JPEG':
                    raise ValueError(f'{seq}: unexpected label format')
                gray = np.asarray(im.convert('L'))
            if ((gray > 64) & (gray < 192)).mean() > 0.05:
                raise ValueError(f'{seq}: label is not a binary JPEG')
            truth = gray > 127
            masks = {}
            for arm, path in (('native', native), ('lean', lean), ('samurai', sam)):
                if arm == 'samurai' and i < start:
                    masks[arm] = np.zeros_like(truth)
                else:
                    with Image.open(path / f'{i:06d}.png') as im:
                        masks[arm] = np.asarray(im.convert('L')) > 0
            row = {'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                   'initialization_frame': i == start,
                   **{arm: independent_metric(mask, truth) for arm, mask in masks.items()}}
            if {k: value for k, value in row.items() if k != 'native'} != previous['rows'][i]:
                raise ValueError(f'{seq}: independently reproduced previous score differs at frame {i}')
            rows.append(row)
        summaries[seq] = {arm: {'all': independent_summary(rows, arm, False),
                                'post_prompt': independent_summary(rows, arm, True)}
                          for arm in ('native', 'lean', 'samurai')}
        evidence[seq] = {'prompt_frame': start, 'frames': len(frames),
                         'pre_prompt_present_misses': sum(x['native']['present'] for x in rows if x['frame'] < start),
                         'native_receipt_sha256': digest(native / 'receipt.json'),
                         'lean_receipt_sha256': digest(lean / 'receipt.json'),
                         'samurai_receipt_sha256': digest(sam / 'receipt.json')}
        per_frame[seq] = rows
    lines = ['# Matched automatic-mask MPS comparison', '',
             'Same archived automatic initialization and frame, same input video and checkpoint, and the same fixed inset-excluded seq23 viewport. Post-prompt scores exclude the initialization frame; all later labeled frames are included.', '',
             '| Sequence | Present frames | Present Dice: native / project Lean / SAMURAI | Labeled-empty false alarms: native / project Lean / SAMURAI |',
             '|---|---:|---:|---:|']
    for seq in ('seq11', 'seq20', 'seq23'):
        a, b, c = (summaries[seq][arm]['post_prompt'] for arm in ('native', 'lean', 'samurai'))
        lines.append(f"| {seq} | {a['present_frames']} | {a['mean_present_frame_dice']:.4f} / {b['mean_present_frame_dice']:.4f} / {c['mean_present_frame_dice']:.4f} | {a['false_alarm_frames']}/{a['empty_frames']} / {b['false_alarm_frames']}/{b['empty_frames']} / {c['false_alarm_frames']}/{c['empty_frames']} |")
    lines += ['', 'These sequences were selected and inspected previously; this is exploratory rather than evidence of independent generalization. Historical YOLO scanned frames until its first detection; there is no redetection in these runs. Pre-prompt misses are excluded from the table and included in the all-frame summaries.', '',
              'Project Lean is a modified 16,384-token window with indexed RoPE, multiscale anchor and trained bypass, not original author Lean. SAMURAI retains its author memory-selection thresholds and a float32 MPS compatibility cast. Native SAM2 uses the unmodified pinned architecture with a scoped blocking pointer transfer to avoid an observed MPS temporary-buffer race. Short replay gates do not prove full-video numerical equivalence or validity on other videos.', '',
              'All three systems use MPS with fast math and fallback disabled. Sequence 23 crop was chosen retrospectively; patient independence, detector training overlap, label completeness and actual acquisition timing are not established. Frozen predictions and source-image hashes were verified for every sequence before labels were decoded.', '',
              'The native runner did not bind separate hashes for shared model-builder and prompt helper dependencies in its gate or run receipts. Their supplemental hashes were measured during the full runs, not before them, so they do not independently establish pre-run identity.']
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = out / 'report.md'
    report.write_text('\n'.join(lines) + '\n')
    (out / 'results.json').write_text(json.dumps({'scope': 'Exploratory automatic-mask comparison', 'summaries': summaries,
                                                'evidence': evidence, 'rows': per_frame,
                                                'report_sha256': digest(report), 'scorer_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print(report.read_text())


if __name__ == '__main__':
    main()

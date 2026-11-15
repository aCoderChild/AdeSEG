#!/usr/bin/env python3
"""Audit archived CMSA against current automatic-SAM2 target frame identities.

This is an unequal-input descriptive comparison; it does not rerun inference
or alter any archived output. The historical CMSA receipt did not record source
image hashes, so matching the manifest does not prove historical image bytes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.score_samurai_automatic_priority import score


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    base = a.experiment_root
    cmsa = base/'cmsa_net_polypgen_v1/full_attempt01'
    matched = base/'native_sam2_automatic_priority_v1/matched_three_system_report_attempt01/results.json'
    setup, old_rows, comparison = read(cmsa/'setup.json'), read(cmsa/'frames.json'), read(matched)
    checkpoint = base/'cmsa_net_polypgen_v1/assets/ckpt_epoch_007.pth'
    author = ROOT/'external/CMSA-Net'
    revision = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if (revision != setup['author_revision'] or dirty
            or digest(author/'cmsa_project/scripts/my_test.py') != setup['author_test_sha256']
            or digest(checkpoint) != setup['checkpoint_sha256'] or digest(MANIFEST) != setup['manifest_sha256']
            or setup['runner_sha256'] != '871b7548ad43328e8961d4aff698a0896b661be6502181a2428db63f66b20a55'
            or 'Author normalization retained' not in setup['note']
            or setup['probability_threshold'] != 128 or setup['device'] != 'cpu'
            or comparison['report_sha256'] != digest(matched.parent/'report.md')):
        raise ValueError('Archived CMSA or comparison provenance differs')
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    pre = read(base/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03/manifest.json')
    by_seq = {seq: [r for r in old_rows if r['sequence'] == seq] for seq in ('seq11', 'seq20', 'seq23')}
    rows, summaries, evidence = {}, {}, {}
    data_root = (ROOT/'data').resolve()
    for seq in by_seq:
        frames = videos[seq]
        archived = by_seq[seq]
        current = comparison['rows'][seq]
        native = base/f'native_sam2_automatic_priority_v1/{seq}_full_mps_attempt01'
        receipt = read(native/'receipt.json')
        source_dir = Path(pre['sequences'][seq]['source'])/seq
        metadata = read(source_dir/'frames.json')
        if (digest(source_dir/'frames.json') != pre['sequences'][seq]['frames_metadata_sha256']
                or digest(native/'receipt.json') != comparison['evidence'][seq]['native_receipt_sha256']
                or [r['frame'] for r in archived] != list(range(len(frames)))
                or len(current) != len(frames) or len(receipt['records']) != len(frames)
                or len(metadata) != len(frames)):
            raise ValueError(f'{seq}: source, manifest or frame coverage differs')
        seq_rows = []
        for i, frame in enumerate(frames):
            saved = cmsa/'predictions'/frame.image_path.resolve().relative_to(data_root).with_suffix('.png')
            image_sha, label_sha = digest(frame.image_path), digest(frame.mask_path)
            source_record, reference = receipt['records'][i], current[i]
            if (digest(saved) != archived[i]['prediction_sha256'] or image_sha != source_record['image_sha256']
                    or image_sha != metadata[i]['image_sha256'] or label_sha != metadata[i]['label_sha256']
                    or source_record['source_frame_index'] != frame.frame_index
                    or reference['source_frame_index'] != frame.frame_index or reference['frame'] != i):
                raise ValueError(f'{seq} frame {i}: source identity or frozen prediction differs')
            with Image.open(saved) as image:
                pred = np.asarray(image.convert('L')) >= 128
            truth = load_label_mask(frame.mask_path, [1]) == 1
            metrics = calculate_scores(pred, truth)
            original = archived[i]
            if (metrics['dice'] != original['dice'] or metrics['iou'] != original['iou']
                    or bool(truth.any()) != original['present']):
                raise ValueError(f'{seq} frame {i}: archived CMSA score does not reproduce')
            cmsa_metric = score(pred, truth)
            if cmsa_metric['present'] != reference['native']['present']:
                raise ValueError(f'{seq} frame {i}: ground-truth presence differs')
            seq_rows.append({'frame': i, 'source_frame_index': frame.frame_index,
                             'image_path': str(frame.image_path), 'image_sha256_now': image_sha,
                             'label_sha256_now': label_sha, 'cmsa_prediction_sha256': digest(saved),
                             'post_prompt': reference['post_prompt'],
                             'initialization_frame': reference['initialization_frame'],
                             'cmsa': cmsa_metric})
        post = [row for row in seq_rows if row['post_prompt']]
        positive = [r['cmsa']['dice'] for r in post if r['cmsa']['present']]
        empty = [r['cmsa'] for r in post if not r['cmsa']['present']]
        if (len(positive) != comparison['summaries'][seq]['native']['post_prompt']['present_frames']
                or len(empty) != comparison['summaries'][seq]['native']['post_prompt']['empty_frames']):
            raise ValueError(f'{seq}: scored post-prompt frame sets differ')
        summaries[seq] = {'post_prompt_frames': len(post), 'present_frames': len(positive),
                          'mean_present_frame_dice': float(np.mean(positive)) if positive else None,
                          'empty_frames': len(empty), 'false_alarm_frames': sum(x['false_alarm'] for x in empty)}
        evidence[seq] = {'frames': len(frames), 'prompt_frame': receipt['prompt_frame'],
                         'native_receipt_sha256': digest(native/'receipt.json'),
                         'preflight_source_metadata_sha256': digest(source_dir/'frames.json')}
        rows[seq] = seq_rows
    lines = ['# CMSA-Net saved-output comparability audit', '',
             'Descriptive scores below use the exact post-initialization frame indices and current labels in the recent automatic-mask SAM2 comparison; the initialization frame is excluded. Predictions are the archived released-column-normalization CPU run, thresholded at 128 after its saved output resize.', '',
             '| Sequence | Present frames | CMSA Dice | Labeled-empty false alarms |', '|---|---:|---:|---:|']
    for seq in by_seq:
        s = summaries[seq]
        lines.append(f"| {seq} | {s['present_frames']} | {s['mean_present_frame_dice']:.4f} | {s['false_alarm_frames']}/{s['empty_frames']} |")
    lines += ['', '**This is not an equal-input comparison.** CMSA is an unprompted standalone Res2Net model and has processed the video from frame 0. The SAM2 systems receive an archived automatic mask at frames 2, 13 and 0 for seq11, seq20 and seq23 respectively. CMSA uses a different checkpoint and CPU runtime; the SAM2 runs use MPS.', '',
              'For seq23 CMSA saw the original full frame, including the small inset; the recent SAM2 systems saw a fixed inset-excluded crop and their predictions were restored to full-frame coordinates. Both are scored against full-size labels, but their input fields of view differ. The crop was chosen retrospectively. No crop or alteration was applied to the archived CMSA output in this audit.', '',
              'The released normalization changes the first three image columns rather than applying channel-wise normalization. The saved run follows that released behavior; the source alone does not prove the checkpoint training preprocessing. Its 352×352 resized input, quantized output and threshold differ from SAM2 processing. The author loop uses image references and overlapping clips; frame 0 can depend on frame 1 and is excluded from every table row. Labels are loaded by its data pipeline and for output resizing but are not passed as masks to the model.', '',
              'The CMSA manifest digest and output-prediction hashes match, every saved score and presence flag was reproduced using the current labels, and the current source-image and label hashes match the automatic comparison preflight. The historical CMSA receipt did not save per-image input hashes, so this cannot prove those bytes were unchanged at CMSA inference time. Its historical runner hash differs from the current runner; the archived setup identifies the released preprocessing, but does not preserve a complete runnable source snapshot.', '',
              'These videos were selected and inspected previously. Detector training overlap, patient independence and label completeness are unknown. This audit does not isolate an architectural effect, establish clinical performance, or turn the comparison into an independent evaluation.']
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = out/'report.md'
    report.write_text('\n'.join(lines)+'\n')
    result = {'passed': True, 'scope': 'Descriptive matched target indices, unequal inputs', 'summaries': summaries,
              'evidence': evidence, 'rows': rows, 'cmsa_setup_sha256': digest(cmsa/'setup.json'),
              'cmsa_frames_sha256': digest(cmsa/'frames.json'), 'automatic_results_sha256': digest(matched),
              'auditor_sha256': digest(Path(__file__)), 'report_sha256': digest(report)}
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(report.read_text())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Write a concise report from completed SASVi and SAMRefiner audits."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', required=True, type=Path)
    p.add_argument('--samrefiner', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    cohort = json.loads((args.cohort / 'results.json').read_text())
    analysis = json.loads((args.cohort / 'cohort_analysis.json').read_text())
    failures = json.loads((args.cohort / 'failure_analysis.json').read_text())
    refiner = json.loads((args.samrefiner / 'audit_results.json').read_text())
    low, high = analysis['sequence_bootstrap_interval']
    rlow, rhigh = refiner['sequence_bootstrap_interval']
    lines = ['# SASVi and SAMRefiner on PolypGen', '',
        '## SASVi/CMSA — full CPU cohort', '',
        'Released SASVi video loop with a CMSA-Net binary-mask overseer, a numerical filename-order fix, a three-frame confirmation fix, and small-mask prompt handling. '
        'This is an adaptation; CMSA uses multi-frame clips and SASVi uses lookahead, so it is not causal streaming. '
        'Ground truth was used only for the retrospective evaluation cutoff and scoring.', '',
        f"21 sequences; {sum(v['present'] for v in cohort['per_sequence'].values())} labeled-present and {cohort['empty_targets']} labeled-empty post-cutoff frames. "
        'Present Dice is averaged within sequence and then equally over sequences.', '',
        '| System | Equal-sequence present Dice | False alarms on labeled-empty frames |', '|---|---:|---:|',
        f"| CMSA overseer alone | {cohort['cmsa_equal_sequence_dice']:.5f} | {cohort['cmsa_false_alarms']} / {cohort['empty_targets']} |",
        f"| SASVi with CMSA overseer | {cohort['sasvi_equal_sequence_dice']:.5f} | {cohort['sasvi_false_alarms']} / {cohort['empty_targets']} |", '',
        f"Paired difference: {analysis['paired_mean_difference']:+.5f}, exploratory sequence-bootstrap interval [{low:+.5f}, {high:+.5f}]. "
        f"Wins/losses/ties: {analysis['wins']}/{analysis['losses']}/{analysis['ties']}.", '',
        '| Sequence | Present frames | CMSA Dice | SASVi/CMSA Dice | CMSA false alarms | SASVi false alarms | Empty frames |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in cohort['per_sequence'].items():
        lines.append(f"| {name} | {item['present']} | {item['cmsa_present_dice']:.5f} | {item['sasvi_present_dice']:.5f} | "
                     f"{item['cmsa_false_alarms']} | {item['sasvi_false_alarms']} | {item['empty']} |")
    for name, item in failures.items():
        a, b = item['totals']['cmsa'], item['totals']['sasvi']
        lines += ['', f"{name}: nonempty output on labeled-present frames: CMSA {a['present_nonempty']}/{item['present_targets']}, "
                  f"SASVi/CMSA {b['present_nonempty']}/{item['present_targets']}. "
                  f"Present-frame pooled TP/FP/FN: CMSA {a['tp']}/{a['fp']}/{a['fn']}; "
                  f"SASVi/CMSA {b['tp']}/{b['fp']}/{b['fn']}."]
    lines += ['', '## SAMRefiner — fixed-frame pilot', '',
        f"Released SAMRefiner with original SAM ViT-B and five iterations, using saved YOLO-box SAM2 masks on {refiner['frames']} selected frames "
        f"({refiner['present']} labeled-present, {refiner['empty']} labeled-empty) across 21 sequences. "
        'The empty-mask fallback stays empty; there is no video propagation.', '',
        '| System | Equal-sequence present Dice | False alarms on labeled-empty frames |', '|---|---:|---:|',
        f"| YOLO-box SAM2 mask | {refiner['base_equal_sequence_dice']:.5f} | {refiner['false_alarms_before_after'][0]} / {refiner['empty']} |",
        f"| SAMRefiner output | {refiner['refined_equal_sequence_dice']:.5f} | {refiner['false_alarms_before_after'][1]} / {refiner['empty']} |", '',
        f"Paired difference: {refiner['paired_difference']:+.5f}, exploratory sequence-bootstrap interval [{rlow:+.5f}, {rhigh:+.5f}]; "
        f"excluding seq10: {refiner['seq10_excluded_difference']:+.5f}.", '',
        'These two tables use different frame sets and must not be compared as a head-to-head system ranking. '
        'The detector training split is unknown; the dataset has been inspected; annotation completeness is unverified. '
        'The corrected MPS cohort is provisional because full seq20 replays with identical recorded prompts differ substantially, '
        'including with fixed seeds and deterministic-algorithm settings. CPU seq20 was reproduced exactly in three independent runs. '
        'The root cause of MPS instability is unresolved. Early frame-misordered SASVi runs are invalid and marked as such.', '']
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()

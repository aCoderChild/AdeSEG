#!/usr/bin/env python3
"""Write the human-readable report for an audited merged automatic Lean run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--historical', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.merged / 'completion.json').read_text())
    if not receipt['passed'] or receipt['results_sha256'] != digest(a.merged / 'results.json'):
        raise ValueError('Merged result is incomplete or altered')
    if receipt['historical_comparison_sha256'] != digest(a.historical / 'comparison.json'):
        raise ValueError('Historical comparison changed')
    result = json.loads((a.merged / 'results.json').read_text())
    history = json.loads((a.historical / 'comparison.json').read_text())
    old = {x['sequence']: x for x in history['sequences']}
    d = lambda x: f'{x:.4f}' if x is not None else 'n/a'
    lines = [
        '# Automatic Lean + early-feature prompt + multi-scale anchor', '',
        '**Interpretation:** native SAM2 and Lean were initialized with a ground-truth mask; the new system was initialized automatically by YOLO and its own predicted mask. The comparison confounds initialization, prompting, and the memory change. It cannot isolate the effect of either new component.', '',
        'The new system chooses the first detection at confidence 0.25, uses its box and early-FPN foreground/background points, converts the resulting automatic mask into a Lean initialization, and propagates without re-detection. Before initialization it predicts empty masks. Labels are used for scoring, not prompting.', '',
        'Multi-scale context is a weighted anchor-score blend from the same memory grid at original, 2× pooled and 4× pooled scales. It is inspired by multi-scale context, not CMSA-Net’s learned multi-stage CMA implementation. The Lean token budget remains 1,024 of 4,096 spatial tokens where selection is active.', '',
        f"## Automatic system: all frames", '',
        f"- Frames: {result['all_frames']}; labeled present: {result['all_present_frames']}; labeled empty: {result['all_empty_frames']}.",
        f"- Equal-sequence mean present-frame Dice: {d(result['all_present_dice_equal_sequence'])}.",
        f"- False alarms on labeled-empty frames: {result['all_false_alarms']} / {result['all_empty_frames']} (any predicted foreground pixel).",
        f"- Labeled-present frames before first detection: {result['labeled_present_before_detection']}; sequences with first detection on a labeled-empty frame: {result['first_detection_labeled_empty_sequences']}.",
        f"- No-detection sequences: {', '.join(result['no_detection_sequences']) or 'none'}.",
        f"- Prompt-point status: {result['point_status_counts']}; initialization-mask status: {result['mask_status_counts']}.",
        f"- Active multi-scale memory events: {result['memory_events']} across {result['sequences_with_memory_events']} sequences; audited selection events: {result['selection_events']}.", '',
        '## Historical target-frame IDs', '',
        f"Same {result['historical_matched_frames']} frame IDs ({result['historical_matched_present']} labeled present, {result['historical_matched_empty']} labeled empty); different initialization as stated above. Dice is averaged first over present frames in each sequence, then equally over sequences.", '',
        '| System | Initialization | Present-frame Dice | Empty-frame false alarms |',
        '| --- | --- | ---: | ---: |',
        f"| Native SAM2 (existing) | Ground-truth mask | {d(result['historical_native_dice'])} | {result['historical_native_false_alarms']} / {result['historical_matched_empty']} |",
        f"| Lean-SAM2 (existing) | Ground-truth mask | {d(result['historical_lean_dice'])} | {result['historical_lean_false_alarms']} / {result['historical_matched_empty']} |",
        f"| Combined system (new) | YOLO + automatic mask | {d(result['automatic_combined_historical_dice'])} | {result['automatic_combined_historical_false_alarms']} / {result['historical_matched_empty']} |", '',
        '## Per-sequence results', '',
        'Dice columns refer to the historical frame IDs; the first-detection index is zero-based over the complete sequence.', '',
        '| Sequence | First detection | Detection label | Native Dice | Lean Dice | Combined Dice | Combined false alarms |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for name, row in result['sequences'].items():
        h = old[name]
        index = row['first_detection_index'] if row['first_detection_index'] is not None else 'none'
        label = {True: 'present', False: 'empty', None: 'none'}[row['first_detection_labeled_present']]
        lines.append(f"| {name} | {index} | {label} | {d(h['native_dice'])} | {d(h['lean_dice'])} | {d(row['historical_present_dice'])} | {row['historical_false_alarms']} / {row['historical_empty']} |")
    lines.extend(['', 'The detector checkpoint has PolypGen training metadata; the available metadata does not establish its training/evaluation split or independence from this cohort. These are exploratory results on the selected sequences, not an independent generalization estimate.', '',
                  f"Machine-readable results and per-sequence run locations: `results.json` and `locations.json` in this directory. Historical source: `{a.historical}`. Protocol: `protocols/lean_auto_multiscale_early_prompt_v1.json` in the repository.", ''])
    output = validate_output(a.merged / 'report.md')
    if output.exists():
        raise ValueError('Report already exists')
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit Lean's initial automatic-mask prompt round trip from saved outputs.

No model inference is run. Scores are retrospective and never enter prompts.
"""
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
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def score(mask, truth):
    denominator = int(mask.sum() + truth.sum())
    return 2 * int(np.count_nonzero(mask & truth)) / denominator if denominator else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge = read(args.merged / 'completion.json')
    if not merge['passed'] or digest(args.merged / 'results.json') != merge['results_sha256']:
        raise ValueError('Merged automatic system results changed')
    locations = read(args.merged / 'locations.json')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    results = {}
    for name, location in locations.items():
        run = Path(location)
        folder = run / name
        completion = read(run / 'completion.json')
        audit = read(run / 'independent_audit.json')
        if (not completion['passed'] or not audit['passed'] or
            digest(run / 'records.json') != completion['records_sha256'] or
            audit['records_sha256'] != completion['records_sha256']):
            raise ValueError(f'{name}: source run incomplete')
        record = read(run / 'records.json')[0]
        for key, filename in [('prompt_sha256', 'prompt.json'), ('frames_sha256', 'frames.json')]:
            if digest(folder / filename) != record[key]:
                raise ValueError(f'{name}: source receipt changed')
        prompt = read(folder / 'prompt.json')
        index = prompt['first_detection_index']
        if index is None:
            results[name] = {'first_detection_index': None}
            continue
        rows = read(folder / 'frames.json')
        frame = videos[name][index]
        initial_file = folder / 'prompt_refined.png'
        output_file = folder / f'{index:06d}.png'
        if (digest(initial_file) != prompt['refined_mask_sha256'] or
            digest(output_file) != rows[index]['mask_sha256'] or
            digest(frame.image_path) != rows[index]['image_sha256'] or
            digest(frame.mask_path) != rows[index]['label_sha256']):
            raise ValueError(f'{name}: saved mask or input changed')
        source = np.asarray(Image.open(initial_file)) > 0
        output = np.asarray(Image.open(output_file)) > 0
        truth = load_label_mask(frame.mask_path, [1]) == 1
        if source.shape != output.shape or source.shape != truth.shape:
            raise ValueError(f'{name}: image dimensions differ')
        xor = int(np.count_nonzero(source ^ output))
        union = int(np.count_nonzero(source | output))
        present = bool(truth.any())
        source_dice = score(source, truth) if present else None
        output_dice = score(output, truth) if present else None
        results[name] = {'first_detection_index': index, 'labeled_present': present,
            'prompt_status': prompt['point_status'], 'lean_mask_status': prompt['lean_mask_status'],
            'different_pixels': xor, 'different_fraction_of_image': xor / source.size,
            'mask_pair_iou': int(np.count_nonzero(source & output)) / union if union else 1.,
            'source_area_fraction': float(source.mean()), 'output_area_fraction': float(output.mean()),
            'refined_prompt_dice': source_dice, 'post_mask_prompt_dice': output_dice,
            'dice_change': output_dice - source_dice if present else None}
    changed = [x for x in results.values() if x.get('first_detection_index') is not None]
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'Retrospective comparison of the saved refined prompt with its first Lean propagation output; no model inference or prompt intervention.',
              'merged_results_sha256': merge['results_sha256'], 'sequences': results,
              'mean_image_pixel_difference': float(np.mean([x['different_fraction_of_image'] for x in changed])),
              'mean_mask_pair_iou': float(np.mean([x['mask_pair_iou'] for x in changed]))}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Initial mask-prompt round-trip audit', '', result['scope'], '',
             'These are original-resolution saved binary masks. Label-derived Dice is reported only for labeled-present prompt frames.', '',
             '| Sequence | Prompt frame | Label | Prompt Dice | Propagated Dice | Dice change | Pair IoU | Changed image pixels |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in results.items():
        if row['first_detection_index'] is None:
            lines.append(f'| {name} | none | — | — | — | — | — | — |')
            continue
        fmt = lambda x: '—' if x is None else f'{x:.5f}'
        lines.append(f"| {name} | {row['first_detection_index']} | {'present' if row['labeled_present'] else 'empty'} | "
                     f"{fmt(row['refined_prompt_dice'])} | {fmt(row['post_mask_prompt_dice'])} | {fmt(row['dice_change'])} | "
                     f"{row['mask_pair_iou']:.5f} | {100*row['different_fraction_of_image']:.3f}% |")
    lines.extend(['', 'A small initial change does not prove later propagation is correct. The conversion re-encodes an automatic mask; it does not provide ground-truth localization. Empty annotations may be incomplete.', ''])
    (out / 'report.md').write_text('\n'.join(lines))
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'),
        'source_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__':
    main()

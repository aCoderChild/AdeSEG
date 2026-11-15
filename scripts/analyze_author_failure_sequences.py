#!/usr/bin/env python3
"""Read-only, frame-matched analysis of completed PolypGen author baselines."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402

EXPERIMENT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
DIRS = {
    'native': '08_region_readout_polypgen_v1/serial_release_v1/sam2_full',
    'lean': 'lean_polypgen_full_v1/attempt03/full',
    'region': '08_region_readout_polypgen_v1/serial_release_v1/sam2_region_full',
    'shuffled': '08_region_readout_polypgen_v1/serial_release_v1/sam2_shuffled_full',
    'samurai': '12_author_region_attention_polypgen_cpu_v1/samurai_full',
    'sam2long': '12_author_region_attention_polypgen_cpu_v1/sam2long_full',
}
KEYS = ('checkpoint_sha256', 'manifest_sha256', 'device', 'torch', 'precision',
        'config', 'apply_postprocessing', 'memory_stride')
SEQUENCES = [f'seq{i}' for i in range(2, 24) if i != 7]


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    present = [r for r in rows if r['present']]
    empty = [r for r in rows if not r['present']]
    # For a present GT, zero Dice plus precision 1 means the predicted mask
    # is empty under evaluation.segmentation._ratio's zero-denominator rule.
    empty_pred = sum(r['scores']['dice'] == 0 and r['scores']['precision'] == 1 for r in present)
    disjoint = sum(r['scores']['dice'] == 0 and r['scores']['precision'] == 0 for r in present)
    return {'present': len(present), 'empty': len(empty),
            'dice': float(np.mean([r['scores']['dice'] for r in present])),
            'precision': float(np.mean([r['scores']['precision'] for r in present])),
            'recall': float(np.mean([r['scores']['recall'] for r in present])),
            'empty_predictions': empty_pred, 'disjoint_predictions': disjoint,
            'false_alarms': sum(r['absent_fp'] is True for r in empty)}


def group_ranges(indices):
    result = []
    for i in indices:
        if result and i == result[-1][-1] + 1:
            result[-1].append(i)
        else:
            result.append([i])
    return [[a[0], a[-1], len(a)] for a in result]


def panel(seq, frame, records, inputs, output):
    """Show input and GT followed by same-frame author masks."""
    source = inputs[frame]
    image = Image.open(source['image']).convert('RGB')
    gt = load_label_mask(Path(source['mask']), [1]) == 1
    thumb = (320, 240)
    strips = []
    for name in ('GT', *DIRS):
        mask = gt if name == 'GT' else np.asarray(Image.open(EXPERIMENT / DIRS[name] / seq / f"frame_{records[name][frame]['frame']:06d}.png")) > 0
        if mask.shape != gt.shape:
            raise ValueError('Original dimensions differ')
        base = np.asarray(image).copy()
        colored = base.copy()
        colored[mask] = (0.4 * colored[mask] + 0.6 * np.array([255, 50, 30])).astype(np.uint8)
        out = Image.fromarray(colored).resize(thumb)
        strips.append((name, out))
    canvas = Image.new('RGB', (thumb[0] * len(strips), thumb[1] + 28), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, tile) in enumerate(strips):
        draw.text((i * thumb[0] + 4, 4), name, fill='black')
        canvas.paste(tile, (i * thumb[0], 28))
    canvas.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    setups = {k: read(EXPERIMENT / v / 'setup.json') for k, v in DIRS.items()}
    for name, setup in setups.items():
        for key in KEYS:
            if key == 'config' and name == 'samurai':
                continue  # SAMURAI's author config enables its own method.
            if setup[key] != setups['native'][key]:
                raise ValueError(f'{name}: mismatch in {key}')
        if not read(EXPERIMENT / DIRS[name] / 'delivery.json')['completed']:
            raise ValueError(f'{name}: incomplete run')
    lean_audit = read(EXPERIMENT / 'lean_polypgen_full_v1/attempt03/independent_mask_audit.json')
    if not lean_audit['passed']:
        raise ValueError('Lean/native mask audit not passed')
    cmsa = read(EXPERIMENT / 'cmsa_net_polypgen_v1/full_attempt01/results.json')['per_sequence']
    out = {'limitations': ['Retrospective; PolypGen sequences previously inspected.',
           'SAM2Long final-path selection uses future sequence information.',
           'SAMURAI starts at the first prompt; the other models stage the full sequence.',
           'CMSA is unprompted and uses a different model/checkpoint; descriptive only.',
           'GT-empty labels are treated as supplied; no assertion of annotation completeness.',
           'Frame records cannot identify a causal memory token or prove a failure cause.'],
           'source_sha256': {k: digest(EXPERIMENT / v / 'setup.json') for k, v in DIRS.items()},
           'sequences': {}}
    cache = {}
    for seq in SEQUENCES:
        rows, inputs = {}, {}
        for name, relative in DIRS.items():
            folder = EXPERIMENT / relative / seq
            records = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
            if len({r['source_frame_index'] for r in records}) != len(records):
                raise ValueError(f'duplicate targets {name} {seq}')
            rows[name] = {r['source_frame_index']: r for r in records}
            inputs[name] = {r['frame']: r for r in read(folder / 'inputs.json')}
        native = rows['native']
        for name in DIRS:
            if list(rows[name]) != list(native) or any(rows[name][i]['present'] != native[i]['present'] for i in native):
                raise ValueError(f'target mismatch {name} {seq}')
            for i in native:
                if (inputs[name][i]['image_sha256'], inputs[name][i]['mask_sha256']) != (inputs['native'][i]['image_sha256'], inputs['native'][i]['mask_sha256']):
                    raise ValueError(f'input hash mismatch {name} {seq} {i}')
        stats = {name: metrics(list(rows[name].values())) for name in DIRS}
        if len(native) != cmsa[seq]['matched_targets'] or stats['native']['present'] != cmsa[seq]['present_targets']:
            raise ValueError(f'CMSA counts mismatch {seq}')
        result = {'methods': stats, 'cmsa_dice': cmsa[seq]['mean_present_dice'],
                  'cmsa_false_alarms': cmsa[seq]['false_alarms'],
                  'delta': {k: stats[k]['dice'] - stats['native']['dice'] for k in DIRS if k != 'native'}}
        present = [i for i in native if native[i]['present']]
        for name in ('lean', 'region', 'shuffled', 'samurai', 'sam2long'):
            ranked = sorted(present, key=lambda i: rows[name][i]['scores']['dice'] - native[i]['scores']['dice'])
            result[f'{name}_worst_frames'] = [{'frame': i, 'delta': rows[name][i]['scores']['dice'] - native[i]['scores']['dice'],
                'native_dice': native[i]['scores']['dice'], 'method_dice': rows[name][i]['scores']['dice']} for i in ranked[:5]]
        out['sequences'][seq] = result
        if seq in ('seq6', 'seq11', 'seq17', 'seq20', 'seq23'):
            gt_areas = []
            for i in present:
                gt_areas.append(float((load_label_mask(Path(inputs['native'][i]['mask']), [1]) == 1).mean()))
            prompt = min(i for i in inputs['native'] if (load_label_mask(Path(inputs['native'][i]['mask']), [1]) == 1).any())
            prompt_area = float((load_label_mask(Path(inputs['native'][prompt]['mask']), [1]) == 1).mean())
            result['gt_area'] = {'prompt_frame': prompt, 'prompt_fraction': prompt_area,
                                  'present_median_fraction': float(np.median(gt_areas)),
                                  'present_min_fraction': min(gt_areas), 'present_max_fraction': max(gt_areas)}
            for name in DIRS:
                ids = [i for i in present if rows[name][i]['scores']['dice'] == 0]
                result['methods'][name]['zero_dice_ranges'] = group_ranges(ids)
            cache[seq] = (rows, inputs['native'])
    for seq in ('seq6', 'seq17', 'seq23'):
        frame = out['sequences'][seq]['lean_worst_frames'][0]['frame']
        panel(seq, frame, *cache[seq], output / f'{seq}_frame{frame}.png')
    for seq in ('seq20', 'seq11'):
        rows, inputs = cache[seq]
        present = [i for i in rows['native'] if rows['native'][i]['present']]
        frame = present[0] if seq == 'seq20' else min(present, key=lambda i: rows['native'][i]['scores']['precision'])
        panel(seq, frame, rows, inputs, output / f'{seq}_frame{frame}.png')
    (output / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print('Wrote', output, flush=True)


if __name__ == '__main__':
    main()

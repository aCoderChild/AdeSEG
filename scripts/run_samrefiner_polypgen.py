#!/usr/bin/env python3
"""Apply released SAMRefiner to saved, GT-free SAM2 masks on matched frames."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402

AUTHOR = ROOT / 'external/SAMRefiner'
REVISION = '4bb7f95738c7a2e805f189210fb09e02a2197557'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', action='append')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(['git', '-C', str(AUTHOR), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise RuntimeError('Unexpected SAMRefiner revision')
    subprocess.run(['git', '-C', str(AUTHOR), 'diff', '--quiet', 'HEAD', '--', 'sam_refiner.py', 'utils.py', 'segment-anything'], check=True)
    audit = json.loads((args.input_run / 'independent_audit.json').read_text())
    if not audit['passed'] or digest(args.input_run / 'records.json') != audit['records_sha256']:
        raise ValueError('Input run not independently audited')
    rows = json.loads((args.input_run / 'records.json').read_text())
    if args.sequence:
        rows = [r for r in rows if r['sequence'] in args.sequence]
        if set(args.sequence) != {r['sequence'] for r in rows}:
            raise ValueError('Unknown or absent sequence')
    sys.path.insert(0, str(AUTHOR / 'segment-anything'))
    sys.path.insert(0, str(AUTHOR))
    from segment_anything import sam_model_registry
    from sam_refiner import sam_refiner
    import FastGeodis
    torch.set_num_threads(2)
    sam = sam_model_registry['vit_b'](checkpoint=str(args.checkpoint)).to('cpu').eval()
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    args.output.mkdir(parents=True)
    setup = {'method': 'released SAMRefiner, vit_b, default five iterations', 'revision': REVISION,
             'samrefiner_sha256': digest(AUTHOR / 'sam_refiner.py'), 'utils_sha256': digest(AUTHOR / 'utils.py'),
             'checkpoint_sha256': digest(args.checkpoint), 'input_records_sha256': audit['records_sha256'],
             'manifest_sha256': digest(MANIFEST), 'runner_sha256': digest(Path(__file__)),
             'device': 'cpu', 'torch_threads': 2, 'torch': torch.__version__,
             'input': 'saved YOLO-box SAM2 masks; empty input remains empty', 'ground_truth': 'evaluation only'}
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    outputs = []
    for row in rows:
        name, idx = row['sequence'], row['index']
        frame = data[name][idx]
        if digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['target_mask_sha256']:
            raise ValueError(f'Input frame changed: {name}/{idx}')
        source = args.input_run / name / f'{idx:06d}_yolo_box.png'
        if digest(source) != row['conditions']['yolo_box']['mask_sha256']:
            raise ValueError(f'Input mask changed: {name}/{idx}')
        coarse = np.asarray(Image.open(source)) > 0
        if coarse.any():
            refined, _, _ = sam_refiner(str(frame.image_path), [coarse.astype(np.uint8)], sam)
            pred = refined[0].astype(bool)
            fallback = False
        else:
            pred = np.zeros_like(coarse)
            fallback = True
        if pred.shape != coarse.shape:
            raise ValueError('Output shape differs')
        dst = args.output / name / f'{idx:06d}.png'
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8) * 255).save(dst)
        truth = load_label_mask(frame.mask_path, [1]) == 1
        denominator = int(pred.sum() + truth.sum())
        score = 2 * int(np.count_nonzero(pred & truth)) / denominator if denominator else 1.0
        outputs.append({'sequence': name, 'index': idx, 'present': bool(truth.any()), 'dice': score,
                        'false_alarm': bool(pred.any()) if not truth.any() else None,
                        'empty_input_fallback': fallback, 'input_sha256': digest(source), 'mask_sha256': digest(dst)})
        print(f'{name}/{idx}: input_empty={fallback}, dice={score:.5f}', flush=True)
    (args.output / 'records.json').write_text(json.dumps(outputs, indent=2) + '\n')
    (args.output / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(args.output / 'records.json'),
                                                              'frames': len(outputs)}, indent=2) + '\n')


if __name__ == '__main__':
    main()

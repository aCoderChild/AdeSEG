#!/usr/bin/env python3
"""Post-hoc prompt localization audit; labels never enter the model."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_prompt_feature_refinement import box_iou, gt_box  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.merged / 'completion.json').read_text())
    if not receipt['passed'] or digest(a.merged / 'results.json') != receipt['results_sha256']:
        raise ValueError('Merged results changed')
    locations = json.loads((a.merged / 'locations.json').read_text())
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    summary = {}
    for name, path in locations.items():
        run = Path(path); folder = run / name
        done = json.loads((run / 'completion.json').read_text())
        audit = json.loads((run / 'independent_audit.json').read_text())
        if not done['passed'] or not audit['passed'] or digest(run / 'records.json') != done['records_sha256'] or audit['records_sha256'] != done['records_sha256']:
            raise ValueError(f'Invalid run: {name}')
        record = json.loads((run / 'records.json').read_text())[0]
        if digest(folder / 'prompt.json') != record['prompt_sha256'] or digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError('Prompt or frame receipt changed')
        prompt = json.loads((folder / 'prompt.json').read_text())
        index = prompt['first_detection_index']
        if index is None:
            summary[name] = {'detection': False}; continue
        rows = json.loads((folder / 'frames.json').read_text())
        frame = videos[name][index]
        if digest(frame.mask_path) != rows[index]['label_sha256']:
            raise ValueError('Label changed')
        gt = load_label_mask(frame.mask_path, [1]) == 1
        box = prompt['box']; x0, y0, x1, y1 = box
        h, w = gt.shape
        crop = gt[max(0, int(math.floor(y0))):min(h, int(math.ceil(y1))), max(0, int(math.floor(x0))):min(w, int(math.ceil(x1)))]
        entry = {'detection': True, 'frame': index, 'present': bool(gt.any()), 'confidence': prompt['confidence'],
                 'box_fraction_of_image': (x1-x0)*(y1-y0)/(h*w),
                 'gt_covered_by_box': float(crop.sum()/gt.sum()) if gt.any() else None,
                 'box_vs_gt_box_iou': box_iou(box, gt_box(gt)) if gt.any() else None}
        points = prompt['points']
        entry['positive_point_in_labeled_foreground'] = None if not gt.any() or points is None else bool(gt[min(h-1,int(points[0][1])), min(w-1,int(points[0][0]))])
        entry['negative_point_in_labeled_foreground'] = None if not gt.any() or points is None else bool(gt[min(h-1,int(points[1][1])), min(w-1,int(points[1][0]))])
        for typ in ('provisional', 'refined'):
            file = folder / f'prompt_{typ}.png'
            if digest(file) != prompt[f'{typ}_mask_sha256']:
                raise ValueError('Prompt mask changed')
            mask = np.asarray(Image.open(file)) > 0
            denom = int(mask.sum()+gt.sum())
            entry[f'{typ}_dice'] = 2*int((mask & gt).sum())/denom if gt.any() else None
            entry[f'{typ}_fraction_of_image'] = float(mask.mean())
            entry[f'{typ}_pixel_precision'] = float((mask & gt).sum()/mask.sum()) if gt.any() and mask.any() else None
            entry[f'{typ}_pixel_recall'] = float((mask & gt).sum()/gt.sum()) if gt.any() else None
        summary[name] = entry
    out = validate_output(a.output)
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'scope': 'Retrospective prompt-frame evaluation only. Labeled-empty detections may reflect incomplete annotations. Point membership uses one image pixel and is not independent validation.', 'sequences': summary}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()

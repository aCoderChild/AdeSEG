#!/usr/bin/env python3
"""Render original first YOLO boxes on images without labels or new inference."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.merged / 'completion.json').read_text())
    if not receipt['passed'] or digest(a.merged / 'results.json') != receipt['results_sha256']:
        raise ValueError('Merged reference differs')
    locations = json.loads((a.merged / 'locations.json').read_text())
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    names = sorted(locations, key=lambda x: int(x[3:]))
    fig, axes = plt.subplots(7, 3, figsize=(12, 18), constrained_layout=True)
    examples = []
    for ax, name in zip(axes.flat, names):
        run = Path(locations[name]); folder = run / name
        done = json.loads((run / 'completion.json').read_text()); records = json.loads((run / 'records.json').read_text())
        if not done['passed'] or digest(run / 'records.json') != done['records_sha256']:
            raise ValueError('Source run differs')
        record = records[0]
        if digest(folder / 'prompt.json') != record['prompt_sha256'] or digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError('Prompt or frames changed')
        prompt = json.loads((folder / 'prompt.json').read_text())
        index = prompt['first_detection_index']
        if index is None:
            ax.set_title(f'{name}: no detection'); ax.axis('off'); continue
        row = json.loads((folder / 'frames.json').read_text())[index]
        frame = videos[name][index]
        if digest(frame.image_path) != row['image_sha256']:
            raise ValueError('Image changed')
        image = np.asarray(Image.open(frame.image_path).convert('RGB'))
        x0, y0, x1, y1 = prompt['box']
        ax.imshow(image)
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='cyan', linewidth=2))
        ax.set_title(f"{name} · frame {index} · confidence {prompt['confidence']:.2f}", fontsize=10)
        ax.axis('off')
        examples.append({'sequence': name, 'index': index, 'image_sha256': row['image_sha256'],
                         'box': prompt['box'], 'confidence': prompt['confidence']})
    fig.suptitle('First automatic YOLO box in each sequence (no labels shown)', fontsize=14)
    image_file = out / 'contact_sheet.png'
    fig.savefig(image_file, dpi=140); plt.close(fig)
    (out / 'records.json').write_text(json.dumps({'examples': examples, 'figure_sha256': digest(image_file)}, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(out / 'records.json'),
                                                    'source_sha256': digest(Path(__file__))}, indent=2) + '\n')


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Compare image and saved FPN contrast across an annotated diagonal and neighbors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output

def read(p): return json.loads(p.read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    observation = read(a.observed / 'contrast_observation.json')
    feature_path = a.observed / 'contrast/000002.npz'
    capture = next(x for x in observation['captures'] if x['frame'] == 2)
    if not observation['passed'] or observation['sequence'] != 'seq11' or digest(feature_path) != capture['map_sha256']:
        raise ValueError('Saved feature capture is invalid')
    frames = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'), **load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11']
    frame = frames[2]
    image = np.asarray(Image.open(frame.image_path).convert('RGB'), np.float32)
    if image.shape != (576, 720, 3): raise ValueError('Image geometry changed')
    maps = np.load(feature_path)
    contrast = maps['level0_contrast']
    if contrast.shape != (256, 256): raise ValueError('Unexpected FPN geometry')
    # Endpoints are the two vertices of the straight label edge, verified by
    # contour inspection; ground truth is used only to locate this diagnostic.
    start, end = np.array([374., 17.]), np.array([669., 402.])
    direction = end - start
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    positions = start + np.linspace(.1, .9, 160)[:, None] * direction
    rows = []
    for offset in (-30, -15, 0, 15, 30):
        def sample(array, displacement, resize=False):
            coords = positions + (offset + displacement) * normal
            if resize:
                coords = (coords + .5) * np.array([array.shape[1]/720, array.shape[0]/576]) - .5
            return map_coordinates(array, [coords[:,1], coords[:,0]], order=1, mode='nearest')
        delta = np.stack([sample(image[:,:,c], 5) - sample(image[:,:,c], -5) for c in range(3)], axis=1)
        gap = sample(contrast, 5, True) - sample(contrast, -5, True)
        rows.append({'offset_pixels': offset, 'mean_rgb_distance_0_255': float(np.linalg.norm(delta, axis=1).mean()),
                     'mean_absolute_fpn0_contrast_gap': float(np.abs(gap).mean())})
    result = {'sequence':'seq11', 'frame':2, 'image_sha256':digest(frame.image_path), 'label_sha256':digest(frame.mask_path),
              'observation_sha256':digest(a.observed/'contrast_observation.json'), 'feature_sha256':digest(feature_path),
              'source_sha256':digest(Path(__file__)), 'line_endpoints_xy':[start.tolist(),end.tolist()],
              'offsets_pixels':[-30,-15,0,15,30], 'sample_count':160, 'cross_line_distance_pixels':10,
              'notes':'Exploratory fixed lines; FPN result is one saved foreground-minus-background prototype score, not the full feature tensor.', 'results':rows}
    out = validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()

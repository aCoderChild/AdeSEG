#!/usr/bin/env python3
"""Inspect existing YOLO on fixed crops of saved first-prompt images.

This is an image-only diagnostic: no ground truth, no new thresholds, no video
propagation, and no rule choosing a crop for deployment.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_prompt_feature_refinement import detect_box  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--detector', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    merged = read(a.merged / 'completion.json')
    if not merged['passed'] or digest(a.merged / 'results.json') != merged['results_sha256']:
        raise ValueError('Merged reference changed')
    locations = read(a.merged / 'locations.json')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    config = out / 'ultralytics'; config.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(config); os.environ['WANDB_MODE'] = 'disabled'
    from ultralytics import YOLO
    torch.set_num_threads(2)
    detector = YOLO(str(a.detector))
    cases = {}
    for name in ('seq8', 'seq11', 'seq20', 'seq23'):
        run = Path(locations[name]); folder = run / name
        setup = read(run / 'setup.json'); done = read(run / 'completion.json'); audit = read(run / 'independent_audit.json')
        if (digest(a.detector) != setup['signature']['detector_sha256'] or not done['passed'] or not audit['passed']
            or digest(run / 'records.json') != done['records_sha256'] or audit['records_sha256'] != done['records_sha256']):
            raise ValueError('Detector or source receipt differs')
        record = read(run / 'records.json')[0]
        if digest(folder / 'prompt.json') != record['prompt_sha256'] or digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError('Prompt or frame metadata changed')
        prompt = read(folder / 'prompt.json'); index = prompt['first_detection_index']
        rows = read(folder / 'frames.json'); frame = videos[name][index]
        if digest(frame.image_path) != rows[index]['image_sha256']:
            raise ValueError('Source image changed')
        image = np.asarray(Image.open(frame.image_path).convert('RGB')); height, width = image.shape[:2]
        # Fixed image-space crops, same YOLO settings; no per-case adaptation.
        options = {'full': 0, 'right_two_thirds': width // 3, 'right_half': width // 2}
        predictions = {}
        for key, x0 in options.items():
            box, confidence = detect_box(detector, image[:, x0:])
            if box is not None:
                box = [box[0] + x0, box[1], box[2] + x0, box[3]]
            predictions[key] = {'crop_xyxy': [x0, 0, width, height], 'box_full_image': box, 'confidence': confidence}
        full = predictions['full']
        if (full['box_full_image'] is None or not np.allclose(full['box_full_image'], prompt['box'], rtol=0, atol=1e-4)
            or not np.isclose(full['confidence'], prompt['confidence'], rtol=0, atol=1e-6)):
            raise ValueError(f'{name}: full-image YOLO does not replay')
        cases[name] = {'index': index, 'image_sha256': rows[index]['image_sha256'], 'predictions': predictions}
        print(name, json.dumps(predictions), flush=True)
    payload = {'scope': 'Retrospective, image-only stress test on selected first-prompt frames. Fixed crops were chosen after inspection; this is not a full-video method or a validation of automatic viewport selection.',
               'checkpoint_sha256': digest(a.detector), 'same_full_image_yolo_replayed': True, 'cases': cases}
    (out / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'),
                                                    'source_sha256': digest(Path(__file__))}, indent=2) + '\n')


if __name__ == '__main__': main()

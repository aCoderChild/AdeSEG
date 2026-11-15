#!/usr/bin/env python3
"""Automatic Lean run on the fixed, visually inspected seq23 main viewport."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT, crop_viewport, restore_mask, stage_cropped_frames
from modeling.early_feature_prompt import early_feature_points
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity
from scripts.run_prompt_feature_refinement import detect_box

def main():
    p = argparse.ArgumentParser()
    for key in ('checkpoint', 'detector', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--engineering-gate', action='store_true')
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    a = p.parse_args()
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(out / 'ultralytics')
    os.environ['WANDB_MODE'] = 'disabled'
    from ultralytics import YOLO
    detector = YOLO(str(a.detector))
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    frames = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}['seq23']
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    setup = {'sequence': 'seq23', 'viewport_xyxy': list(VIEWPORT), 'source_image_size': [1280, 720],
             'policy': 'fixed sequence-specific main viewport; first chronological detector box; no later detection',
             'checkpoint_sha256': digest(a.checkpoint), 'detector_sha256': digest(a.detector),
             'manifest_sha256': digest(manifest), 'source_sha256': digest(Path(__file__)),
             'preprocess_sha256': digest(ROOT / 'modeling/seq23_viewport.py'), 'lean_revision': revision,
             'memory_module_sha256': digest(ROOT / 'modeling/lean_multiscale_anchor.py'),
             'rope_module_sha256': digest(ROOT / 'modeling/lean_indexed_spatial_rope.py'),
             'early_prompt_sha256': digest(ROOT / 'modeling/early_feature_prompt.py')}
    if not a.engineering_gate:
        if a.gate_a is None or a.gate_b is None: raise ValueError('Two engineering gates are required')
        gates = []
        for path in (a.gate_a, a.gate_b):
            gate = json.loads((path / 'gate.json').read_text())
            if gate['setup'] != setup or any(digest(path / name) != value for name, value in gate['files'].items()):
                raise ValueError('Engineering gate changed or uses different sources')
            gates.append(gate)
        if gates[0] != gates[1]: raise ValueError('Engineering gates differ')
    (out / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    first = box = confidence = None
    # Initial search stops permanently on the first detection. No label is read here.
    for i, frame in enumerate(frames):
        with Image.open(frame.image_path) as image:
            pixels = crop_viewport(np.asarray(image.convert('RGB')))
        box, confidence = detect_box(detector, pixels)
        if box is not None:
            first = i
            break
    prompt = {'first_detection_index': first, 'box_viewport_xyxy': box, 'confidence': confidence,
              'points_viewport': None, 'point_status': 'no_detection'}
    events, selections, positions = [], [], []
    if first is not None:
        model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
        before = parameter_fingerprints(model)
        events = install_multiscale_anchor(model, .5, selection_events=selections)
        with tempfile.TemporaryDirectory(prefix='cropped_frames_', dir=out) as temp:
            stage_cropped_frames(frames, temp)
            state = model.init_state(video_path=temp, offload_video_to_cpu=True, offload_state_to_cpu=False)
            if state['num_frames'] != len(frames) or (state['video_width'], state['video_height']) != (834, 720):
                raise ValueError('Video loader geometry changed')
            try:
                with torch.inference_mode(), indexed_spatial_rope(model, positions):
                    _, ids, logits = model.add_new_points_or_box(state, frame_idx=first, obj_id=1, box=np.asarray(box, np.float32))
                    if ids != [1]: raise ValueError('Unexpected object id')
                    provisional = logits[0, 0].cpu().numpy() > 0
                    _, backbone, _, _, _ = model._get_image_feature(state, first, batch_size=1)
                    points, status = early_feature_points(backbone, provisional, np.asarray(box, np.float32), seed=20261114)
                    prompt['point_status'] = status
                    if points is not None:
                        prompt['points_viewport'] = points['feature_points']
                        _, ids, logits = model.add_new_points_or_box(state, frame_idx=first, obj_id=1,
                            points=np.asarray(points['feature_points'], np.float32), labels=np.array([1, 0], np.int32), clear_old_points=False)
                    initial = logits[0, 0].cpu().numpy() > 0
                    Image.fromarray(restore_mask(initial).astype(np.uint8) * 255).save(out / 'prompt.png')
                    if initial.any(): model.add_new_mask(state, frame_idx=first, obj_id=1, mask=initial)
                    count = 0
                    for index, ids, logits in model.propagate_in_video(state, start_frame_idx=first, reverse=False):
                        if index != first + count or ids != [1]: raise ValueError('Video order changed')
                        mask = restore_mask(logits[0, 0].cpu().numpy() > 0)
                        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f'{index:06d}.png')
                        count += 1
                        if a.engineering_gate and count == 3: break
                    expected = min(3, len(frames) - first) if a.engineering_gate else len(frames) - first
                    if count != expected:
                        raise ValueError('Incomplete propagation')
            finally:
                model.reset_state(state)
        assert_parameter_integrity(model, before, out, 'after')
    for i in range(first if first is not None else len(frames)):
        Image.fromarray(np.zeros((720, 1280), np.uint8)).save(out / f'{i:06d}.png')
    (out / 'prompt.json').write_text(json.dumps(prompt, indent=2) + '\n')
    (out / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
    (out / 'positions.json').write_text(json.dumps(positions, indent=2) + '\n')
    (out / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    if a.engineering_gate:
        if first is None or not positions or not selections: raise ValueError('Gate did not exercise propagation')
        names = ['prompt.json', 'prompt.png', 'positions.json', 'selections.json', 'events.json']
        names += [f'{i:06d}.png' for i in range(first + min(3, len(frames) - first))]
        result = {'setup': setup, 'files': {name: digest(out / name) for name in names}}
        (out / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'gate_passed': True, 'first_detection': first}, indent=2))
        return
    rows = []
    # Read labels only after all video predictions have been saved.
    for i, frame in enumerate(frames):
        truth = load_label_mask(frame.mask_path, [1]) == 1
        pred = np.asarray(Image.open(out / f'{i:06d}.png')) > 0
        if truth.shape != pred.shape: raise ValueError('Mask shape changed')
        denom = int(truth.sum() + pred.sum())
        rows.append({'index': i, 'present': bool(truth.any()), 'dice': 2 * int((truth & pred).sum()) / denom if denom else 1.,
                     'false_alarm': bool(pred.any()) if not truth.any() else None,
                     'truth_pixels_outside_viewport': int(truth[:, :VIEWPORT[0]].sum()),
                     'image_sha256': digest(frame.image_path), 'label_sha256': digest(frame.mask_path),
                     'mask_sha256': digest(out / f'{i:06d}.png')})
    (out / 'frames.json').write_text(json.dumps(rows, indent=2) + '\n')
    result = {'passed': True, 'frames_sha256': digest(out / 'frames.json'), 'prompt_sha256': digest(out / 'prompt.json'),
              'positions_sha256': digest(out / 'positions.json'), 'selections_sha256': digest(out / 'selections.json'),
              'setup_sha256': digest(out / 'setup.json')}
    (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()

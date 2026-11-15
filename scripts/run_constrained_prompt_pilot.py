#!/usr/bin/env python3
"""Paired first-prompt diagnostic; no labels read before all masks are saved."""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest, stage_frames  # noqa: E402
from modeling.constrained_prompt_feature_refinement import constrained_early_feature_points  # noqa: E402
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'merged', 'gate-a', 'gate-b', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    gates = []
    for path in (a.gate_a, a.gate_b):
        done, result = read(path / 'completion.json'), read(path / 'results.json')
        if not done['passed'] or digest(path / 'results.json') != done['results_sha256']:
            raise ValueError('Engineering gate incomplete')
        if (result['sequence'] != 'seq2' or result['checkpoint_sha256'] != digest(a.checkpoint)
            or result['model_source_sha256'] != digest(ROOT / 'modeling/constrained_prompt_feature_refinement.py')
            or result['arms']['original']['mask_sha256'] == result['arms']['constrained']['mask_sha256']):
            raise ValueError('Gate does not exercise the current changed selection')
        gates.append(result)
    if gates[0] != gates[1]: raise ValueError('Independent gates differ')
    completed = read(a.merged / 'completion.json')
    if not completed['passed'] or digest(a.merged / 'results.json') != completed['results_sha256']:
        raise ValueError('Original cohort not complete')
    locations = read(a.merged / 'locations.json')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'), **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    repo, revision = source_identity('lean')
    setup = {'scope': 'First automatic prompt frame only; paired same box, not video propagation',
             'sequences': list(locations), 'revision': revision, 'checkpoint_sha256': digest(a.checkpoint),
             'reference_results_sha256': completed['results_sha256'],
             'gate_results_sha256': [digest(x / 'results.json') for x in (a.gate_a, a.gate_b)],
             'runner_sha256': digest(Path(__file__)), 'protocol_sha256': digest(ROOT / 'protocols/constrained_early_prompt_v1.json'),
             'method_sha256': digest(ROOT / 'modeling/constrained_prompt_feature_refinement.py'),
             'original_sha256': digest(ROOT / 'modeling/early_feature_prompt.py'),
             'manifest_sha256': digest(ROOT / 'data/polypgen_sequence.jsonl'),
             'runtime': 'CPU float32, 2 torch threads'}
    (out / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    torch.set_num_threads(2)
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
    before = parameter_fingerprints(model)
    records = []
    for name, location in locations.items():
        source = Path(location); folder = source / name
        old_done, audit = read(source / 'completion.json'), read(source / 'independent_audit.json')
        if not old_done['passed'] or not audit['passed'] or digest(source / 'records.json') != old_done['records_sha256'] or audit['records_sha256'] != old_done['records_sha256']:
            raise ValueError(f'{name}: source incomplete')
        old_record = read(source / 'records.json')[0]
        if digest(folder / 'prompt.json') != old_record['prompt_sha256'] or digest(folder / 'frames.json') != old_record['frames_sha256']:
            raise ValueError(f'{name}: source prompt changed')
        old_prompt = read(folder / 'prompt.json'); index = old_prompt['first_detection_index']
        if index is None: raise ValueError(f'{name}: no prior detection')
        frames = videos[name]; frame_rows = read(folder / 'frames.json')
        if digest(frames[index].image_path) != frame_rows[index]['image_sha256']:
            raise ValueError(f'{name}: image changed')
        target = out / name; target.mkdir()
        with tempfile.TemporaryDirectory(dir=target, prefix='staged_') as temp:
            stage_frames(frames, Path(temp))
            state = model.init_state(video_path=temp, offload_video_to_cpu=True, offload_state_to_cpu=False)
            try:
                with torch.inference_mode():
                    box = np.asarray(old_prompt['box'], np.float32)
                    _, ids, logits = model.add_new_points_or_box(state, frame_idx=index, obj_id=1, box=box)
                    if ids != [1]: raise ValueError('Unexpected object')
                    provisional = logits[0, 0].detach().cpu().numpy() > 0
                    if digest(folder / 'prompt_provisional.png') != old_prompt['provisional_mask_sha256'] or not np.array_equal(provisional, np.asarray(Image.open(folder / 'prompt_provisional.png')) > 0):
                        raise ValueError(f'{name}: provisional mask does not replay')
                    _, backbone, _, _, _ = model._get_image_feature(state, index, batch_size=1)
                    old_points, old_status = early_feature_points(backbone, provisional, box, seed=20261114)
                    if old_status != old_prompt['point_status'] or (old_points is not None and old_points['feature_points'] != old_prompt['points']):
                        raise ValueError(f'{name}: original points do not replay')
                    points, status = constrained_early_feature_points(backbone, provisional, box)
                    if points is not None:
                        _, ids, logits = model.add_new_points_or_box(state, frame_idx=index, obj_id=1,
                            points=np.asarray(points['feature_points'], np.float32), labels=np.array([1, 0], np.int32), clear_old_points=False)
                        if ids != [1]: raise ValueError('Unexpected object')
                    mask = logits[0, 0].detach().cpu().numpy() > 0
                    Image.fromarray(mask.astype(np.uint8)*255).save(target / 'constrained.png')
                    records.append({'sequence': name, 'frame': index, 'point_status': status,
                                    'points': None if points is None else points['feature_points'],
                                    'mask_sha256': digest(target / 'constrained.png'),
                                    'old_prompt_sha256': old_record['prompt_sha256'],
                                    'image_sha256': frame_rows[index]['image_sha256'],
                                    'label_sha256': frame_rows[index]['label_sha256']})
            finally:
                model.reset_state(state)
                del state
                gc.collect()
        (out / 'progress.json').write_text(json.dumps({'completed': [x['sequence'] for x in records]}, indent=2) + '\n')
        print(f'{name}: prompt prediction saved, status={status}', flush=True)
    assert_parameter_integrity(model, before, out, 'after')
    (out / 'predictions.json').write_text(json.dumps(records, indent=2) + '\n')
    # All predictions for all sequences are complete before opening any labels.
    scored = []
    for row in records:
        name, index = row['sequence'], row['frame']; source = Path(locations[name]) / name
        label = videos[name][index].mask_path
        if digest(label) != row['label_sha256']: raise ValueError('Label changed')
        truth = load_label_mask(label, [1]) == 1
        masks = {'box_only': np.asarray(Image.open(source / 'prompt_provisional.png')) > 0,
                 'original_points': np.asarray(Image.open(source / 'prompt_refined.png')) > 0,
                 'constrained_points': np.asarray(Image.open(out / name / 'constrained.png')) > 0}
        old_prompt = read(source / 'prompt.json')
        if digest(source / 'prompt_refined.png') != old_prompt['refined_mask_sha256']:
            raise ValueError('Original refined mask changed')
        scores = {}
        for arm, pred in masks.items():
            if pred.shape != truth.shape: raise ValueError('Prediction shape differs')
            den = int(pred.sum() + truth.sum())
            scores[arm] = {'dice': 2 * int((pred & truth).sum()) / den if truth.any() and den else None,
                           'foreground_on_labeled_empty': bool(pred.any()) if not truth.any() else None}
        points = row['points']
        point_truth = None if not truth.any() or points is None else [bool(truth[int(y), int(x)]) for x, y in points]
        scored.append({**row, 'present': bool(truth.any()), 'scores': scores, 'constrained_point_membership': point_truth})
    present = [r for r in scored if r['present']]
    summary = {'scope': setup['scope'], 'frames': len(scored), 'present_frames': len(present),
               'mean_present_dice': {arm: float(np.mean([r['scores'][arm]['dice'] for r in present])) for arm in ('box_only','original_points','constrained_points')},
               'fallbacks': sum(r['point_status'] != 'available' for r in scored), 'sequences': scored}
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'),
                                                  'predictions_sha256': digest(out / 'predictions.json')}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__': main()

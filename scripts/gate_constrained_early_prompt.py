#!/usr/bin/env python3
"""Real Lean prompt-only engineering gate with exact previous-mask replay."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames  # noqa: E402
from modeling.constrained_prompt_feature_refinement import constrained_early_feature_points  # noqa: E402
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', default='seq3')
    a = p.parse_args()
    setup = json.loads((a.reference / 'setup.json').read_text())
    done = json.loads((a.reference / 'completion.json').read_text())
    audit = json.loads((a.reference / 'independent_audit.json').read_text())
    if not done['passed'] or not audit['passed'] or digest(a.reference / 'records.json') != done['records_sha256'] or audit['records_sha256'] != done['records_sha256']:
        raise ValueError('Reference gate incomplete')
    if digest(a.checkpoint) != setup['signature']['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from reference')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'), **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    frames = videos[a.sequence]; reference = a.reference / a.sequence
    prompt = json.loads((reference / 'prompt.json').read_text())
    index = prompt['first_detection_index']
    if index is None: raise ValueError('Expected an automatic prompt')
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    frame_dir = out / 'input'; frame_dir.mkdir(); stage_frames(frames, frame_dir)
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
    before = parameter_fingerprints(model)
    state = model.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=False)
    box = np.asarray(prompt['box'], np.float32)
    results = {}
    with torch.inference_mode():
        for arm in ('original', 'constrained'):
            _, ids, initial = model.add_new_points_or_box(state, frame_idx=index, obj_id=1, box=box)
            if ids != [1]: raise ValueError('Object ID differs')
            provisional = initial[0, 0].detach().cpu().numpy() > 0
            expected = np.asarray(Image.open(reference / 'prompt_provisional.png')) > 0
            if not np.array_equal(provisional, expected): raise ValueError('Box-only mask differs')
            _, backbone, _, _, _ = model._get_image_feature(state, index, batch_size=1)
            points, status = (early_feature_points(backbone, provisional, box, seed=20261114) if arm == 'original'
                              else constrained_early_feature_points(backbone, provisional, box))
            if points is None or status != 'available': raise ValueError('Gate does not exercise point selection')
            xy = points['feature_points']
            if arm == 'original' and xy != prompt['points']: raise ValueError('Existing point selection differs')
            if arm == 'constrained':
                if not provisional[int(xy[0][1]), int(xy[0][0])] or provisional[int(xy[1][1]), int(xy[1][0])]:
                    raise ValueError('Constrained candidates violate provisional mask')
            _, ids, output = model.add_new_points_or_box(state, frame_idx=index, obj_id=1,
                points=np.asarray(xy, np.float32), labels=np.array([1, 0], np.int32), clear_old_points=False)
            mask = output[0, 0].detach().cpu().numpy() > 0
            if arm == 'original' and not np.array_equal(mask, np.asarray(Image.open(reference / 'prompt_refined.png')) > 0):
                raise ValueError('Original refined mask differs')
            Image.fromarray(mask.astype(np.uint8)*255).save(out / f'{arm}.png')
            results[arm] = {'points': xy, 'status': status, 'mask_sha256': digest(out / f'{arm}.png')}
            model.reset_state(state)
    assert_parameter_integrity(model, before, out, 'after')
    result = {'source': 'label-free same first YOLO box; prompt-frame engineering gate only',
              'sequence': a.sequence, 'frame': index, 'revision': revision, 'checkpoint_sha256': digest(a.checkpoint),
              'reference_records_sha256': done['records_sha256'], 'model_source_sha256': digest(ROOT / 'modeling/constrained_prompt_feature_refinement.py'),
              'original_provisional_and_refined_masks_match': True, 'arms': results}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json')}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

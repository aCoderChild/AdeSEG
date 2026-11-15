#!/usr/bin/env python3
"""Full-video constrained-point comparison against audited automatic Lean prompts."""
import argparse
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
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from modeling.constrained_prompt_feature_refinement import constrained_early_feature_points  # noqa: E402
from modeling.lean_multiscale_anchor import install_multiscale_anchor  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402

PROTOCOL = ROOT / 'protocols/constrained_early_prompt_v1.json'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'


def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'merged', 'historical', 'pilot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=('replay', 'candidate'), required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    a = p.parse_args()
    protocol = read(PROTOCOL)
    if a.mode == 'replay' and a.sequence != 'seq3': raise ValueError('Replay gate must use seq3')
    if a.mode == 'candidate' and a.sequence not in protocol['cohort']: raise ValueError('Sequence outside frozen cohort')
    merge_done = read(a.merged / 'completion.json')
    if not merge_done['passed'] or digest(a.merged / 'results.json') != merge_done['results_sha256'] or digest(a.historical / 'comparison.json') != merge_done['historical_comparison_sha256']:
        raise ValueError('Reference comparison incomplete or changed')
    location = Path(read(a.merged / 'locations.json')[a.sequence]); folder = location / a.sequence
    old_setup = read(location / 'setup.json'); old_done = read(location / 'completion.json'); audit = read(location / 'independent_audit.json')
    if not old_done['passed'] or not audit['passed'] or digest(location / 'records.json') != old_done['records_sha256'] or audit['records_sha256'] != old_done['records_sha256']:
        raise ValueError('Old run incomplete')
    signature = old_setup['signature']
    sources = {'early_module_sha256': ROOT / 'modeling/early_feature_prompt.py',
               'prompt_module_sha256': ROOT / 'modeling/prompt_feature_refinement.py',
               'memory_module_sha256': ROOT / 'modeling/lean_multiscale_anchor.py',
               'lean_runner_sha256': ROOT / 'scripts/run_author_baseline.py',
               'runner_sha256': ROOT / 'scripts/run_lean_auto_multiscale.py',
               'bypass_sha256': ROOT / 'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth',
               'lean_config_sha256': ROOT / 'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml',
               'manifest_sha256': MANIFEST}
    if (digest(a.checkpoint) != signature['checkpoint_sha256'] or any(digest(path) != signature[k] for k, path in sources.items())
        or torch.__version__ != signature['torch'] or np.__version__ != signature['numpy'] or sys.version != signature['python']):
        raise ValueError('Model, input or source differs from original run')
    gate_identity = {'runner_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL),
                     'checkpoint_sha256': digest(a.checkpoint), 'merged_results_sha256': merge_done['results_sha256'],
                     'pilot_predictions_sha256': digest(a.pilot / 'predictions.json'),
                     'candidate_source_sha256': digest(ROOT / 'modeling/constrained_prompt_feature_refinement.py'),
                     'source_signature': signature}
    if a.mode == 'candidate':
        if a.gate_a is None or a.gate_b is None: raise ValueError('Two replay gates required')
        gates = []
        for g in (a.gate_a, a.gate_b):
            meta, receipt = read(g / 'setup.json'), read(g / 'completion.json')
            if meta['mode'] != 'replay' or meta['gate_identity'] != gate_identity or not receipt['passed'] or digest(g / 'records.json') != receipt['records_sha256']:
                raise ValueError('Replay gate invalid')
            gates.append(read(g / 'records.json'))
        if gates[0] != gates[1]: raise ValueError('Independent replays differ')
    pilot_done, pilot_audit = read(a.pilot / 'completion.json'), read(a.pilot / 'independent_audit.json')
    pilot_setup = read(a.pilot / 'setup.json')
    geometry = read(a.pilot / 'geometry_audit.json')
    if (not pilot_done['passed'] or not pilot_audit['passed'] or not geometry['passed']
        or digest(a.pilot / 'predictions.json') != pilot_done['predictions_sha256']
        or digest(a.pilot / 'results.json') != pilot_done['results_sha256']
        or pilot_audit['results_sha256'] != pilot_done['results_sha256']
        or pilot_setup['method_sha256'] != gate_identity['candidate_source_sha256']
        or pilot_setup['checkpoint_sha256'] != gate_identity['checkpoint_sha256']
        or geometry['predictions_sha256'] != pilot_done['predictions_sha256']):
        raise ValueError('Prompt pilot or audits invalid')
    pilot_rows = {r['sequence']: r for r in read(a.pilot / 'predictions.json')}
    old_record = read(location / 'records.json')[0]
    for key, filename in [('prompt_sha256','prompt.json'), ('frames_sha256','frames.json'), ('selections_sha256','selections.json')]:
        if digest(folder / filename) != old_record[key]: raise ValueError('Old receipts changed')
    prompt, old_rows = read(folder / 'prompt.json'), read(folder / 'frames.json')
    index = prompt['first_detection_index']
    if index is None: raise ValueError('No original automatic detection')
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}; frames = data[a.sequence]
    if len(frames) != len(old_rows): raise ValueError('Frame count mismatch')
    for frame, row in zip(frames, old_rows):
        if digest(frame.image_path) != row['image_sha256']: raise ValueError('Image changed')
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    (out / 'setup.json').write_text(json.dumps({'mode': a.mode, 'sequence': a.sequence, 'gate_identity': gate_identity,
        'strength': .5, 'original_records_sha256': old_done['records_sha256']}, indent=2) + '\n')
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    if revision != signature['revision']: raise ValueError('Lean source differs')
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
    before = parameter_fingerprints(model)
    selections = []; events = install_multiscale_anchor(model, .5, selection_events=selections)
    for i in range(index):
        with Image.open(frames[i].image_path) as image: w, h = image.size
        Image.fromarray(np.zeros((h,w), np.uint8)).save(out / f'{i:06d}.png')
    with tempfile.TemporaryDirectory(prefix='staged_', dir=out) as temp:
        stage_frames(frames, Path(temp))
        staged = sorted(Path(temp).glob('*.jpg'), key=lambda path: int(path.stem))
        if len(staged) != len(frames) or any(p.resolve() != f.image_path.resolve() for p, f in zip(staged, frames)):
            raise ValueError('Frame order differs')
        state = model.init_state(video_path=temp, offload_video_to_cpu=True, offload_state_to_cpu=False)
        try:
            with torch.inference_mode():
                box = np.asarray(prompt['box'], np.float32)
                _, ids, logits = model.add_new_points_or_box(state, frame_idx=index, obj_id=1, box=box)
                if ids != [1]: raise ValueError('Object ID differs')
                provisional = logits[0,0].detach().cpu().numpy() > 0
                if digest(folder / 'prompt_provisional.png') != prompt['provisional_mask_sha256'] or not np.array_equal(provisional, np.asarray(Image.open(folder / 'prompt_provisional.png')) > 0):
                    raise ValueError('Provisional prompt does not replay')
                _, backbone, _, _, _ = model._get_image_feature(state, index, batch_size=1)
                points, status = early_feature_points(backbone, provisional, box, seed=20261114)
                if status != prompt['point_status'] or (points is not None and points['feature_points'] != prompt['points']):
                    raise ValueError('Feature prompt does not replay')
                if a.mode == 'candidate':
                    points, status = constrained_early_feature_points(backbone, provisional, box)
                    saved = pilot_rows[a.sequence]
                    if status != saved['point_status'] or (None if points is None else points['feature_points']) != saved['points']:
                        raise ValueError('Candidate points differ from pilot')
                if points is not None:
                    _, ids, logits = model.add_new_points_or_box(state, frame_idx=index, obj_id=1,
                        points=np.asarray(points['feature_points'], np.float32), labels=np.array([1,0], np.int32), clear_old_points=False)
                mask = logits[0,0].detach().cpu().numpy() > 0
                if a.mode == 'replay':
                    if digest(folder / 'prompt_refined.png') != prompt['refined_mask_sha256'] or not np.array_equal(mask, np.asarray(Image.open(folder / 'prompt_refined.png')) > 0):
                        raise ValueError('Refined prompt does not replay')
                else:
                    candidate_file = a.pilot / a.sequence / 'constrained.png'
                    if digest(candidate_file) != pilot_rows[a.sequence]['mask_sha256'] or not np.array_equal(mask, np.asarray(Image.open(candidate_file)) > 0):
                        raise ValueError('Candidate prompt differs from audited pilot')
                if mask.any(): model.add_new_mask(state, frame_idx=index, obj_id=1, mask=mask)
                count = 0
                for i, ids, output in model.propagate_in_video(state, start_frame_idx=index, reverse=False):
                    if i != index + count or ids != [1]: raise ValueError('Propagation order differs')
                    Image.fromarray((output[0,0].detach().cpu().numpy() > 0).astype(np.uint8)*255).save(out / f'{i:06d}.png')
                    count += 1
                if count != len(frames)-index: raise ValueError('Incomplete propagation')
        finally: model.reset_state(state)
    assert_parameter_integrity(model, before, out, 'after')
    mask_hashes = [digest(out / f'{i:06d}.png') for i in range(len(frames))]
    if a.mode == 'replay':
        if mask_hashes != [x['mask_sha256'] for x in old_rows] or selections != read(folder / 'selections.json'):
            raise ValueError('Replay differs from original masks or token selections')
    (out / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
    (out / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    if a.mode == 'candidate':
        if not events: raise ValueError('Multi-scale wrapper did not activate')
        rows = []
        # No labels read until every prediction in this video is saved.
        for i, frame in enumerate(frames):
            if digest(frame.mask_path) != old_rows[i]['label_sha256']: raise ValueError('Label changed')
            truth = load_label_mask(frame.mask_path, [1]) == 1
            pred = np.asarray(Image.open(out / f'{i:06d}.png')) > 0
            den = int(pred.sum()+truth.sum()); present = bool(truth.any())
            rows.append({'frame': i, 'present': present, 'dice': 2*int((pred & truth).sum())/den if den else 1.,
                         'false_alarm': bool(pred.any()) if not present else None, 'mask_sha256': mask_hashes[i]})
        (out / 'frames.json').write_text(json.dumps(rows, indent=2) + '\n')
    result = {'sequence': a.sequence, 'frames': len(frames), 'first_detection_index': index,
              'mask_sha256': mask_hashes, 'events': len(events), 'selections': selections,
              'original_records_sha256': old_done['records_sha256'],
              'frames_sha256': digest(out / 'frames.json') if a.mode == 'candidate' else None}
    (out / 'records.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(out / 'records.json')}, indent=2) + '\n')
    print(json.dumps({'sequence': a.sequence, 'mode': a.mode, 'frames': len(frames), 'selections': len(selections)}, indent=2))


if __name__ == '__main__': main()

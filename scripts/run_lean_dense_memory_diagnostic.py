#!/usr/bin/env python3
"""Controlled full-video fixed 100% TAMP retention diagnostic."""
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
from modeling.lean_multiscale_anchor import install_multiscale_anchor  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402

PROTOCOL = ROOT / 'protocols/lean_auto_dense_memory_diagnostic_v1.json'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'


def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'merged', 'historical', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=('replay', 'ablation'), required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    p.add_argument('--dense-gate-a', type=Path)
    p.add_argument('--dense-gate-b', type=Path)
    p.add_argument('--engineering', action='store_true')
    a = p.parse_args()
    protocol = read(PROTOCOL)
    if a.mode == 'replay' and a.sequence != 'seq3': raise ValueError('Replay gate must use seq3')
    if a.engineering and (a.mode != 'ablation' or a.sequence not in protocol['engineering_sequences']):
        raise ValueError('Engineering mode requires a designated dense test sequence')
    if a.mode == 'ablation' and not a.engineering and a.sequence not in protocol['cohort']:
        raise ValueError('Sequence outside frozen cohort')
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
                     'source_signature': signature}
    if a.mode == 'ablation':
        if a.gate_a is None or a.gate_b is None: raise ValueError('Two replay gates required')
        gates = []
        for g in (a.gate_a, a.gate_b):
            meta, receipt = read(g / 'setup.json'), read(g / 'completion.json')
            if meta['mode'] != 'replay' or meta['gate_identity'] != gate_identity or not receipt['passed'] or digest(g / 'records.json') != receipt['records_sha256']:
                raise ValueError('Replay gate invalid')
            gates.append(read(g / 'records.json'))
        if gates[0] != gates[1]: raise ValueError('Independent replays differ')
        if not a.engineering:
            if a.dense_gate_a is None or a.dense_gate_b is None:
                raise ValueError('Two independent dense engineering runs are required')
            dense_gates = []
            for g in (a.dense_gate_a, a.dense_gate_b):
                meta, receipt = read(g / 'setup.json'), read(g / 'completion.json')
                if (meta['mode'] != 'ablation' or not meta['engineering'] or meta['sequence'] != 'seq3'
                    or meta['gate_identity'] != gate_identity or not receipt['passed']
                    or digest(g / 'records.json') != receipt['records_sha256']):
                    raise ValueError('Dense engineering gate invalid')
                dense_record = read(g / 'records.json')
                if (dense_record['keep_ratio'] != 1.0 or not dense_record['selections']
                    or any(x['kept'] != x['candidates'] for x in dense_record['selections'])
                    or digest(g / 'memory_reads.json') != dense_record['memory_reads_sha256']):
                    raise ValueError('Dense engineering token receipt invalid')
                dense_gates.append(dense_record)
            if dense_gates[0] != dense_gates[1]: raise ValueError('Independent dense replays differ')
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
    (out / 'setup.json').write_text(json.dumps({'mode': a.mode, 'sequence': a.sequence, 'engineering': a.engineering, 'gate_identity': gate_identity,
        'strength': .5, 'keep_ratio': .25 if a.mode == 'replay' else 1.0, 'original_records_sha256': old_done['records_sha256']}, indent=2) + '\n')
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    if revision != signature['revision']: raise ValueError('Lean source differs')
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
    before = parameter_fingerprints(model)
    if model.mem_keep_ratio != .25: raise ValueError('Original Lean retention differs')
    if a.mode == 'ablation': model.mem_keep_ratio = 1.0
    selections = []; events = install_multiscale_anchor(model, .5, selection_events=selections)
    reads = []
    def trace_memory(_module, _args, kwargs):
        memory, position = kwargs['memory'], kwargs['memory_pos']
        ptrs = int(kwargs['num_obj_ptr_tokens'])
        if memory.shape != position.shape or ptrs < 0 or ptrs > memory.shape[0]:
            raise ValueError('Memory/position or pointer-tail shape mismatch')
        reads.append({'memory_tokens': int(memory.shape[0]), 'pointer_tokens': ptrs,
                      'batch': int(memory.shape[1]), 'channels': int(memory.shape[2])})
    hook = model.memory_attention.register_forward_pre_hook(trace_memory, with_kwargs=True)
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
                if points is not None:
                    _, ids, logits = model.add_new_points_or_box(state, frame_idx=index, obj_id=1,
                        points=np.asarray(points['feature_points'], np.float32), labels=np.array([1,0], np.int32), clear_old_points=False)
                mask = logits[0,0].detach().cpu().numpy() > 0
                if digest(folder / 'prompt_refined.png') != prompt['refined_mask_sha256'] or not np.array_equal(mask, np.asarray(Image.open(folder / 'prompt_refined.png')) > 0):
                    raise ValueError('Refined prompt does not replay')
                if mask.any(): model.add_new_mask(state, frame_idx=index, obj_id=1, mask=mask)
                count = 0
                for i, ids, output in model.propagate_in_video(state, start_frame_idx=index, reverse=False):
                    if i != index + count or ids != [1]: raise ValueError('Propagation order differs')
                    Image.fromarray((output[0,0].detach().cpu().numpy() > 0).astype(np.uint8)*255).save(out / f'{i:06d}.png')
                    count += 1
                if count != len(frames)-index: raise ValueError('Incomplete propagation')
        finally: model.reset_state(state)
    hook.remove()
    assert_parameter_integrity(model, before, out, 'after')
    if not reads: raise ValueError('No memory attention inputs observed')
    (out / 'memory_reads.json').write_text(json.dumps(reads, indent=2) + '\n')
    mask_hashes = [digest(out / f'{i:06d}.png') for i in range(len(frames))]
    if a.mode == 'replay':
        if mask_hashes != [x['mask_sha256'] for x in old_rows] or selections != read(folder / 'selections.json'):
            raise ValueError('Replay differs from original masks or token selections')
    (out / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
    (out / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    if a.mode == 'ablation':
        if not events or not selections: raise ValueError('Dense-retention path was not exercised')
        if any(x['kept'] != x['candidates'] for x in selections):
            raise ValueError('Dense retention did not keep all spatial candidates')
    if a.mode == 'ablation' and not a.engineering:
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
              'memory_reads_sha256': digest(out / 'memory_reads.json'),
              'keep_ratio': model.mem_keep_ratio,
              'frames_sha256': digest(out / 'frames.json') if a.mode == 'ablation' and not a.engineering else None}
    (out / 'records.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(out / 'records.json')}, indent=2) + '\n')
    print(json.dumps({'sequence': a.sequence, 'mode': a.mode, 'frames': len(frames), 'selections': len(selections)}, indent=2))


if __name__ == '__main__': main()

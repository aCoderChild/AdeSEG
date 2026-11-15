#!/usr/bin/env python3
"""Evaluate fixed multiscale scores on archived native histories; no writes to SAM2."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.multiscale_memory_utility import multiscale_scores, attention_tile_mass, evaluate_scores
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts.memory_token_io import digest, parameter_digest
from scripts.run_author_baseline import tensor_hash
from scripts.run_multiscale_memory_utility import inspect_read
from scripts.run_signed_memory_feature_utility_overlap import activate_model
from evaluation.statistics import sequence_bootstrap_interval

PROTOCOL = ROOT / 'protocols/multiscale_memory_utility_cohort_v1.json'
PILOT = {('seq2', 4, 6), ('seq3', 1, 3), ('seq17', 17, 19), ('seq22', 41, 45)}
SOURCES = ('protocols/multiscale_memory_utility_cohort_v1.json', 'protocols/multiscale_memory_utility_diagnostic_v1.json',
           'scripts/run_multiscale_memory_utility_cohort.py', 'scripts/run_multiscale_memory_utility.py',
           'modeling/multiscale_memory_utility.py', 'modeling/token_diagnostic_replay.py')


def verify_setup(protocol):
    root = Path(protocol['utility_root'])
    for name, key in (('setup.json', 'prior_scientific_setup_sha256'), ('completion.json', 'prior_scientific_completion_sha256')):
        if digest(root / name) != protocol[key]:
            raise RuntimeError(f'Archived utility {name} changed')
    if digest(ROOT / protocol['descriptor_protocol']) != protocol['descriptor_protocol_sha256']:
        raise RuntimeError('Frozen descriptor protocol changed')
    source = json.loads((ROOT / 'protocols/signed_memory_feature_utility_overlap_v1.json').read_text())
    if digest(Path(source['checkpoint'])) != protocol['checkpoint_sha256'] or digest(ROOT / source['manifest']) != source['manifest_sha256']:
        raise RuntimeError('Checkpoint or manifest changed')
    return source, json.loads((root / 'setup.json').read_text())


def load_sequence(protocol, name, frames, archived_setup):
    item = protocol['plan'][name]
    utility_path = Path(protocol['utility_root']) / 'sequences' / name / 'results.json'
    folder = Path(item['bank_dir'])
    if digest(utility_path) != item['utility_results_sha256'] or digest(folder / 'receipt.json') != item['bank_receipt_sha256'] or digest(folder / 'bank.pt') != item['bank_sha256']:
        raise RuntimeError(f'Prior results or native bank changed: {name}')
    receipt = json.loads((folder / 'receipt.json').read_text())
    if receipt['checkpoint_sha256'] != protocol['checkpoint_sha256'] or receipt['sequence'] != name or receipt['num_frames'] != len(frames):
        raise RuntimeError(f'Prior bank provenance mismatch: {name}')
    inputs = archived_setup['inputs'][name]['frames']
    if [e['local_frame'] for e in inputs] != list(range(receipt['prompt'], len(frames))):
        raise RuntimeError('Prior post-prompt input coverage differs')
    for expected in inputs:
        frame = frames[expected['local_frame']]
        if frame.frame_index != expected['source_frame'] or digest(frame.image_path) != expected['image_sha256'] or digest(frame.mask_path) != expected['mask_sha256']:
            raise RuntimeError(f'Input image/mask mismatch: {name} frame {frame.frame_index}')
    bank = torch.load(folder / 'bank.pt', map_location='cpu', weights_only=True)
    if bank['num_frames'] != len(frames) or bank['prompt'] != receipt['prompt'] or bank['object_id'] != 1:
        raise RuntimeError('Native bank structure mismatch')
    records = json.loads(utility_path.read_text())['main']
    return bank, records


def replay_target(model, frames, bank, item):
    from sam2.utils.misc import _load_img_as_tensor
    frame = item['target_frame']
    image, _, _ = _load_img_as_tensor(str(frames[frame].image_path), model.image_size)
    # The native synchronous loader assigns the float64 decoded pixels into a
    # float32 tensor BEFORE applying float32 normalization.
    image = image.float()
    image -= torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    image /= torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    image = image.unsqueeze(0)
    _, vision, positions, sizes = model._prepare_backbone_features(model.forward_image(image))
    prior = {key: {i: v for i, v in bank['outputs'][key].items() if i < frame}
             for key in ('cond_frame_outputs', 'non_cond_frame_outputs')}
    if not prior['cond_frame_outputs']:
        raise RuntimeError('No prior conditioning memory')
    with torch.no_grad(), NativeReadCapture(model) as capture:
        prediction = model.track_step(frame_idx=frame, is_init_cond_frame=False,
            current_vision_feats=vision, current_vision_pos_embeds=positions,
            feat_sizes=sizes, point_inputs=None, mask_inputs=None, output_dict=prior,
            num_frames=bank['num_frames'], track_in_reverse=False, run_mem_encoder=False)
    if prediction['maskmem_features'] is not None or capture.layers != [0, 1, 2, 3]:
        raise RuntimeError('Unexpected memory write or incomplete attention')
    from PIL import Image
    with Image.open(frames[frame].image_path) as source:
        width, height = source.size
    full = F.interpolate(prediction['pred_masks'], size=(height, width), mode='bilinear', align_corners=False)
    if tensor_hash(full) != item['native_prediction_sha256']:
        raise RuntimeError(f'Native full-resolution replay differs: {frame}')
    return capture, prior


def score_target(model, capture, prior, pair):
    spans = pair['spans']
    memory = capture.memory['memory']
    cursor = 0
    for span in spans:
        if span['start'] != cursor or span['stop'] - cursor != 4096:
            raise RuntimeError('Unexpected native spatial layout')
        stored = prior['cond_frame_outputs'].get(span['frame'])
        if stored is None:
            stored = prior['non_cond_frame_outputs'].get(span['frame'])
        if stored is None or not torch.equal(memory[span['start']:span['stop'], 0], stored['maskmem_features'][0].flatten(1).T):
            raise RuntimeError('Memory tokens do not match source bank')
        cursor = span['stop']
    if cursor != memory.shape[0] - capture.memory['num_obj_ptr_tokens']:
        raise RuntimeError('Spatial/pointer boundary mismatch')
    captured = {'memory': capture.memory, 'decoder': capture.decoder, 'outputs': capture.outputs, 'layers': capture.layers}
    projected, rope = inspect_read(model, captured)
    source = next(span for span in spans if span['frame'] == pair['source_frame'])
    start, stop = source['start'], source['stop']
    q, k = projected['q'][0], projected['k'][0, start:stop]
    if q.shape[0] != 4096 or q.shape != k.shape:
        raise RuntimeError('Unexpected projected query/key shape')
    scores = multiscale_scores(q, k)
    scores['native_attention'] = attention_tile_mass(rope['q'], rope['k'], start, stop)
    return {key: value.tolist() for key, value in scores.items()}


def summarize(rows):
    out = {}
    for subset in ('all', 'without_pilot'):
        selected = [r for r in rows if subset == 'all' or not r['pilot_pair']]
        sequences = sorted({r['sequence'] for r in selected})
        if not sequences:
            continue
        result = {'pairs': len(selected), 'sequences': len(sequences), 'methods': {}}
        for method in ('scale_1', 'multiscale', 'native_attention'):
            rho, decision = {}, {}
            for name in sequences:
                entries = [r['metrics'][method] for r in selected if r['sequence'] == name]
                available = [e['spearman_rho'] for e in entries if e['spearman_rho'] is not None]
                if available:
                    rho[name] = float(np.mean(available))
                decision[name] = float(np.mean([e['selected_minus_uniform_dice'] for e in entries]))
            result['methods'][method] = {'equal_sequence_mean_spearman': float(np.mean(list(rho.values()))) if rho else None,
                'sequences_with_spearman': len(rho), 'pairs_with_spearman': sum(r['metrics'][method]['spearman_rho'] is not None for r in selected),
                'equal_sequence_selected_minus_uniform_dice': float(np.mean(list(decision.values())))}
        for label, field in (('spearman', 'spearman_rho'), ('selected_minus_uniform_dice', 'selected_minus_uniform_dice')):
            diff = {}
            for name in sequences:
                matched = [r for r in selected if r['sequence'] == name
                           and r['metrics']['multiscale'][field] is not None
                           and r['metrics']['scale_1'][field] is not None]
                if matched:
                    diff[name] = float(np.mean([r['metrics']['multiscale'][field] - r['metrics']['scale_1'][field] for r in matched]))
            result[f'multiscale_minus_local_{label}'] = {'n_sequences': len(diff),
                'mean': float(np.mean(list(diff.values()))) if diff else None,
                'sequence_bootstrap_95_ci': sequence_bootstrap_interval(diff) if diff else None}
        result['same_selected_tiles'] = sum(r['metrics']['scale_1']['selected_tile_ids'] == r['metrics']['multiscale']['selected_tile_ids'] for r in selected)
        out[subset] = result
    return out


def run(out, names):
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads(PROTOCOL.read_text())
    source, archived_setup = verify_setup(protocol)
    videos = {**load_manifest(ROOT / source['manifest'], 'dev'), **load_manifest(ROOT / source['manifest'], 'test')}
    if not names:
        names = list(protocol['plan'])
    if len(set(names)) != len(names) or not set(names) <= set(protocol['plan']):
        raise ValueError('Invalid sequence selection')
    model, revision = activate_model(source)
    weights = parameter_digest(model)
    out.mkdir(parents=True)
    setup = {'protocol_sha256': digest(PROTOCOL), 'sources': {s: digest(ROOT / s) for s in SOURCES}, 'sequences': names,
             'sam2_revision': revision, 'checkpoint_sha256': source['checkpoint_sha256'], 'torch_version': torch.__version__, 'started_unix': time.time()}
    (out / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    rows = []
    for name in names:
        started = time.monotonic()
        bank, archived = load_sequence(protocol, name, videos[name], archived_setup)
        for item in protocol['plan'][name]['targets']:
            with torch.no_grad():
                capture, prior = replay_target(model, videos[name], bank, item)
                record = next(r for r in archived if r['anchor_target'] == item['target_frame'] and r['source_frame'] == item['source_frame'])
                native = record['arms']['native']
                spans = native['read']['memory_frames']
                if [s['frame'] for s in spans] != item['memory_sources']:
                    raise RuntimeError('Frozen source order differs')
                scores = score_target(model, capture, prior, {'spans': spans, 'source_frame': item['source_frame']})
            # Saved Dice outcomes are used only after descriptors have been computed.
            utility = [float(native['scores']['dice']) - float(record['arms'][f'tile_{i:02d}']['scores']['dice']) for i in range(16)]
            for tile in range(16):
                a = record['arms'][f'tile_{tile:02d}']
                if [x['source_frame'] for x in a['read']['selection_layers']] != [item['source_frame']] * 4:
                    raise RuntimeError('Source-local intervention differs')
            metrics = {mode: evaluate_scores(scores[mode], utility) for mode in ('scale_1', 'multiscale', 'native_attention')}
            rows.append({'sequence': name, **item, 'pilot_pair': (name, item['source_frame'], item['target_frame']) in PILOT,
                         'scores': scores, 'tile_removal_utility': utility, 'metrics': metrics})
        if parameter_digest(model) != weights:
            raise RuntimeError('Model weights changed')
        (out / 'progress.json').write_text(json.dumps({'completed_sequences': len({r['sequence'] for r in rows}), 'expected_sequences': len(names),
                                                        'completed_pairs': len(rows), 'last_sequence': name, 'sequence_seconds': time.monotonic()-started}) + '\n')
        del bank, archived, capture, prior
        gc.collect()
    result = {'scope': protocol['cohort'], 'rows': rows, 'summary': summarize(rows), 'limitations': protocol['limitations'], 'efficacy_evaluated': False}
    path = out / 'results.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'completion.json').write_text(json.dumps({'complete': True, 'setup_sha256': digest(out / 'setup.json'), 'results_sha256': digest(path)}, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence', action='append')
    args = p.parse_args()
    run(args.output_dir, args.sequence)


if __name__ == '__main__':
    main()

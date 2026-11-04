"""Read-only decoder capture on verified native-history SAM2 reads."""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.decoder_training_mask import decoder_training_mask, select_decoder_mask
from modeling.native_history_cache import cached_read, load_bank
from modeling.native_history_training import compact_history_read
from modeling.prediction_conditioned_affinity import memory_role_sequence
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame

PROTOCOL = 'protocols/native_decoder_diagnostic_v1.json'


def native_read(model, sample):
    return model.track_step(frame_idx=sample['frame'], is_init_cond_frame=False,
        current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
        feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
        output_dict=sample['output_dict'], num_frames=sample['num_frames'],
        track_in_reverse=False, run_mem_encoder=True)


def verify_stored(output, stored):
    for key in ('pred_masks', 'object_score_logits', 'obj_ptr', 'maskmem_features'):
        value = output[key]
        if key == 'maskmem_features':
            value = value.to(stored[key].dtype)
        if value.dtype != stored[key].dtype or not torch.equal(value, stored[key]):
            raise RuntimeError(f'Native stored output differs: {key}')
    if len(output['maskmem_pos_enc']) != len(stored['maskmem_pos_enc']) or any(
            not torch.equal(a, b) for a, b in zip(output['maskmem_pos_enc'], stored['maskmem_pos_enc'])):
        raise RuntimeError('Native memory positions differ')


def observe(model, sample):
    calls, handles = [], []
    for i, layer in enumerate(model.memory_attention.layers):
        def hook(_module, _args, kwargs, _output, index=i):
            calls.append({'layer': index, 'keys': kwargs['k'].shape[1],
                'queries': kwargs['q'].shape[1],
                'pointer_keys': kwargs.get('num_k_exclude_rope', 0)})
        handles.append(layer.cross_attn_image.register_forward_hook(hook, with_kwargs=True))
    try:
        with decoder_training_mask(model.sam_mask_decoder) as captured:
            output = native_read(model, sample)
        raw = select_decoder_mask(captured).detach().clone()
        raw_score = captured[0][3].detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return output, raw, raw_score, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    if not args.gate_only and args.gate is None:
        parser.error('The complete diagnostic requires a passed --gate')
    out = validate_output(args.output_dir)
    out.mkdir(exist_ok=False, parents=True)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    base = Path(protocol['drive_root'])
    cache, cv = base / protocol['cache'], base / protocol['cv']
    setup = json.loads((cv / 'setup.json').read_text())
    sources = {**setup['sources'], **{p: digest(ROOT / p) for p in
        (PROTOCOL, 'scripts/diagnose_native_decoder_masks.py', 'tests/test_native_decoder_diagnostic.py')}}
    if any(digest(ROOT / p) != h for p, h in sources.items()):
        raise RuntimeError('Frozen scientific source differs')
    if digest(base / protocol['selection_source']) != protocol['selection_sha256']:
        raise RuntimeError('Outcome-selection source differs')
    records = []
    if args.gate:
        gate = json.loads((args.gate / 'completion.json').read_text())
        if not gate.get('passed') or gate['sources'] != sources or gate['scope'] != 'gate':
            raise RuntimeError('Native decoder gate missing or different')
        if digest(args.gate / 'records.json') != gate['records_sha256']:
            raise RuntimeError('Gate records changed')
        records = json.loads((args.gate / 'records.json').read_text())
        expected = [c for c in protocol['cases'] if c['sequence'] in protocol['gate_sequences']]
        if [{k: r[k] for k in ('sequence', 'category', 'frame')} for r in records] != expected:
            raise RuntimeError('Gate case coverage differs')
    checkpoint = ROOT / protocol['checkpoint']
    if (digest(checkpoint) != setup['checkpoint_sha256'] or
            digest(ROOT / protocol['manifest']) != setup['manifest_sha256']):
        raise RuntimeError('Checkpoint or manifest differs')
    write(out / 'setup.json', {'sources': sources, 'checkpoint_sha256': setup['checkpoint_sha256'],
        'manifest_sha256': setup['manifest_sha256'], 'selection': protocol['cases']})
    videos = {**load_manifest(ROOT / protocol['manifest'], 'dev'),
              **load_manifest(ROOT / protocol['manifest'], 'test')}
    repo, revision = activate_author_repo()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cases = [c for c in protocol['cases'] if not args.gate_only or c['sequence'] in protocol['gate_sequences']]
    reused = len(records)
    pending = [c for c in cases if not args.gate or c['sequence'] not in protocol['gate_sequences']]
    for sequence in sorted({c['sequence'] for c in pending}, key=lambda s: int(s[3:])):
        receipt = json.loads((cache / sequence / 'receipt.json').read_text())
        payload = load_bank(cache / sequence / 'bank.pt', receipt, setup['checkpoint_sha256'],
                            setup['manifest_sha256'], setup['model_config'])
        native = {r['frame']: r for r in json.loads((base / protocol['native'] / f'{sequence}_frames.json').read_text())}
        if receipt['predictions'] != [{'frame': i, 'sha256': r['prediction_sha256']} for i, r in native.items()]:
            raise RuntimeError('Native prediction receipts differ')
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
            checkpoint=checkpoint, model_config=setup['model_config']), repo)
        before = parameter_digest(model)
        frames = videos[sequence]
        with tempfile.TemporaryDirectory(prefix='native_decoder_', dir=out) as folder, torch.inference_mode():
            state, prompt = prepare(model, frames, Path(folder))
            try:
                if prompt != payload['prompt']:
                    raise RuntimeError('Original prompt differs')
                for case in (c for c in pending if c['sequence'] == sequence):
                    frame = case['frame']
                    sample = cached_read(model, state, frame, payload)
                    _, spans = memory_role_sequence(model, frame, sample['output_dict'], sample['num_frames'])
                    sample = compact_history_read(sample, [s['frame'] for s in spans])
                    bank_before = fingerprint(sample['output_dict'])['sha256']
                    stored = payload['outputs']['non_cond_frame_outputs'][frame]
                    ordinary = native_read(model, sample)
                    observed, raw, raw_score, calls = observe(model, sample)
                    verify_stored(ordinary, stored)
                    verify_stored(observed, stored)
                    if not torch.equal(raw_score, stored['object_score_logits']):
                        raise RuntimeError('Captured decoder score differs')
                    if [c['layer'] for c in calls] != [0, 1, 2, 3] or any(
                            c['keys'] - c['pointer_keys'] != 4096 * len(spans) for c in calls):
                        raise RuntimeError('Attention layer or native key schedule differs')
                    _, logits = model._get_orig_video_res_output(state, observed['pred_masks'])
                    if tensor_hash(logits) != native[frame]['prediction_sha256']:
                        raise RuntimeError('Original-resolution native hash differs')
                    if fingerprint(sample['output_dict'])['sha256'] != bank_before:
                        raise RuntimeError('Replay changed the source bank')
                    _, raw_logits = model._get_orig_video_res_output(state, raw)
                    scored = score_frame(logits, frames, frame)
                    if scored['scores'] != native[frame]['scores'] or scored['present'] != native[frame]['present']:
                        raise RuntimeError('Current GT scoring differs from native archive')
                    row = {**case, 'native': scored,
                        'raw_decoder': score_frame(raw_logits, frames, frame),
                        'object_score': float(raw_score.item()), 'attention': calls,
                        'spatial_frames': spans, 'native_prediction_sha256': tensor_hash(logits),
                        'raw_low_res_sha256': tensor_hash(raw), 'native_identity': True,
                        'source_bank_unchanged': True}
                    records.append(row)
                    write(out / 'records.json', records)
                    print(sequence, frame, case['category'], flush=True)
                    del sample, ordinary, observed, raw, logits, raw_logits
            finally:
                model.reset_state(state)
        if parameter_digest(model) != before or fingerprint(payload)['sha256'] != receipt['payload_sha256']:
            raise RuntimeError('Native parameters or cached bank changed')
        del model, payload
        gc.collect()
    if len(records) != len(cases):
        raise RuntimeError('Incomplete planned native replay')
    write(out / 'completion.json', {'passed': True, 'scope': 'gate' if args.gate_only else 'diagnostic',
        'cases': len(records), 'sources': sources, 'sam2_revision': revision,
        'records_sha256': digest(out / 'records.json'), 'training': False,
        'reused_gate_records': reused,
        'note': 'Raw-mask measurements are diagnostic; no altered predictions were propagated or stored.'})


if __name__ == '__main__':
    main()

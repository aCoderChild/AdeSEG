"""Matched oracle-box diagnostic; never propagate oracle predictions."""
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
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.decoder_training_mask import decoder_training_mask, select_decoder_mask
from modeling.native_history_cache import cached_read, load_bank
from modeling.native_history_training import compact_history_read
from modeling.prediction_conditioned_affinity import memory_role_sequence
from scripts.bank_integrity import fingerprint
from scripts.diagnose_native_decoder_masks import native_read, verify_stored
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_soft_role_affinity_cv import score_frame

PROTOCOL = 'protocols/oracle_memory_conditioning_diagnostic_v1.json'


def boxed_read(model, sample, points, *, no_memory):
    layers, handles = [], []
    for i, layer in enumerate(model.memory_attention.layers):
        handles.append(layer.cross_attn_image.register_forward_hook(
            lambda m, a, o, index=i: layers.append(index)))
    try:
        with decoder_training_mask(model.sam_mask_decoder) as captured:
            output = model.track_step(frame_idx=sample['frame'], is_init_cond_frame=no_memory,
                current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                feat_sizes=sample['sizes'], point_inputs=points, mask_inputs=None,
                output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                track_in_reverse=False, run_mem_encoder=False)
        raw = select_decoder_mask(captured).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    if layers != ([] if no_memory else [0, 1, 2, 3]):
        raise RuntimeError('Wrong oracle memory-attention path')
    return output, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    if not args.gate_only and args.gate is None:
        parser.error('Complete diagnostic requires a passed --gate')
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    base = Path(protocol['drive_root'])
    previous = base / protocol['previous']
    completed = json.loads((previous / 'full/completion.json').read_text())
    sources = {**completed['sources'], **{p: digest(ROOT / p) for p in
        (PROTOCOL, 'scripts/diagnose_oracle_memory_conditioning.py',
         'external/sam2/sam2/utils/misc.py')}}
    if any(digest(ROOT / p) != h for p, h in sources.items()):
        raise RuntimeError('Frozen source differs')
    if digest(previous / 'full/records.json') != protocol['selection_sha256']:
        raise RuntimeError('Prior case records differ')
    prior = {(r['sequence'], r['frame']): r for r in json.loads((previous / 'full/records.json').read_text())}
    setup = json.loads((previous / 'full/setup.json').read_text())
    model_config = json.loads((base / protocol['cache'] / 'setup.json').read_text())['model_config']
    checkpoint = ROOT / protocol['checkpoint']
    if (digest(checkpoint) != setup['checkpoint_sha256'] or
            digest(ROOT / protocol['manifest']) != setup['manifest_sha256']):
        raise RuntimeError('Checkpoint or manifest differs')
    records = []
    if args.gate:
        gate = json.loads((args.gate / 'completion.json').read_text())
        if not gate.get('passed') or gate['scope'] != 'gate' or gate['sources'] != sources:
            raise RuntimeError('Oracle engineering gate differs')
        if digest(args.gate / 'records.json') != gate['records_sha256']:
            raise RuntimeError('Gate records changed')
        records = json.loads((args.gate / 'records.json').read_text())
        expected = [c for c in protocol['cases'] if c['sequence'] in protocol['gate_sequences']]
        if [{k:r[k] for k in ('sequence', 'frame', 'category')} for r in records] != expected:
            raise RuntimeError('Gate case coverage differs')
        for r in records:
            if digest(Path(r['tensor_path'])) != r['tensor_sha256']:
                raise RuntimeError('Gate tensors changed')
    cases = [c for c in protocol['cases'] if not args.gate_only or c['sequence'] in protocol['gate_sequences']]
    pending = [c for c in cases if not args.gate or c['sequence'] not in protocol['gate_sequences']]
    write(out / 'setup.json', {'sources': sources, 'cases': cases,
        'checkpoint_sha256': setup['checkpoint_sha256'], 'manifest_sha256': setup['manifest_sha256']})
    videos = {**load_manifest(ROOT / protocol['manifest'], 'dev'),
              **load_manifest(ROOT / protocol['manifest'], 'test')}
    repo, revision = activate_author_repo()
    from sam2.modeling.sam2_utils import sample_box_points
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for sequence in sorted({c['sequence'] for c in pending}, key=lambda s:int(s[3:])):
        receipt = json.loads((base / protocol['cache'] / sequence / 'receipt.json').read_text())
        payload = load_bank(base / protocol['cache'] / sequence / 'bank.pt', receipt,
            setup['checkpoint_sha256'], setup['manifest_sha256'], model_config)
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
            checkpoint=checkpoint, model_config=model_config), repo)
        if not model.directly_add_no_mem_embed:
            raise RuntimeError('Expected native image/no-memory initialisation path')
        weights = parameter_digest(model)
        frames = videos[sequence]
        with tempfile.TemporaryDirectory(prefix='oracle_box_', dir=out) as folder, torch.inference_mode():
            state, prompt = prepare(model, frames, Path(folder))
            try:
                if prompt != payload['prompt']:
                    raise RuntimeError('Original prompt differs')
                for case in (c for c in pending if c['sequence'] == sequence):
                    frame = case['frame']
                    target = torch.from_numpy((load_label_mask(frames[frame].mask_path, [1]) == 1).copy())[None,None]
                    if not target.any():
                        raise RuntimeError('Oracle box requires a GT-present target')
                    box, labels = sample_box_points(target, noise=0.0)
                    points = {'point_coords': box.float() / torch.tensor([state['video_width'], state['video_height']]) * model.image_size,
                              'point_labels': labels}
                    if model._use_multimask(True, points) != model._use_multimask(False, points):
                        raise RuntimeError('Oracle paths select different multimask policies')
                    sample = cached_read(model, state, frame, payload)
                    _, spans = memory_role_sequence(model, frame, sample['output_dict'], sample['num_frames'])
                    sample = compact_history_read(sample, [s['frame'] for s in spans])
                    bank_sha = fingerprint(sample['output_dict'])['sha256']
                    ordinary = native_read(model, sample)
                    verify_stored(ordinary, payload['outputs']['non_cond_frame_outputs'][frame])
                    _, native_logits = model._get_orig_video_res_output(state, ordinary['pred_masks'])
                    native = score_frame(native_logits, frames, frame)
                    if native != prior[(sequence,frame)]['native']:
                        raise RuntimeError('Native replay/GT scores differ from Run141')
                    outputs, tensors = {}, {'target': target, 'native': ordinary['pred_masks'].detach().clone()}
                    for name, no_memory in [('oracle_memory', False), ('oracle_no_memory', True)]:
                        observed, raw = boxed_read(model, sample, points, no_memory=no_memory)
                        _, logits = model._get_orig_video_res_output(state, observed['pred_masks'])
                        outputs[name] = {'scores': score_frame(logits, frames, frame),
                            'object_score': float(observed['object_score_logits'].item())}
                        tensors[name] = observed['pred_masks'].detach().clone()
                        tensors[name+'_raw'] = raw
                    _, ids, public_logits = model.add_new_points_or_box(state, frame_idx=frame,
                        obj_id=1, box=box.float().reshape(4))
                    if ids != [1] or tensor_hash(public_logits) != outputs['oracle_no_memory']['scores']['prediction_sha256']:
                        raise RuntimeError('No-memory oracle differs from author public box API')
                    if fingerprint(sample['output_dict'])['sha256'] != bank_sha:
                        raise RuntimeError('Oracle read changed native memory')
                    path = out / f'{sequence}_{frame}_masks.pt'
                    save_tensor(tensors, path)
                    records.append({**case, 'native': native, **outputs, 'original_box': box.tolist(),
                        'native_identity': True, 'public_api_identity': True, 'bank_unchanged': True,
                        'tensor_path': str(path), 'tensor_sha256': digest(path)})
                    write(out / 'records.json', records)
                    print(sequence, frame, case['category'], flush=True)
                    del sample, ordinary, native_logits, tensors, observed, raw, logits, public_logits
            finally:
                model.reset_state(state)
        if parameter_digest(model) != weights or fingerprint(payload)['sha256'] != receipt['payload_sha256']:
            raise RuntimeError('Frozen weights or source payload changed')
        del model, payload
        gc.collect()
    if len(records) != len(cases):
        raise RuntimeError('Incomplete oracle diagnostic')
    write(out / 'completion.json', {'passed': True, 'scope': 'gate' if args.gate_only else 'diagnostic',
        'cases': len(records), 'sources': sources, 'records_sha256': digest(out / 'records.json'),
        'sam2_revision': revision, 'training': False,
        'note': 'GT oracle diagnostic only. No oracle outputs were used as temporal memory.'})


if __name__ == '__main__':
    main()

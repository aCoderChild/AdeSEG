"""Capture actual native encoder inputs to diagnose re-encoding drift; no GT oracle."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features, clone_tensors
from scripts.mavos_value_utils import activate_author_repo, atomic_json, atomic_torch_save, prepare
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources, _source_hashes, _tree_digest, _model_digest
from scripts.trace_key_adapter_parity import tensor_difference


def run(out):
    out = validate_output(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    sources = _source_hashes()
    sources['scripts/trace_mask_memory_encoding.py'] = digest(Path(__file__))
    sources['scripts/trace_key_adapter_parity.py'] = digest(ROOT / 'scripts/trace_key_adapter_parity.py')
    identity = {'sources': sources, 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
                'threads': 1, 'torch_version': torch.__version__, 'scope': 'Native seq2 source-frame2 input trace only; no GT oracle or efficacy'}
    if identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Pretrained checkpoint changed')
    atomic_json(out / 'identity.json', identity)
    for name in sources:
        p = out / 'source' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261109)
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('Native revision changed')
    model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
        checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
    model.requires_grad_(False).eval()
    original = model._encode_new_memory
    signature = inspect.signature(original)
    core = _model_digest(model)
    video = load_manifest(MANIFEST, 'dev')['seq2']
    start = prompt_index(video, {'initialization': 'first_labeled_frame'})
    frames = video[start:start + 9]
    latest = {}
    def captured(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        inputs = clone_tensors(dict(bound.arguments))
        output = original(*args, **kwargs)
        latest.clear()
        latest.update(inputs=inputs, output=clone_tensors(output))
        return output
    runs, sample = {}, None
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        for label in ('unhooked', 'captured'):
            directory = Path(temporary) / label
            directory.mkdir()
            state, prompt = prepare(model, frames, directory)
            model._encode_new_memory = captured if label == 'captured' else original
            records = {}
            with torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1]:
                            raise RuntimeError('Trace owner changed')
                        bank = state['output_dict_per_obj'][0]
                        stored = bank['cond_frame_outputs'].get(frame, bank['non_cond_frame_outputs'].get(frame))
                        records[str(frame)] = _tree_digest(stored)
                        if label == 'captured' and frame == 2:
                            sample = clone_tensors(latest)
                            sample['stored'] = clone_tensors(stored)
                finally:
                    iterator.close()
            if label == 'captured':
                recaptured = capture_current_features(model, state, 2)
            model.reset_state(state)
            runs[label] = records
        model._encode_new_memory = original
        if sample is None:
            raise RuntimeError('No source-frame2 capture')
        atomic_torch_save(out / 'native_source_inputs.pt', sample)
        args = sample['inputs']
        stored = sample['stored']
        reconstructed = F.interpolate(stored['pred_masks'].float(), size=(model.image_size, model.image_size),
                                      mode='bilinear', align_corners=False)
        report = {'identity': identity, 'wrapper_native_bank_hashes_equal': runs['unhooked']==runs['captured'],
                  'native_bank_hashes': runs,
                  'high_res_mask_difference': tensor_difference(args['pred_masks_high_res'], reconstructed),
                  'vision_difference': tensor_difference(args['current_vision_feats'], recaptured['vision']),
                  'native_is_mask_from_pts': args['is_mask_from_pts'], 'variants': {}}
        for features_label, vision in [('captured', args['current_vision_feats']), ('recaptured', recaptured['vision'])]:
            for masks_label, high in [('captured', args['pred_masks_high_res']), ('reconstructed', reconstructed)]:
                for mode, context in [('inference', torch.inference_mode()), ('no_grad', torch.no_grad())]:
                    with context:
                        value, positions = original(vision, args['feat_sizes'], high,
                            args['object_score_logits'], args['is_mask_from_pts'])
                    report['variants'][f'{features_label}_{masks_label}_{mode}'] = {
                        'float_native_output': tensor_difference(value, sample['output'][0]),
                        'stored_bank': tensor_difference(value.to(stored['maskmem_features'].dtype), stored['maskmem_features']),
                        'positions': tensor_difference(positions, stored['maskmem_pos_enc'])}
        report['sam2_unchanged'] = _model_digest(model) == core
        report['efficacy_authorized'] = False
        _guard_sources(sources)
        atomic_json(out / 'results.json', report)
        atomic_json(out / 'completion.json', {'trace_complete': True, 'results_sha256': digest(out / 'results.json'), 'efficacy_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        run(args.output_dir)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise

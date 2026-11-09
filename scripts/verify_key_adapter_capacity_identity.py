"""Extend existing SAM2 engineering checks to full memory and object isolation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from datasets.video import load_manifest
from modeling.complementary_memory.training import clone_tensors
from modeling.token_conditioned_key_adapter import NativeTokenConditionedKeyAdapter
from scripts.diagnose_key_adapter_optimization import MEMORY_FIELDS, update
from scripts.mavos_value_utils import activate_author_repo, atomic_json, prepare
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources, _make_heads, _model_digest, _source_hashes, _tree_digest,
)
from scripts.verify_key_adapter_process_resume import require_reference
from scripts.verify_token_conditioned_key_adapter import davis_frames, require_layer_order
from scripts.verify_token_conditioned_key_adapter_runner import _adamw, _samples

DESIGN = ROOT / 'protocols/key_adapter_capacity_identity_v1.json'
REFERENCE = CHECKPOINT_DEFAULT.parents[1] / 'key_adapter_optimization_validity_v1/process_resume_check_v1/reference'


def require_owned_capacity(calls, object_ids, length=9):
    if {c['object_id'] for c in calls} != set(object_ids):
        raise RuntimeError('Observed read owners differ from prompted IDs')
    for object_id in object_ids:
        rows = [c for c in calls if c['object_id'] == object_id]
        require_layer_order(rows, length - 1)
        if [(c['frame'], c['layer']) for c in rows] != [(f, l) for f in range(1, length) for l in range(4)]:
            raise RuntimeError('Missing, repeated or reordered owned frame/layer read')
        if any(c['pointer_keys'] <= 0 or not 0 < c['spatial_keys'] <= 28672 for c in rows):
            raise RuntimeError('Pointer or spatial budget invariant failed')
        if any(c['spatial_keys'] != 28672 for c in rows if c['frame'] >= 7):
            raise RuntimeError('Full seven-frame memory budget was not reached')


def propagate(model, frames, out, heads=None, mode='token', ids=(1,), indexed_mask=None):
    records, raw = [], {}
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        directory = Path(temporary)
        if indexed_mask is None:
            state, prompt = prepare(model, frames, directory)
            if prompt != 0:
                raise RuntimeError('Prompt must be frame0 of the fixed clip')
        else:
            video = directory / 'video'
            video.mkdir()
            for index, image in enumerate(frames):
                shutil.copy2(image, video / f'{index:05d}.jpg')
            state = model.init_state(video_path=str(video), offload_video_to_cpu=True, offload_state_to_cpu=False)
            for object_id in ids:
                if not np.any(indexed_mask == object_id):
                    raise RuntimeError('Prompt has no pixels for a declared ID')
                model.add_new_mask(state, frame_idx=0, obj_id=object_id,
                    mask=torch.from_numpy((indexed_mask == object_id).copy()))
        context = NativeTokenConditionedKeyAdapter(model, heads, mode, inference_state=state) if heads is not None else nullcontext()
        with context as observer, torch.inference_mode():
            iterator = model.propagate_in_video(state, start_frame_idx=0)
            try:
                for frame, returned_ids, logits in iterator:
                    if list(returned_ids) != list(ids) or not torch.isfinite(logits).all():
                        raise RuntimeError('Persistent IDs or finite outputs violated')
                    owned = {}
                    for index, object_id in enumerate(returned_ids):
                        bank = state['output_dict_per_obj'][state['obj_id_to_idx'][object_id]]
                        value = bank['cond_frame_outputs'].get(frame, bank['non_cond_frame_outputs'].get(frame))
                        owned[str(object_id)] = {'output_logits_sha256': tensor_hash(logits[index]),
                            'memory_sha256': _tree_digest({f: value[f] for f in MEMORY_FIELDS})}
                        raw.setdefault(str(object_id), {})[str(frame)] = owned[str(object_id)]['memory_sha256']
                    records.append({'frame': int(frame), 'objects': owned})
            finally:
                iterator.close()
            calls = list(observer.calls) if observer is not None else []
        model.reset_state(state)
    if len(records) != 9:
        raise RuntimeError('Incomplete nine-frame propagation')
    if heads is not None:
        require_owned_capacity(calls, ids)
    return {'records': records, 'raw_object_banks': raw, 'calls': calls}


def final_frame_fixture(model, frames, out):
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        state, prompt = prepare(model, frames, Path(temporary))
        if prompt != 0:
            raise RuntimeError('Fixed prompt changed')
        with torch.inference_mode():
            iterator = model.propagate_in_video(state, start_frame_idx=0)
            try:
                for frame, ids, _ in iterator:
                    if ids != [1]:
                        raise RuntimeError('Training fixture changed owner')
                    if frame == 7:
                        bank = clone_tensors(state['output_dict_per_obj'][0])
                        break
            finally:
                iterator.close()
        if 8 in bank['non_cond_frame_outputs']:
            raise RuntimeError('Query prediction leaked into its own memory fixture')
        samples = _samples(model, state, frames, [8])
        model.reset_state(state)
    return {'bank': bank, 'bank_sha256': _tree_digest(bank), 'samples': samples}


def run(args):
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    design = json.loads(DESIGN.read_text())
    old_identity = json.loads((REFERENCE / 'identity.json').read_text())
    reference = require_reference(REFERENCE, old_identity)
    _guard_sources(old_identity['sources'])
    audit = json.loads((REFERENCE.parent / 'audit_results.json').read_text())
    if not audit['passed'] or audit['efficacy_authorized']:
        raise RuntimeError('Completed engineering resume audit required')
    sources = _source_hashes()
    for relative in ('scripts/verify_key_adapter_capacity_identity.py',
                     'scripts/verify_key_adapter_process_resume.py', 'scripts/diagnose_key_adapter_optimization.py',
                     'scripts/trace_key_adapter_parity.py', 'tests/test_key_adapter_capacity_identity.py',
                     'protocols/key_adapter_capacity_identity_v1.json'):
        sources[relative] = digest(ROOT / relative)
    identity = {'sources': sources, 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
                'manifest_sha256': digest(MANIFEST),
                'reference_results_sha256': digest(REFERENCE / 'results.json'),
                'resume_audit_sha256': digest(REFERENCE.parent / 'audit_results.json'),
                'adapter_files': {m: digest(REFERENCE / f'{m}_final.pt') for m in ('token', 'global')},
                'threads': 1, 'torch_version': torch.__version__, 'python': sys.version,
                'torch_build': torch.__config__.show()}
    if identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Pretrained checkpoint changed')
    if identity['manifest_sha256'] != protocol['data']['manifest_sha256'] or identity['manifest_sha256'] != old_identity['manifest_sha256']:
        raise RuntimeError('Frozen manifest changed')
    atomic_json(out / 'identity.json', identity)
    atomic_json(out / 'protocol_frozen.json', design)
    for relative in sources:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol['training']['seed'])
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('Pinned SAM2 revision changed')
    def model_and_heads():
        model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
            checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
        model.requires_grad_(False).eval()
        heads = {}
        for mode in ('token', 'global'):
            state = torch.load(REFERENCE / f'{mode}_final.pt', map_location='cpu', weights_only=True)
            if _tree_digest(state) != reference['arms'][mode]['final_state_sha256']:
                raise RuntimeError('Adapter state differs from audited reference')
            heads[mode] = _make_heads(model, protocol)
            heads[mode].load_state_dict(state['heads'], strict=True)
            heads[mode].eval()
        return model, heads
    results = {'identity': identity, 'polypgen': {}, 'davis': {}, 'input_file_hashes': {}, 'efficacy_authorized': False}
    videos = load_manifest(MANIFEST, 'dev')
    for name in design['polypgen']['sequences']:
        if name not in protocol['data']['train_sequences']:
            raise RuntimeError('Capacity gate may only read TRAIN PolypGen sequences')
        start = prompt_index(videos[name], {'initialization': 'first_labeled_frame'})
        frames = videos[name][start:start + 9]
        if len(frames) != 9:
            raise RuntimeError('Incomplete fixed clip')
        model, heads = model_and_heads()
        core = _model_digest(model)
        row = {'files': [{'frame': f.frame_index, 'image': digest(f.image_path), 'mask': digest(f.mask_path)} for f in frames], 'arms': {}}
        for frame in frames:
            for path in (frame.image_path, frame.mask_path):
                results['input_file_hashes'][str(path)] = digest(path)
        native = propagate(model, frames, out)
        item = final_frame_fixture(model, frames, out)
        for mode in ('token', 'global'):
            _guard_sources(sources)
            zero = propagate(model, frames, out, _make_heads(model, protocol), mode)
            if zero['records'] != native['records']:
                raise RuntimeError(f'{name}/{mode} native zero-adapter parity differs')
            trained = propagate(model, frames, out, heads[mode], mode)
            if not all(any(c['key_delta_l1'] > 0 for c in trained['calls'] if c['layer'] == layer) for layer in range(4)):
                raise RuntimeError('Audited trained adapter has no effect in a layer')
            optimizer = _adamw(heads[mode], protocol)
            gradients = []
            for _ in range(2):
                loss, _, writes, diagnostics = update(model, heads[mode], optimizer, item, name, mode, protocol)
                if any(c['spatial_keys'] != 28672 for c in diagnostics['memory_read_trace']):
                    raise RuntimeError('Gradient check did not read full native capacity')
                gradients.append({'loss': loss, 'writes': writes, 'diagnostics': diagnostics})
            row['arms'][mode] = {'zero': zero, 'trained': trained, 'full_capacity_gradients': gradients}
        if _model_digest(model) != core:
            raise RuntimeError('SAM2 parameters changed')
        results['polypgen'][name] = row
        atomic_json(out / 'progress.json', results)
        print(json.dumps({'completed_polypgen_capacity': name}), flush=True)
        del model, heads, item
    images, masks = davis_frames(design['davis']['video'], 9)
    labels = np.asarray(Image.open(masks[0]))
    model, heads = model_and_heads()
    core = _model_digest(model)
    if model.non_overlap_masks or model.non_overlap_masks_for_mem_enc:
        raise RuntimeError('Isolation check requires native non-overlap coupling disabled')
    native = propagate(model, images, out, ids=(1, 2), indexed_mask=labels)
    results['davis']['files'] = {'images': [digest(x) for x in images], 'frame0_mask': digest(masks[0])}
    for path in [*images, masks[0]]:
        results['input_file_hashes'][str(path)] = digest(path)
    for mode in ('token', 'global'):
        _guard_sources(sources)
        zero = propagate(model, images, out, _make_heads(model, protocol), mode, (1, 2), labels)
        if zero['records'] != native['records']:
            raise RuntimeError(f'DAVIS/{mode} zero-adapter parity differs')
        joint = propagate(model, images, out, heads[mode], mode, (1, 2), labels)
        reverse = propagate(model, images, out, heads[mode], mode, (2, 1), labels)
        if reverse['raw_object_banks'] != joint['raw_object_banks']:
            raise RuntimeError('Prompt insertion order changed per-object memory')
        isolated = {}
        for object_id in (1, 2):
            single = propagate(model, images, out, heads[mode], mode, (object_id,), labels)
            if single['raw_object_banks'][str(object_id)] != joint['raw_object_banks'][str(object_id)]:
                raise RuntimeError('Another object altered an isolated object memory bank')
            isolated[str(object_id)] = single
        results['davis'][mode] = {'zero': zero, 'joint': joint, 'reverse': reverse, 'isolated': isolated}
        atomic_json(out / 'progress.json', results)
        print(json.dumps({'completed_davis_identity': mode}), flush=True)
    if _model_digest(model) != core:
        raise RuntimeError('SAM2 parameters changed')
    _guard_sources(sources)
    if any(digest(Path(path)) != expected for path, expected in results['input_file_hashes'].items()):
        raise RuntimeError('An input image/mask changed during verification')
    if digest(CHECKPOINT_DEFAULT) != identity['checkpoint_sha256'] or digest(MANIFEST) != identity['manifest_sha256']:
        raise RuntimeError('Checkpoint or manifest changed during verification')
    if any(digest(REFERENCE / f'{mode}_final.pt') != expected for mode, expected in identity['adapter_files'].items()):
        raise RuntimeError('An adapter checkpoint changed during verification')
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'completion.json', {'passed': True, 'efficacy_authorized': False,
                                         'results_sha256': digest(out / 'results.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        run(args)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise

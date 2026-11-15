"""Source-bound fixed-read study helpers for exploratory source-group data."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from scripts.freeze_source_group_runtime_v1 import current_identity, validate_source_freeze
from scripts.memory_token_io import digest
from scripts.prepare_source_group_refiner_manifest_v1 import SEQUENCES, VALIDATION, identity_hash, verify_capture_metadata
from scripts.run_source_group_capture_v1 import verify_short_gate
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('region', 'global', 'no_contrast', 'mask_only')
SEED = 20261115
EPOCHS = 10
PARAMETERS = 48005
LOSS = 'mean(BCEWithLogits) + soft Dice'
SOURCES = ('scripts/source_group_refiner_shared_v1.py', 'scripts/train_source_group_refiner_v1.py',
           'scripts/verify_source_group_refiner_v1.py', 'scripts/score_source_group_refiner_v1.py',
           'scripts/prepare_source_group_refiner_manifest_v1.py', 'scripts/source_group_training_label_guard_v1.py',
           'scripts/source_group_label_guard_v1.py', 'scripts/train_multilevel_refiner_seq4_empty_v2.py',
           'scripts/train_region_mixed_loss_v1.py', 'scripts/memory_token_io.py', 'datasets/io.py', 'datasets/video.py',
           'modeling/foreground_conditioned_refiner.py', 'modeling/multilevel_contrast_refiner.py',
           'modeling/lean_channel_attention.py', 'protocols/SOURCE_GROUP_REFINER_FIXED_READ_V1.md')


def read(path):
    return json.loads(Path(path).read_text())


def dependencies():
    return {path: digest(ROOT/path) for path in SOURCES}


def model_for(arm):
    if arm not in ARMS:
        raise ValueError('Unknown matched arm')
    model = ForegroundConditionedRefiner(gate_mode='global' if arm == 'global' else 'region',
                                        contrast_head=arm != 'no_contrast')
    model._mask_only = arm == 'mask_only'
    if sum(x.numel() for x in model.parameters()) != PARAMETERS:
        raise ValueError('Matched model parameter count differs')
    return model


def schedule(samples):
    train = sorted((x for x in samples if x['split'] == 'train'), key=lambda x: (x['sequence'], x['frame']))
    return [[[train[i]['sequence'], train[i]['frame']] for i in np.random.default_rng(SEED+epoch).permutation(len(train))]
            for epoch in range(EPOCHS)]


def initial_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_prepared(folder, *, check_tensors=False):
    path = Path(folder)/'manifest.json'; m = read(path); done = read(Path(folder)/'completion.json')
    train_paths, blocked_paths = mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    if (not done.get('passed') or done.get('manifest_sha256') != digest(path) or not m.get('passed')
            or m.get('runner_sha256') != digest(ROOT/'scripts/prepare_source_group_refiner_manifest_v1.py')
            or m.get('feature_pipeline_sha256') != identity_hash(m['feature_pipeline'])
            or m.get('label_guard', {}).get('open_attempts') != 0
            or m.get('label_guard', {}).get('blocked_paths') != len(train_paths | blocked_paths)):
        raise ValueError('Prepared source-group manifest differs')
    for key in ('preflight', 'source_freeze', 'short_gate', 'checkpoint', 'capture_map'):
        if digest(m[key+'_path']) != m[key+'_sha256']:
            raise ValueError(f'{key} changed after manifest freeze')
    pre = Path(m['preflight_path']).parent; freeze_path = Path(m['source_freeze_path'])
    validate_source_freeze(read(freeze_path), read(freeze_path.parent/'completion.json'), digest(freeze_path),
                           current_identity(pre, Path(m['reference_capture_path']), Path(m['checkpoint_path'])))
    gate = read(m['short_gate_path']); preflight = read(m['preflight_path'])
    verify_short_gate(gate, m['preflight_sha256'], m['source_freeze_sha256'],
                      digest(ROOT/'scripts/run_source_group_capture_v1.py'),
                      digest(ROOT/'scripts/audit_source_group_capture_v1.py'), gate['identity']['runtime_environment'])
    if set(m['sequences']) != set(SEQUENCES):
        raise ValueError('Unexpected source-group coverage')
    for seq, evidence in m['sequences'].items():
        for kind in ('reference_receipt', 'capture_receipt', 'audit'):
            if digest(evidence[kind+'_path']) != evidence[kind+'_sha256']:
                raise ValueError(f'{seq}: audited evidence changed')
        rpath = Path(evidence['reference_receipt_path']); cpath = Path(evidence['capture_receipt_path'])
        ref, cap, audit = read(rpath), read(cpath), read(evidence['audit_path'])
        spec = preflight['sequences'][seq]
        verify_capture_metadata(seq, spec, ref, cap, audit, preflight_sha=m['preflight_sha256'],
                                freeze_sha=m['source_freeze_sha256'], gate_sha=m['short_gate_sha256'],
                                identity=gate['identity'])
        if (audit['reference_receipt_sha256'] != digest(rpath) or audit['capture_receipt_sha256'] != digest(cpath)
                or Path(audit['reference_receipt_path']).resolve() != rpath.resolve()
                or Path(audit['capture_receipt_path']).resolve() != cpath.resolve()):
            raise ValueError(f'{seq}: audit no longer binds full receipts')
        archive = Path(spec['source'])/seq
        if digest(archive/'prompt_refined.png') != spec['archived_prompt_mask_sha256']:
            raise ValueError(f'{seq}: archived prompt changed')
        with Image.open(archive/'prompt_refined.png') as image:
            prompt = np.asarray(image.convert('L')) > 0
        for path, receipt in ((rpath, ref), (cpath, cap)):
            conditioning = path.parent/'conditioning_mask.png'
            if digest(conditioning) != receipt['conditioning_sha256']:
                raise ValueError(f'{seq}: conditioning changed')
            with Image.open(conditioning) as image:
                if not np.array_equal(np.asarray(image.convert('L')) > 0, prompt):
                    raise ValueError(f'{seq}: conditioning no longer matches prompt')
        for row in ref['rows']:
            for path in (rpath, cpath):
                if digest(path.parent/f"{row['frame']:06d}.png") != row['mask_sha256']:
                    raise ValueError(f'{seq}: frozen trajectory mask changed')
    samples = m['samples']; keys = [(x['sequence'], x['frame']) for x in samples]
    if (len(samples) != 214 or len(set(keys)) != 214 or sum(x['split']=='train' for x in samples) != 164
            or sum(x['split']=='validation' for x in samples) != 50
            or any(x['capture_device']!='mps' or x['split'] != ('validation' if x['sequence'] in VALIDATION else 'train') for x in samples)):
        raise ValueError('Prepared samples or split differ')
    for seq in SEQUENCES:
        if sorted(x['frame'] for x in samples if x['sequence']==seq) != m['sequences'][seq]['targets']:
            raise ValueError(f'{seq}: selected targets differ')
    for item in samples:
        if any(digest(item[k]) != item[k+'_sha256'] for k in ('image', 'feature', 'snapshot')):
            raise ValueError('Frozen sample input changed')
        if check_tensors:
            read_capture(item)
    return m, digest(path)


def ensure_target(item, shape):
    from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target
    return load_target(item, shape)


def author_source():
    return author_sha256()

"""Label-free fold ownership and warm-reference validation for video follow-up.

This module does not run inference, choose a checkpoint from held-out scores,
or prove equality of unrecorded model state.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from datasets.video import load_manifest
from scripts.crossfit_observer_capture_shared_v1 import DATA, EXISTING, phase_artifact
from scripts.crossfit_training_contract_v1 import fold_split
from scripts.memory_token_io import digest

NO_INITIALIZATION = frozenset(('seq1', 'seq7'))
PROPOSED_ARM = 'foreground_background_asinh'


def require(value, message):
    if not value:
        raise ValueError(message)


def read_bound(path, expected):
    path = Path(path).resolve(strict=True)
    require(digest(path) == expected, f'Input changed: {path.name}')
    value = json.loads(path.read_text())
    require(digest(path) == expected, f'Input changed during reading: {path.name}')
    return value


def heldout_fold(sequence):
    owners = [fold for fold in range(1, 6) if sequence in fold_split(fold)[2]]
    require(len(owners) == 1, 'Patient must belong to exactly one test fold')
    fold = owners[0]
    train, validation, test = fold_split(fold)
    require(sequence in test and sequence not in train and sequence not in validation,
            'Patient is not held out from training and validation')
    return fold


def load_heldout_checkpoint(batch, preflight_path, preflight, preflight_sha, sequence):
    """Load the predeclared arm only after binding its own held-out fold."""
    from scripts.warm_crossfit_fold_training_v1 import load_preflight, trained_model
    require(sequence not in NO_INITIALIZATION, 'No initialization for this patient')
    checked_plan, checked_sha = load_preflight(preflight_path)
    require(checked_sha == preflight_sha and checked_plan == preflight,
            'Provided checkpoint preflight differs from bound bytes')
    batch = Path(batch).resolve(strict=True)
    fold = heldout_fold(sequence)
    train, validation, test = fold_split(fold)
    definition = preflight['folds'][fold - 1]
    require(definition.get('fold') == fold and definition.get('train') == sorted(train) and
            definition.get('validation') == sorted(validation) and definition.get('test') == sorted(test),
            'Frozen checkpoint fold differs')
    folder = batch / 'train' / f'fold{fold}' / PROPOSED_ARM
    receipt_path = folder / 'completion.json'
    receipt_sha = phase_artifact(batch, f'train_f{fold}_{PROPOSED_ARM}', receipt_path)
    receipt = read_bound(receipt_path, receipt_sha); checkpoint = folder / 'checkpoint.pt'
    checkpoint_sha = digest(checkpoint)
    require(receipt.get('passed') is True and receipt.get('fold') == fold and
            receipt.get('arm') == PROPOSED_ARM and receipt.get('preflight_sha256') == preflight_sha and
            receipt.get('checkpoint_sha256') == checkpoint_sha and receipt.get('training_steps', 0) > 0 and
            receipt.get('label_guard', {}).get('blocked_open_attempts') == 0,
            'Held-out checkpoint receipt differs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(digest(checkpoint) == checkpoint_sha and digest(receipt_path) == receipt_sha and
            payload.get('training_steps') == receipt['training_steps'], 'Fold checkpoint changed during loading')
    model = trained_model(payload, arm=PROPOSED_ARM, fold=fold, preflight=preflight,
                          preflight_sha=preflight_sha)
    return model, {'fold': fold, 'arm': PROPOSED_ARM, 'checkpoint_path': str(checkpoint),
                   'checkpoint_sha256': checkpoint_sha, 'receipt_path': str(receipt_path),
                   'receipt_sha256': receipt_sha}


def validate_reference(sequence, reference_path, reference_sha, *, expected_frames,
                       expected_start, expected_mask_sha, expected_state_sha, expected_shapes):
    """Verify every retained warm mask and the named provenance fields.

    The historical references did not save each compact state tensor; exact
    mask and named trajectory verification must not be called full-state parity.
    """
    path = Path(reference_path).resolve(strict=True)
    record = read_bound(path, reference_sha)
    identity = record.get('identity', {})
    state = record.get('model_state_sha256', identity.get('model_state_sha256'))
    require(record.get('passed') is True and record.get('mode') == 'reference' and
            record.get('sequence') == sequence and record.get('prompt_frame') == expected_start and
            ((expected_mask_sha is None and sequence == 'seq6' and 'conditioning_sha256' not in record) or
             (expected_mask_sha is not None and record.get('conditioning_sha256') == expected_mask_sha)) and
            state == expected_state_sha and
            (record.get('label_guard', {}).get('open_attempts') == 0 or
             (sequence == 'seq6' and expected_mask_sha is None and 'label_guard' not in record)),
            'Warm reference provenance differs')
    rows = record.get('rows', [])
    require(len(expected_shapes) == expected_frames and
            [r.get('frame') for r in rows] == list(range(expected_start, expected_frames)),
            'Warm reference frame coverage differs')
    for row in rows:
        file = path.parent / f"{row['frame']:06d}.png"
        require(digest(file) == row['mask_sha256'], 'Warm reference mask differs')
        with Image.open(file) as image:
            require(image.format == 'PNG' and image.size == tuple(expected_shapes[row['frame']]) and
                    set(np.unique(np.asarray(image.convert('L')))).issubset({0, 255}),
                    'Warm reference mask geometry or encoding differs')
        require(digest(file) == row['mask_sha256'], 'Warm reference mask changed during decoding')
    require(digest(path) == reference_sha, 'Warm reference receipt changed')
    for key in ('positions', 'selection', 'anchor_events', 'token_window_events'):
        require(isinstance(record.get(key), list), f'Missing warm reference trace: {key}')
    return record


def old_sequence_inputs(sequence, manifest, expected_manifest_sha):
    """Resolve one of nine historical warm references without annotation reads."""
    require(sequence in manifest['sequences'], 'Not a historical warm patient')
    require(read_bound(EXISTING / 'manifest.json', expected_manifest_sha) == manifest,
            'Historical prepared manifest object differs from bound bytes')
    preflight = read_bound(manifest['preflight_path'], manifest['preflight_sha256'])
    spec = preflight['sequences'][sequence]; archive = Path(spec['source']) / sequence
    prompt = read_bound(archive / 'prompt.json', spec['archived_prompt_sha256'])
    mask_path = archive / 'prompt_refined.png'
    require(digest(mask_path) == spec['archived_prompt_mask_sha256'], 'Archived prompt mask changed')
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L')) > 0
    require(digest(mask_path) == spec['archived_prompt_mask_sha256'], 'Prompt mask changed during read')
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}; frames = videos[sequence]
    require(len(frames) == spec['frames'] and len(spec['image_sha256']) == len(frames) and
            all(digest(frame.image_path) == sha for frame, sha in zip(frames, spec['image_sha256'], strict=True)) and
            prompt.get('first_detection_index') == spec['prompt_frame'], 'Historical video inputs differ')
    evidence = manifest['sequences'][sequence]
    shapes = []
    for frame in frames:
        with Image.open(frame.image_path) as image:
            shapes.append(image.size)
    require(all(digest(frame.image_path) == sha for frame, sha in zip(frames, spec['image_sha256'], strict=True)),
            'Historical image changed during geometry decoding')
    record = validate_reference(sequence, evidence['reference_receipt_path'], evidence['reference_receipt_sha256'],
        expected_frames=len(frames), expected_start=spec['prompt_frame'],
        expected_mask_sha=spec['archived_prompt_mask_sha256'],
        expected_state_sha=manifest['feature_pipeline']['model_state_sha256'], expected_shapes=shapes)
    capture = read_bound(evidence['capture_receipt_path'], evidence['capture_receipt_sha256'])
    audit = read_bound(evidence['audit_path'], evidence['audit_sha256'])
    identity = record.get('identity', {})
    require(capture.get('passed') is True and capture.get('mode') == 'capture' and
            capture.get('sequence') == sequence and capture.get('identity') == identity and
            audit.get('passed') is True and audit.get('full_sequence') is True and
            audit.get('sequence') == sequence and audit.get('identity') == identity and
            audit.get('reference_receipt_path') == evidence['reference_receipt_path'] and
            audit.get('reference_receipt_sha256') == evidence['reference_receipt_sha256'] and
            audit.get('capture_receipt_path') == evidence['capture_receipt_path'] and
            audit.get('capture_receipt_sha256') == evidence['capture_receipt_sha256'] and
            audit.get('auditor_sha256') == digest(Path(__file__).resolve().parents[1] /
                                                   'scripts/audit_source_group_capture_v1.py') and
            audit.get('label_guard', {}).get('open_attempts') == 0 and
            identity.get('checkpoint_sha256') == manifest['checkpoint_sha256'] and
            identity.get('preflight_sha256') == manifest['preflight_sha256'] and
            identity.get('source_freeze_sha256') == manifest['source_freeze_sha256'] and
            all(capture.get(k) == record.get(k) for k in
                ('rows', 'positions', 'selection', 'anchor_events', 'token_window_events')),
            'Historical warm capture, source lineage or recorded trajectory differs')
    for row in capture['rows']:
        require(digest(Path(evidence['capture_receipt_path']).parent / f"{row['frame']:06d}.png") ==
                row['mask_sha256'], 'Historical observer mask changed')
    require(record.get('prompt_sha256') == spec['archived_prompt_sha256'] and
            digest(EXISTING / 'manifest.json') == expected_manifest_sha, 'Historical prompt or manifest differs')
    return {'frames': frames, 'image_sha256': spec['image_sha256'], 'image_sizes': shapes, 'start': spec['prompt_frame'],
        'mask': mask, 'box': prompt['box'], 'points': prompt['points'], 'roi_xyxy': None,
        'reference_path': evidence['reference_receipt_path'], 'reference_sha256': evidence['reference_receipt_sha256'],
        'reference': record}


def fresh_sequence_inputs(sequence, aggregate, certificate, index, warm):
    """Resolve an audited V2 warm reference, including special seq6 reuse."""
    from scripts.run_crossfit_warm_video_v2 import video_inputs
    observations = [x for x in aggregate['observations'] if x['sequence'] == sequence]
    require(observations, 'Fresh sequence has no verified captures')
    first = observations[0]
    for item in observations:
        require(item['reference_receipt_path'] == first['reference_receipt_path'] and
                item['reference_receipt_sha256'] == first['reference_receipt_sha256'] and
                item['roi_xyxy'] == first['roi_xyxy'], 'Fresh reference lineage differs')
    if sequence == 'seq6':
        require(first['reference_kind'] == 'historical_warm_reference' and
                first['reference_receipt_path'] == certificate['seq6_historical_reference']['reference_path'] and
                first['reference_receipt_sha256'] == certificate['seq6_historical_reference']['reference_sha256'],
                'Seq6 reference differs from certified historical trajectory')
    else:
        require(first['reference_kind'] == 'matched_warm_reference', 'Fresh warm reference kind differs')
        reference = Path(first['reference_receipt_path']).resolve(strict=True)
        require(phase_artifact(reference.parent.parent.parent, f'{sequence}_reference', reference) ==
                first['reference_receipt_sha256'], 'Fresh warm reference phase differs')
    frames, records, start, _, mask, box, points, roi = video_inputs(sequence, index, warm)
    require(roi == first['roi_xyxy'], 'Fresh reference viewport differs')
    reference_path = Path(first['reference_receipt_path']).resolve(strict=True)
    receipt = read_bound(reference_path, first['reference_receipt_sha256'])
    shapes = []
    for frame in frames:
        with Image.open(frame.image_path) as image:
            shapes.append(image.size)
    require(all(digest(frame.image_path) == row['image_sha256'] for frame, row in zip(frames, records, strict=True)),
            'Fresh image changed during geometry decoding')
    record = validate_reference(sequence, reference_path, first['reference_receipt_sha256'],
        expected_frames=len(frames), expected_start=start,
        expected_mask_sha=None if sequence == 'seq6' else receipt['conditioning_sha256'],
        expected_state_sha=warm['old_identity']['model_state_sha256'], expected_shapes=shapes)
    if sequence == 'seq6':
        gate = certificate['seq6_gate_capture']; gate_path = Path(gate['receipt_path']).resolve(strict=True)
        require(phase_artifact(gate_path.parent.parent.parent, 'seq6_capture', gate_path) == gate['receipt_sha256'],
                'Seq6 gate capture phase differs')
        gate_record = read_bound(gate_path, gate['receipt_sha256'])
        require(gate_record.get('passed') is True and gate_record.get('sequence') == 'seq6' and
                gate_record.get('mode') == 'capture' and gate_record.get('label_guard', {}).get('open_attempts') == 0 and
                record.get('prompt_sha256') == warm['fresh_prompts']['seq6']['prompt_metadata_sha256'] and
                all(gate_record[k] == record[k] for k in
                    ('rows', 'positions', 'selection', 'anchor_events', 'token_window_events')),
                'Seq6 historical trajectory or gate differs')
        conditioning = gate_path.parent / 'conditioning_mask.png'; conditioning_sha = gate_record['conditioning_sha256']
    else:
        conditioning = reference_path.parent / 'conditioning_mask.png'; conditioning_sha = record['conditioning_sha256']
    require(digest(conditioning) == conditioning_sha, 'Fresh conditioning file changed')
    with Image.open(conditioning) as image:
        saved = np.asarray(image.convert('L')) > 0
    require(digest(conditioning) == conditioning_sha and np.array_equal(saved, mask),
            'Fresh reference conditioning differs from archived prompt')
    if sequence == 'seq23':
        for row in record['rows']:
            with Image.open(reference_path.parent / f"{row['frame']:06d}.png") as image:
                require(not np.asarray(image.convert('L'))[:, :roi[0]].any(), 'Seq23 reference escaped viewport')
    return {'frames': frames, 'image_sha256': [x['image_sha256'] for x in records], 'image_sizes': shapes, 'start': start,
            'mask': mask, 'box': box, 'points': points, 'roi_xyxy': roi,
            'reference_path': str(reference_path), 'reference_sha256': first['reference_receipt_sha256'],
            'reference': record}

"""Contracts for exploratory, patient-grouped cross-fit training.

This module does not launch training or authorize reuse of feature captures.
It supplies a fold-scoped mask guard, a deterministic patient-balanced update
schedule, and viewport-aware training targets.  The guard protects CPython
``open`` events; it is not a native-code sandbox.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.preflight_exploratory_crossfit_capture_v2 import FOLDS, NOINIT
from scripts.project_archived_label_hashes_v1 import project_label_hashes
from scripts.memory_token_io import digest


def fold_split(fold):
    """Return immutable train/validation/test groups from the frozen plan."""
    if type(fold) is not int or not 1 <= fold <= len(FOLDS):
        raise ValueError('Fold must be an integer from 1 to 5')
    population = [s for group in FOLDS for s in group]
    if len(population) != 23 or len(set(population)) != 23 or set(population) != {f'seq{i}' for i in range(1, 24)}:
        raise ValueError('Frozen patient coverage differs')
    test = frozenset(FOLDS[fold - 1])
    validation = frozenset(FOLDS[fold % len(FOLDS)])
    train = frozenset(population) - test - validation
    if train & test or train & validation or test & validation:
        raise ValueError('Fold group overlap')
    return train, validation, test


def patient_balanced_schedule(samples, fold, seed, epochs, updates_per_patient=12):
    """Equalize optimizer-step counts by patient without looking at labels.

    Each sequence has a seeded shuffled read stream that continues across
    epochs; when exhausted it is reshuffled. Each initialized training sequence
    has exactly the same number of updates in every epoch. A separate seeded
    permutation mixes the resulting updates across patients. This can repeat
    short sequences and subsample longer sequences in an individual epoch; it
    is not the same as one full weighted-batch gradient step, particularly
    with Adam. The seed and order must be frozen and reused for every arm.
    """
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError('Invalid seed or epoch count')
    if type(updates_per_patient) is not int or updates_per_patient < 1:
        raise ValueError('Invalid per-patient update count')
    train, _, _ = fold_split(fold)
    by_sequence = {}
    all_keys = set()
    for item in samples:
        key = (item['sequence'], item['frame'])
        if (key[0] not in {f'seq{i}' for i in range(1, 24)} or key[0] in NOINIT
                or type(key[1]) is not int or key[1] < 0 or key in all_keys):
            raise ValueError('Invalid or duplicate sample identity')
        all_keys.add(key)
        if key[0] in train:
            by_sequence.setdefault(key[0], []).append(key)
    expected = train - frozenset(NOINIT)
    if set(by_sequence) != expected or any(not rows for rows in by_sequence.values()):
        raise ValueError('Initialized training sequence coverage differs')
    by_sequence = {key: sorted(value) for key, value in sorted(by_sequence.items())}
    generators = {sequence: np.random.default_rng(np.random.SeedSequence([seed, fold, int(sequence[3:])]))
                  for sequence in by_sequence}
    bags = {sequence: [] for sequence in by_sequence}
    orders = []
    for epoch in range(epochs):
        entries = []
        for sequence, rows in by_sequence.items():
            picks = []
            while len(picks) < updates_per_patient:
                if not bags[sequence]:
                    bags[sequence] = [rows[int(i)] for i in generators[sequence].permutation(len(rows))]
                take = min(updates_per_patient - len(picks), len(bags[sequence]))
                picks.extend(bags[sequence][:take])
                del bags[sequence][:take]
            entries.extend(picks)
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold, epoch, 0xA17]))
        order = [list(entries[int(i)]) for i in rng.permutation(len(entries))]
        if Counter(seq for seq, _ in order) != Counter({s: updates_per_patient for s in by_sequence}):
            raise ValueError('Patient update counts differ')
        orders.append(order)
    return orders


def mask_path_sets(manifest, train_sequences):
    """Derive training and protected mask paths without opening mask files."""
    manifest = Path(manifest)
    allowed, blocked, seen_sequences = set(), set(), set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen_sequences.add(row['video_id'])
        path = Path(row['mask'])
        if not path.is_absolute():
            path = manifest.parent / path
        target = allowed if row['video_id'] in train_sequences and row['video_id'] not in NOINIT else blocked
        target.add(os.path.realpath(path))
    if seen_sequences != {f'seq{i}' for i in range(1, 24)} or not allowed or not blocked or allowed & blocked:
        raise ValueError('Mask path guard sets are invalid')
    return allowed, blocked


def install_fold_training_guard(manifest, fold):
    """Block every non-training and no-initialization mask in this process."""
    train, _, _ = fold_split(fold)
    allowed, blocked = mask_path_sets(manifest, train)
    state = {'train_paths': len(allowed), 'blocked_paths': len(blocked), 'blocked_open_attempts': 0}

    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) in blocked:
                state['blocked_open_attempts'] += 1
                raise PermissionError('Non-training or no-initialization mask open blocked')

    sys.addaudithook(audit)
    return state


def aligned_target(label_path, sequence, video_shape, source_shape):
    """Decode a full-source mask and align it with the frozen capture view.

    Seq23 is the sole retrospective cropped view. Other sequences must match
    the full source geometry exactly. No resizing or label-derived decisions
    are made here; resizing of logits uses the existing native SAM2 routine.
    Caller must verify label hashes before and after this function.
    """
    h, w = map(int, source_shape)
    if min(h, w) <= 0 or len(video_shape) != 2:
        raise ValueError('Invalid source geometry')
    mask = load_label_mask(Path(label_path), [1])
    if mask.shape != (h, w):
        raise ValueError('Label differs from original source geometry')
    if sequence == 'seq23':
        x0, y0, x1, y1 = map(int, VIEWPORT)
        if (h, w) != (720, 1280) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('Seq23 source viewport differs')
        mask = mask[y0:y1, x0:x1]
    if tuple(mask.shape) != tuple(video_shape):
        raise ValueError('Label and capture-view geometry differ')
    return torch.from_numpy((mask == 1).astype(np.float32))[None, None]


def training_label_hashes(fold, observations, source_frames):
    """Project hashes solely for authorized training observations.

    ``frame`` and metadata locator are ordered-frame ordinals. The original
    dataset frame index is a separate field and is checked against the frozen
    plan. This routine does not open mask files or parse archived outcomes.
    Capture/audit receipts and the plan itself must already be verified by
    the caller. No scoring phase is implemented here.
    """
    train, _, _ = fold_split(fold)
    groups = {}
    seen = set()
    for item in observations:
        sequence = item['sequence']
        if sequence not in train or sequence in NOINIT:
            raise PermissionError('Only initialized fold-training sequences are authorized')
        ordinal = item['frame']
        if (type(ordinal) is not int or ordinal < 0 or
                type(item['label_metadata_locator']) is not int or
                item['label_metadata_locator'] != ordinal or
                ordinal >= len(source_frames[sequence])):
            raise ValueError('Historical metadata ordinal differs')
        key = (sequence, ordinal)
        if key in seen:
            raise ValueError('Duplicate training observation')
        seen.add(key)
        frame = source_frames[sequence][ordinal]
        if (frame['frame'] != ordinal or
                frame['source_frame_index'] != item['source_frame_index'] or
                frame['image_sha256'] != item['image_sha256']):
            raise ValueError('Training observation differs from frozen source frame')
        binding = {'path': item['label_metadata_path'], 'sha256': item['label_metadata_sha256']}
        if sequence in groups and groups[sequence]['binding'] != binding:
            raise ValueError('Historical metadata binding differs within sequence')
        group = groups.setdefault(sequence, {'binding': binding, 'ordinals': []})
        group['ordinals'].append(ordinal)
    if not groups:
        raise ValueError('No authorized training observations')
    output = {}
    for sequence, group in groups.items():
        hashes = project_label_hashes(group['binding'], len(source_frames[sequence]), group['ordinals'])
        output.update({(sequence, ordinal): value for ordinal, value in hashes.items()})
    return output


def checked_aligned_training_target(label_path, expected_sha256, sequence, video_shape, source_shape):
    """Check source bytes around decoding; call only inside a training guard."""
    if digest(label_path) != expected_sha256:
        raise ValueError('Training label hash differs before decoding')
    target = aligned_target(label_path, sequence, video_shape, source_shape)
    if digest(label_path) != expected_sha256:
        raise ValueError('Training label changed during decoding')
    return target


def restore_view_mask(mask, sequence, source_shape):
    """Restore a binary view prediction to source coordinates for reporting.

    Pixels outside the fixed seq23 viewport are filled with background; this
    does not establish that the model evaluated those pixels.
    """
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isin(array, (0, 1, False, True)).all():
        raise ValueError('Prediction must be a two-dimensional binary mask')
    h, w = map(int, source_shape)
    if sequence != 'seq23':
        if (h, w) != array.shape:
            raise ValueError('Full-view mask geometry differs')
        return array.astype(bool, copy=True)
    x0, y0, x1, y1 = map(int, VIEWPORT)
    if (h, w) != (720, 1280) or array.shape != (y1 - y0, x1 - x0):
        raise ValueError('Seq23 view or source geometry differs')
    restored = np.zeros((h, w), dtype=bool)
    restored[y0:y1, x0:x1] = array.astype(bool)
    return restored


def validate_warm_index_structure(plan, old_manifest, aggregate, *, index_sha256,
                                  warm_preflight_sha256, warm_gate_path, warm_gate_sha256,
                                  certificate_path, certificate_sha256):
    """Check the 214+143 index relationships without authorizing training.

    This is a structural check, not a provenance certificate: the caller must
    separately verify plan, gate and aggregate file hashes, reconstruct the
    compatibility audit, check every receipt and tensor hash, and enforce
    fold-specific mask authorization before using the returned identities.
    No label, tensor, or other file is opened here.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def valid_sha(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

    require(all(valid_sha(x) for x in (index_sha256, warm_preflight_sha256, warm_gate_sha256,
                                       certificate_sha256)),
            'Invalid plan or gate digest')
    require(aggregate.get('passed') is True and aggregate.get('plan_sha256') == index_sha256 and
            aggregate.get('index_plan_sha256') == index_sha256 and
            aggregate.get('warm_preflight_sha256') == warm_preflight_sha256 and
            aggregate.get('warm_gate_sha256') == warm_gate_sha256 and
            aggregate.get('warm_gate_path') == str(warm_gate_path) and
            aggregate.get('certificate_path') == str(certificate_path) and
            aggregate.get('certificate_sha256') == certificate_sha256 and
            aggregate.get('existing_manifest_sha256') == plan['existing_manifest_sha256'] and
            aggregate.get('existing_feature_pipeline_sha256') == plan['feature_pipeline_sha256'] and
            old_manifest.get('feature_pipeline_sha256') == plan['feature_pipeline_sha256'],
            'Warm aggregate identity differs')
    old_rows, new_rows = old_manifest['samples'], aggregate['observations']
    require(len(old_rows) == len(plan['existing_reads']) == 214 and
            len(new_rows) == len(plan['fresh_reads']) == 143, 'Warm index read counts differ')
    old_plan = {(x['sequence'], x['frame']): x for x in plan['existing_reads']}
    new_plan = {(x['sequence'], x['frame']): x for x in plan['fresh_reads']}
    old_map = {(x['sequence'], x['frame']): x for x in old_rows}
    new_map = {(x['sequence'], x['frame']): x for x in new_rows}
    require(len(old_plan) == len(old_map) == 214 and len(new_plan) == len(new_map) == 143 and
            set(old_map) == set(old_plan) and set(new_map) == set(new_plan) and
            not set(old_map) & set(new_map), 'Warm index identities differ or overlap')
    for key, row in old_map.items():
        sequence, frame = key
        expected = old_plan[key]
        frames = plan['source_frames'][sequence]
        require(type(frame) is int and 0 <= frame < len(frames), 'Historical frame ordinal differs')
        source = frames[frame]
        require(source['frame'] == frame and row['source_frame_index'] == source['source_frame_index'] and
                row['image_sha256'] == expected['image_sha256'] == source['image_sha256'] and
                all(row[name] == expected[name + '_path'] and
                    row[name + '_sha256'] == expected[name + '_sha256']
                    for name in ('feature', 'snapshot')),
                'Historical capture identity differs')
    for (sequence, frame), row in new_map.items():
        expected = new_plan[(sequence, frame)]
        frames = plan['source_frames'][sequence]
        require(type(frame) is int and 0 <= frame < len(frames), 'Warm frame ordinal differs')
        source = frames[frame]
        require(source['frame'] == frame and row['source_frame_index'] == source['source_frame_index'] ==
                expected['source_frame_index'] and row['image_sha256'] == source['image_sha256'] ==
                expected['image_sha256'] and row['label_metadata_locator'] == frame,
                'Warm source-frame identity differs')
        prompt = plan['prompts'][sequence]
        kind = 'historical_warm_reference' if sequence == 'seq6' else 'matched_warm_reference'
        require(row['view'] == prompt['view'] and row['roi_xyxy'] == prompt['prompt_provenance']['roi_xyxy'] and
                row.get('reference_kind') == kind and
                row.get('warm_gate_path') == str(warm_gate_path) and
                row.get('warm_gate_sha256') == warm_gate_sha256 and
                row.get('certificate_path') == str(certificate_path) and
                row.get('certificate_sha256') == certificate_sha256,
                'Warm view, reference or gate differs')
        for stem in ('label_metadata', 'reference_receipt', 'capture_receipt', 'sequence_audit'):
            require(isinstance(row.get(stem + '_path'), str) and bool(row[stem + '_path']) and
                    valid_sha(row.get(stem + '_sha256')), f'Warm {stem} binding differs')
        for stem in ('feature', 'snapshot'):
            require(isinstance(row.get(stem + '_file'), str) and bool(row[stem + '_file']) and
                    valid_sha(row.get(stem + '_sha256')), f'Warm {stem} tensor binding differs')
        if sequence == 'seq6':
            require(row['sequence_audit_path'] == str(warm_gate_path) and
                    row['sequence_audit_sha256'] == warm_gate_sha256,
                    'Seq6 reuse must bind its compatibility audit')
    return {'existing_keys': frozenset(old_map), 'fresh_keys': frozenset(new_map)}

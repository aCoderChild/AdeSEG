"""Three-fold metric routing with the original scoring and audit metrics.

Labels are only decoded after the caller has frozen and verified all 18
prediction sets. This helper does not run verification by itself.
"""
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.crossfit_training_contract_3fold_v1 import fold_split
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.score_boundary_video_factorial_v1 import metric as primary_metric
from scripts.score_warm_crossfit_v1 import label_hashes, summarize


def compute(index, samples, records, *, independent=False):
    if independent:
        from scripts.audit_boundary_video_factorial_v1 import label, metrics
    else:
        label = lambda path: load_label_mask(path, [1]) == 1
        metrics = primary_metric
    hashes = label_hashes(index, samples)
    by_sequence = defaultdict(list)
    for key in sorted(samples):
        item = samples[key]; owners = [i for i in range(1, 4) if key[0] in fold_split(i)[2]]
        require(len(owners) == 1, 'Patient must have exactly one held-out fold')
        fold = owners[0]
        path = Path(item['label']); require(digest(path) == hashes[key], 'Label changed before decoding')
        truth = label(path); require(digest(path) == hashes[key], 'Label changed during decoding')
        row = {'sequence': key[0], 'frame': key[1], 'fold': fold}
        if key[0] == 'seq23':
            require(truth.shape == (720, 1280), 'Seq23 source-label geometry differs')
            row['viewport'] = {}
        for name in ('baseline', *ARMS):
            arm = ARMS[0] if name == 'baseline' else name
            kind = 'baseline' if name == 'baseline' else 'refined'
            saved = records[(fold, arm)]['mapping'][key]
            file = Path(saved[kind + '_path'])
            require(digest(file) == saved[kind + '_sha256'], 'Prediction changed before scoring')
            with Image.open(file) as image:
                mask = np.asarray(image.convert('L')) > 0
            require(digest(file) == saved[kind + '_sha256'] and mask.shape == truth.shape,
                    'Prediction changed or target geometry differs')
            row[name] = metrics(mask, truth)
            if key[0] == 'seq23':
                x0, y0, x1, y1 = VIEWPORT
                row['viewport'][name] = metrics(mask[y0:y1, x0:x1], truth[y0:y1, x0:x1])
        by_sequence[key[0]].append(row)
    names = ('baseline', *ARMS); flat = [row for sequence in sorted(by_sequence) for row in by_sequence[sequence]]
    summaries = {}
    for name in names:
        per = {seq: summarize(rows, name) for seq, rows in sorted(by_sequence.items())}
        means = [x['mean_present_dice'] for x in per.values() if x['mean_present_dice'] is not None]
        summaries[name] = {'pooled': summarize(flat, name), 'per_sequence': per,
                           'patient_macro_present_dice': math.fsum(means) / len(means) if means else None,
                           'patients_with_present_reads': len(means)}
    def paired(items, arm):
        delta = [r[arm]['dice'] - r['baseline']['dice'] for r in items if r['baseline']['present']]
        return {'present_reads': len(delta), 'mean_present_dice_difference':
                math.fsum(delta) / len(delta) if delta else None,
                'improved': sum(x > 0 for x in delta), 'worsened': sum(x < 0 for x in delta),
                'unchanged': sum(x == 0 for x in delta)}
    comparisons = {arm: {'pooled': paired(flat, arm),
                         'per_sequence': {seq: paired(items, arm) for seq, items in sorted(by_sequence.items())}}
                   for arm in ARMS}
    viewport = {name: summarize([r['viewport'] for r in by_sequence.get('seq23', [])], name) for name in names}
    return {'rows': dict(by_sequence), 'summaries': summaries, 'paired_vs_baseline': comparisons,
            'seq23_viewport_summaries': viewport,
            'no_initialization': ['seq1', 'seq7'], 'no_initialization_dice': None}


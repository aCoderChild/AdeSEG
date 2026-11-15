"""Block all non-training manifest masks from CPython opens during training.

This audit hook cannot intercept arbitrary native-code system calls and is an
engineering control alongside source review, not a universal sandbox.
"""
import json
import os
from pathlib import Path
import sys

TRAIN = frozenset(('seq2', 'seq3', 'seq4', 'seq5', 'seq12', 'seq13', 'seq14'))


def mask_path_sets(manifest):
    manifest = Path(manifest)
    train, blocked = set(), set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row['mask'])
        if not path.is_absolute():
            path = manifest.parent/path
        (train if row['video_id'] in TRAIN else blocked).add(os.path.realpath(path))
    if train & blocked or not train or not blocked:
        raise ValueError('Train and protected mask paths overlap or are empty')
    return train, blocked


def install_training_mask_guard(manifest):
    train, blocked = mask_path_sets(manifest)
    counts = {'train_paths': len(train), 'blocked_paths': len(blocked), 'blocked_open_attempts': 0}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if os.path.realpath(os.fsdecode(args[0])) in blocked:
            counts['blocked_open_attempts'] += 1
            raise PermissionError('Non-training label open blocked')

    sys.addaudithook(audit)
    return counts

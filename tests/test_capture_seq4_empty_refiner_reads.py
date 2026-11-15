import json
from types import SimpleNamespace

import pytest
import torch

from scripts.capture_seq4_empty_refiner_reads import TRACE_FILES, verify_match
from scripts.memory_token_io import digest


def _matched_pair(tmp_path):
    reference = tmp_path / 'ref'
    output = tmp_path / 'out'
    for folder in (reference, output):
        folder.mkdir()
        (folder / 'features').mkdir()
        (folder / 'snapshots').mkdir()
        for index in range(48):
            (folder / f'{index:06d}.png').write_bytes(bytes([index]))
        for path in ('features/000019.pt', 'snapshots/000019.pt'):
            torch.save({'frame': 19, 'value': torch.tensor([1., 2.])}, folder / path)
        for name in TRACE_FILES[1:]:
            (folder / name).write_text('{}')
        (folder / 'token_window_events.json').write_text('[{"frame": 19}]')
        masks = [digest(folder / f'{i:06d}.png') for i in range(48)]
        (folder / 'records.json').write_text(json.dumps({'frames': 48, 'mask_sha256': masks}))
    events = [{'frame': i} for i in (17, 18, 19)]
    return reference, output, SimpleNamespace(events=[{'frame': 19}]), events


def test_requires_exact_prior_trajectory_and_overlapping_tensors(tmp_path):
    reference, output, runtime, events = _matched_pair(tmp_path)
    verify_match(reference, output, runtime, events, events)
    (output / '000017.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='frame 17'):
        verify_match(reference, output, runtime, events, events)


def test_rejects_missing_read_or_changed_overlap(tmp_path):
    reference, output, runtime, events = _matched_pair(tmp_path)
    with pytest.raises(ValueError, match='reads missing'):
        verify_match(reference, output, runtime, events, events[:-1])
    torch.save({'frame': 19, 'value': torch.tensor([1., 3.])}, output / 'snapshots/000019.pt')
    with pytest.raises(ValueError, match='Overlapping capture'):
        verify_match(reference, output, runtime, events, events)

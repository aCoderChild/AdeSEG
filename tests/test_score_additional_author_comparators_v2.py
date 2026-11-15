import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import score_additional_author_comparators_v2 as score


def test_invalid_output_rejected_before_guard_or_label_access(monkeypatch):
    calls = []
    def invalid(_):
        calls.append('output')
        raise ValueError('outside experiment')
    monkeypatch.setattr(score, 'new_output', invalid)
    monkeypatch.setattr(score, 'install_phase_mask_guard', lambda _: calls.append('guard'))
    monkeypatch.setattr(score, 'load_label_mask', lambda *_: calls.append('label'))
    with pytest.raises(ValueError, match='outside experiment'):
        score.score('run_map', 'verification', 'invalid')
    assert calls == ['output']


def test_failed_verification_does_not_open_labels(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    run_map = tmp_path / 'map.json'
    run_map.write_text('{}')
    verification = tmp_path / 'verification.json'
    verification.write_text(json.dumps({'passed': False}))
    calls = []
    monkeypatch.setattr(score, 'new_output', lambda _: output)
    monkeypatch.setattr(score, 'install_phase_mask_guard', lambda _: {'active': True, 'open_attempts': 0})
    monkeypatch.setattr(score, 'load_label_mask', lambda *_: calls.append('label'))
    with pytest.raises(ValueError, match='verification'):
        score.score(run_map, verification, output)
    assert not calls and not output.exists()


def test_phase_guard_blocks_then_permits_mask_opens(tmp_path):
    # CPython audit hooks are permanent for a process, so exercise in a child.
    train = tmp_path / 'train.png'
    other = tmp_path / 'other.png'
    train.write_text('train')
    other.write_text('other')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(x) for x in (
        {'video_id': 'seq2', 'mask': str(train)}, {'video_id': 'seq8', 'mask': str(other)})))
    code = '''
import sys
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
s=install_phase_mask_guard(sys.argv[1])
for p in sys.argv[2:]:
    try: open(p).read()
    except PermissionError: pass
    else: raise AssertionError('mask was not blocked')
assert s['open_attempts']==2
s['active']=False
for p in sys.argv[2:]: open(p).read()
'''
    subprocess.run([sys.executable, '-c', code, str(manifest), str(train), str(other)], check=True)
def test_receipt_bound_to_launch_hash(tmp_path):
    import json
    from scripts.memory_token_io import digest
    from scripts.score_additional_author_comparators_v2 import read_bound_receipt
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'masks': ['original']}))
    frozen = digest(receipt)
    assert read_bound_receipt(receipt, frozen)['masks'] == ['original']
    receipt.write_text(json.dumps({'masks': ['changed']}))
    with pytest.raises(ValueError, match='changed after launch'):
        read_bound_receipt(receipt, frozen)

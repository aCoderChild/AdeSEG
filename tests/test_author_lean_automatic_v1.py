import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import run_author_lean_automatic_v1 as runner
from scripts import verify_author_lean_automatic_gate_v1 as verifier
from scripts.memory_token_io import digest


def test_scope_covers_only_initialized_sequences():
    assert len(runner.SEQUENCES) == 21
    assert set(runner.SEQUENCES) == set(runner.ADDITIONAL) | set(runner.PRIORITY)
    assert 'seq1' not in runner.SEQUENCES and 'seq7' not in runner.SEQUENCES


@pytest.fixture
def smokes(tmp_path, monkeypatch):
    mapping, frames = {}, {}
    for sequence, start in [('seq6', 2), ('seq23', 0)]:
        root = tmp_path / sequence; root.mkdir()
        frames[sequence] = []
        records = []
        for i in range(start + 5):
            source = root / f'source{i}.png'
            Image.fromarray(np.full((4, 6), i, np.uint8)).save(source)
            frames[sequence].append(SimpleNamespace(frame_index=i * 2, image_path=source))
            file = root / f'{i:06d}.png'
            Image.fromarray(np.full((4, 6), 255 if i >= start else 0, np.uint8)).save(file)
            records.append({'frame': i, 'source_frame_index': i * 2,
                'image_sha256': digest(source), 'mask_sha256': digest(file), 'post_prompt': i >= start})
        ident = {'sequence': sequence, 'source': 'same'}
        rec = {'passed': True, 'full_video': False, 'identity': ident, 'prompt_frame': start,
               'prompt': {'sequence': sequence}, 'label_guard': {'open_attempts': 0},
               'state_sha256': 'same-weights', 'process_id': 1 if sequence == 'seq6' else 2,
               'positions': [{'spatial_keys': 4096}], 'records': records}
        (root / 'receipt.json').write_text(json.dumps(rec)); mapping[sequence] = root
    monkeypatch.setattr(verifier, 'inputs', lambda seq, *args: (frames[seq],
        2 if seq == 'seq6' else 0, None, {'sequence': seq}))
    monkeypatch.setattr(verifier, 'identity', lambda seq, *args: {'sequence': seq, 'source': 'same'})
    return mapping


def test_gate_reconstructs_and_rejects_changed_prediction(smokes):
    args = (None, None, None, None)
    gate = verifier.audit(smokes, *args)
    assert gate['passed'] and gate['state_sha256'] == 'same-weights'
    Image.fromarray(np.zeros((4, 6), np.uint8)).save(smokes['seq6'] / '000002.png')
    with pytest.raises(ValueError, match='identity'):
        verifier.audit(smokes, *args)


def test_gate_rejects_different_model_state(smokes):
    path = smokes['seq23'] / 'receipt.json'
    rec = json.loads(path.read_text()); rec['state_sha256'] = 'other'; path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='model states'):
        verifier.audit(smokes, None, None, None, None)


def test_gate_rejects_opened_label(smokes):
    path = smokes['seq6'] / 'receipt.json'
    rec = json.loads(path.read_text()); rec['label_guard']['open_attempts'] = 1; path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='guard'):
        verifier.audit(smokes, None, None, None, None)


def test_scoring_cannot_open_label_before_all_frozen(tmp_path, monkeypatch):
    from scripts import score_author_lean_automatic_v1 as scorer
    verification = tmp_path/'verification'; verification.mkdir()
    (verification/'verification.json').write_text('{}')
    monkeypatch.setattr(scorer, 'new_output', lambda p: p)
    monkeypatch.setattr(scorer, 'install_phase_mask_guard', lambda _: {'open_attempts': 0, 'active': True})
    monkeypatch.setattr(scorer, 'read_bound', lambda *args: {'passed': True,
        'scorer_sha256': digest(Path(scorer.__file__)), 'label_guard': {'open_attempts': 0}})
    monkeypatch.setattr(scorer, 'validate_frozen', lambda *args: (_ for _ in ()).throw(ValueError('unfinished')))
    monkeypatch.setattr(scorer, 'load_label_mask', lambda *args: pytest.fail('label read before verification'))
    with pytest.raises(ValueError, match='unfinished'):
        scorer.score(tmp_path)


def test_launcher_scopes_all_21_before_scoring(tmp_path):
    from scripts.run_author_lean_automatic_study_v1 import commands
    names = [name for name, _ in commands(tmp_path)]
    assert names[:3] == ['smoke_seq6', 'smoke_seq23', 'gate']
    assert names[3:24] == [f'full_{seq}' for seq in runner.SEQUENCES]
    assert names[24:] == ['verify', 'score', 'audit']


def test_patient_macro_and_frame_weighted_have_distinct_denominators():
    from scripts.score_author_lean_automatic_v1 import overall_summary
    result = overall_summary({'one': {'present_frames': 1, 'mean_present_dice': 1.0, 'empty_frames': 2, 'false_alarms': 1},
                              'two': {'present_frames': 3, 'mean_present_dice': 0.0, 'empty_frames': 1, 'false_alarms': 1},
                              'empty': {'present_frames': 0, 'mean_present_dice': None, 'empty_frames': 3, 'false_alarms': 0}})
    assert result == {'patients_with_present_frames': 2, 'present_frames': 4,
        'patient_macro_present_dice': .5, 'frame_weighted_present_dice': .25,
        'labeled_empty_frames': 6, 'labeled_empty_false_alarms': 2}

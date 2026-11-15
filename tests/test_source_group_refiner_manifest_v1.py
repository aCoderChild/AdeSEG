import copy
import json

import pytest

import scripts.prepare_source_group_refiner_manifest_v1 as bridge


def test_only_archived_frame_and_label_hash_fields_are_projected(tmp_path):
    path = tmp_path/'frames.json'
    path.write_text(json.dumps([{'index': 0, 'source_frame_index': 41, 'label_sha256': 'a'*64,
                                 'present': True, 'dice': 0.8, 'false_alarm': False}]))
    assert bridge.label_hash_projection(path) == [{'index': 0, 'source_frame_index': 41, 'label_sha256': 'a'*64}]
    path.write_text('[{"index":0,"source_frame_index":41}]')
    with pytest.raises(ValueError):
        bridge.label_hash_projection(path)


def test_capture_map_requires_complete_unique_nonaliasing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, 'validate_output', lambda path: path)
    mapping = {seq: {kind: str(tmp_path/f'{seq}_{kind}') for kind in ('reference', 'capture', 'audit')}
               for seq in bridge.SEQUENCES}
    assert len(bridge.resolve_capture_map(mapping)) == 9
    changed = copy.deepcopy(mapping); changed.pop('seq15')
    with pytest.raises(ValueError, match='every frozen sequence'):
        bridge.resolve_capture_map(changed)
    changed = copy.deepcopy(mapping); changed['seq15']['audit'] = changed['seq2']['audit']
    with pytest.raises(ValueError, match='aliases'):
        bridge.resolve_capture_map(changed)
    path = tmp_path/'duplicates.json'; path.write_text('{"seq2": {}, "seq2": {}}')
    with pytest.raises(ValueError, match='Duplicate'):
        bridge.read_unique_json(path)


def test_only_complete_audited_matching_full_captures_are_accepted(monkeypatch):
    monkeypatch.setattr(bridge, 'digest', lambda _: 'auditor')
    spec = {'frames': 4, 'prompt_frame': 1, 'targets': [2, 3]}
    identity = {'preflight_sha256': 'pre', 'source_freeze_sha256': 'freeze'}
    common = {'passed': True, 'sequence': 'seq3', 'identity': identity, 'limit': 0,
              'full_sequence': True, 'label_guard': {'open_attempts': 0},
              'short_gate_sha256': 'gate', 'prompt_frame': 1, 'targets': [2, 3],
              'rows': [{'frame': 1}, {'frame': 2}, {'frame': 3}],
              'positions': [], 'selection': [], 'anchor_events': [], 'token_window_events': []}
    ref = {**common, 'mode': 'reference'}
    cap = {**copy.deepcopy(common), 'mode': 'capture', 'observations': [{'frame': 2}, {'frame': 3}]}
    audit = {'passed': True, 'sequence': 'seq3', 'identity': identity, 'limit': 0,
             'full_sequence': True, 'label_guard': {'open_attempts': 0}, 'targets': [2, 3],
             'frames': 3, 'auditor_sha256': 'auditor'}
    args = dict(preflight_sha='pre', freeze_sha='freeze', gate_sha='gate', identity=identity)
    bridge.verify_capture_metadata('seq3', spec, ref, cap, audit, **args)
    changed = copy.deepcopy(cap); changed['full_sequence'] = False
    with pytest.raises(ValueError):
        bridge.verify_capture_metadata('seq3', spec, ref, changed, audit, **args)
    changed = copy.deepcopy(cap); changed['observations'].pop()
    with pytest.raises(ValueError):
        bridge.verify_capture_metadata('seq3', spec, ref, changed, audit, **args)
    changed = copy.deepcopy(cap); changed['positions'] = ['changed']
    with pytest.raises(ValueError):
        bridge.verify_capture_metadata('seq3', spec, ref, changed, audit, **args)
    with pytest.raises(ValueError):
        bridge.verify_capture_metadata('seq3', spec, ref, cap, audit, **{**args, 'gate_sha': 'other'})

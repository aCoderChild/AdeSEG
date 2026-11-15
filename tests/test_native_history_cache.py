from copy import deepcopy

import pytest
import torch

from modeling.native_history_cache import bank_payload, cached_read, load_bank
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest
from scripts.run_regional_refinement import save_tensor


def fixture(object_id=7):
    position = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    def entry(i):
        return {"maskmem_features": torch.full((1, 4, 2, 2), float(i), dtype=torch.bfloat16),
                "pred_masks": torch.full((1, 1, 4, 4), float(i)),
                "maskmem_pos_enc": [position.expand(1, -1, -1, -1)],
                "obj_ptr": torch.full((1, 16), float(object_id)),
                "object_score_logits": torch.tensor([[float(i)]])}
    bank = {"cond_frame_outputs": {0: entry(0)},
            "non_cond_frame_outputs": {i: entry(i) for i in range(1, 4)}}
    state = {"num_frames": 4, "obj_id_to_idx": {object_id: 0},
             "output_dict_per_obj": {0: bank}, "cached_features": {}}
    return state


def test_serialization_preserves_native_dtype_layout_and_shared_positions(tmp_path):
    payload = bank_payload(fixture(), 7)
    path = tmp_path / 'bank.pt'
    save_tensor(payload, path)
    receipt = {"object_id": 7, "num_frames": 4, "prompt": 0,
               "bank_sha256": digest(path), "payload_sha256": fingerprint(payload)["sha256"],
               "checkpoint_sha256": 'checkpoint', "manifest_sha256": 'manifest', "model_config": 'config'}
    loaded = load_bank(path, receipt, 'checkpoint', 'manifest', 'config')
    assert fingerprint(loaded)["sha256"] == fingerprint(payload)["sha256"]
    first = loaded['outputs']['cond_frame_outputs'][0]['maskmem_pos_enc'][0]
    later = loaded['outputs']['non_cond_frame_outputs'][3]['maskmem_pos_enc'][0]
    assert first.untyped_storage().data_ptr() == later.untyped_storage().data_ptr()
    assert loaded['outputs']['non_cond_frame_outputs'][1]['maskmem_features'].dtype == torch.bfloat16
    with pytest.raises(ValueError, match='provenance'):
        load_bank(path, receipt, 'wrong checkpoint', 'manifest', 'config')
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='Cannot load verified tensor'):
        load_bank(path, receipt, 'checkpoint', 'manifest', 'config')


def test_cached_read_is_past_only_instance_specific_and_restores_state():
    state = fixture(7)
    other = fixture(42)['output_dict_per_obj'][0]
    state['obj_id_to_idx'][42] = 1
    state['output_dict_per_obj'][1] = other
    original = state['output_dict_per_obj'][0]
    other_before = fingerprint(other)['sha256']
    payload = bank_payload(state, 7)
    class Model:
        def _get_image_feature(self, state, frame, batch):
            state['cached_features'][frame] = True
            return None, None, [torch.tensor([[float(frame)]])], [torch.ones(1)], [(1, 1)]
    model = Model()
    sample = cached_read(model, state, 2, payload)
    changed = deepcopy(payload)
    changed['outputs']['non_cond_frame_outputs'][3]['maskmem_features'].fill_(float('nan'))
    again = cached_read(model, state, 2, changed)
    assert fingerprint(sample['output_dict'])['sha256'] == fingerprint(again['output_dict'])['sha256']
    assert set(sample['output_dict']['non_cond_frame_outputs']) == {1}
    assert sample['output_dict']['non_cond_frame_outputs'][1]['obj_ptr'][0, 0] == 7
    assert state['output_dict_per_obj'][0] is original
    assert fingerprint(other)['sha256'] == other_before
    second = cached_read(model, state, 2, bank_payload(state, 42))
    assert second['output_dict']['non_cond_frame_outputs'][1]['obj_ptr'][0, 0] == 42
    assert state['output_dict_per_obj'][1] is other
    with pytest.raises(ValueError, match='differs'):
        cached_read(model, state, 0, payload)
    bad = {**payload, 'object_id': 99}
    with pytest.raises(ValueError, match='differs'):
        cached_read(model, state, 2, bad)


def test_partial_banks_cannot_be_published_as_complete():
    state = fixture()
    del state['output_dict_per_obj'][0]['non_cond_frame_outputs'][3]
    with pytest.raises(ValueError, match='complete'):
        bank_payload(state, 7)

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from datasets.sam2_clip_sampler import sample_object_clip
from datasets.video import ManifestFrame
from modeling.cfbi_training import capture_training_read
from modeling.multi_object_video import propagate_object_prompts
from modeling.prediction_conditioned_affinity import _stored_role_map
from modeling.regional_low_rank import RegionalLowRankLayer


def test_batched_roles_equal_independent_objects_without_union():
    masks = torch.full((3, 1, 256, 256), -20.)
    masks[0, :, :128] = 20
    masks[1, :, :, 128:] = 20
    actual = _stored_role_map(masks, "cpu")
    for index in range(3):
        assert torch.equal(actual[index:index + 1], _stored_role_map(masks[index:index + 1], "cpu"))
    assert actual[2, :, 0].min() > .99
    assert not torch.equal(actual[0], actual[1])


def test_low_rank_object_batch_has_no_cross_object_features_or_gradients():
    torch.manual_seed(9)
    layer = RegionalLowRankLayer(channels=8, rank=2)
    with torch.no_grad():
        for region in layer.regions:
            region.lora_B.normal_()
    memory = torch.randn(3, 7, 8, requires_grad=True)
    roles = torch.rand(3, 5, 3).softmax(-1)
    actual = layer(memory, roles, pointers=2)
    for index in range(3):
        torch.testing.assert_close(actual[index:index + 1],
            layer(memory[index:index + 1], roles[index:index + 1], 2))
    actual[0].sum().backward()
    assert memory.grad[0].abs().sum() > 0
    assert memory.grad[1:].count_nonzero() == 0
    assert torch.equal(actual[:, -2:], memory[:, -2:])
    changed = memory.detach().clone()
    changed[1] *= 100
    assert torch.equal(layer(changed, roles, 2)[0], actual.detach()[0])


def test_training_capture_selects_persistent_id_not_slot_zero():
    model = SimpleNamespace(_get_image_feature=lambda *args: (None, None, [], [], []))
    state = {"cached_features": {1: True}, "num_frames": 3, "obj_id_to_idx": {7: 0, 42: 1},
             "output_dict_per_obj": {
                 i: {"cond_frame_outputs": {0: {"pred_masks": torch.tensor([float(i)])}},
                     "non_cond_frame_outputs": {}} for i in range(2)}}
    sample = capture_training_read(model, state, 1, object_id=42)
    assert sample["output_dict"]["cond_frame_outputs"][0]["pred_masks"].item() == 1
    sample["output_dict"]["cond_frame_outputs"][0]["pred_masks"].zero_()
    assert state["output_dict_per_obj"][1]["cond_frame_outputs"][0]["pred_masks"].item() == 1
    with pytest.raises(KeyError):
        capture_training_read(model, state, 1, object_id=99)


def test_author_sampler_keeps_multiple_nonconsecutive_ids(tmp_path):
    frames = []
    for frame in range(4):
        mask = np.zeros((8, 8), np.uint16)
        mask[:4] = 7
        mask[4:] = 300
        path = tmp_path / f"{frame}.png"
        Image.fromarray(mask).save(path)
        frames.append(ManifestFrame("multi", frame, path, path))
    selected, ids = sample_object_clip(frames, 3, 9, label_ids=(7, 300), max_objects=2)
    assert set(ids) == {7, 300} and len(selected) == 3
    assert (selected, ids) == sample_object_clip(frames, 3, 9, label_ids=(7, 300), max_objects=2)


class PromptModel:
    def add_new_mask(self, state, frame_idx, obj_id, mask):
        state.setdefault("ids", []).append(obj_id)
        state.setdefault("events", []).append((frame_idx, obj_id))

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
        for frame in range(start_frame_idx, start_frame_idx + max_frame_num_to_track + 1):
            yield frame, list(state["ids"]), None


def test_later_object_is_prompted_online_and_frames_are_not_repeated():
    state = {"num_frames": 5}
    prompts = {0: {7: np.ones((2, 2), bool)}, 2: {42: np.ones((2, 2), bool)}}
    results = list(propagate_object_prompts(PromptModel(), state, prompts))
    assert [frame for frame, _, _ in results] == list(range(5))
    assert [ids for _, ids, _ in results] == [[7], [7], [7, 42], [7, 42], [7, 42]]
    assert state["events"] == [(0, 7), (2, 42)]


def test_invalid_prompt_schedule_fails_before_native_mutation():
    state = {"num_frames": 5}
    with pytest.raises(ValueError, match="one first prompt"):
        list(propagate_object_prompts(PromptModel(), state,
            {0: {7: np.ones((2, 2))}, 2: {7: np.ones((2, 2))}}))
    assert "events" not in state

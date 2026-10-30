import pytest
from types import SimpleNamespace

import torch

from scripts.analyze_v2_trajectories import longest_lower_run, first_pruning


def test_lower_runs_respect_absence_ties_and_manifest_indices():
    frames = [7, 8, 9, 10, 11, 12, 13]
    present = [True, True, False, True, True, True, True]
    delta = [-.1, -.2, -.8, -.3, 0., -.1, -.2]
    assert longest_lower_run(frames, present, delta) == {
        "start": 7, "end": 8, "length": 2, "summed_dice_deficit": pytest.approx(.3)}
    assert longest_lower_run(frames, present, [0.]*7) is None


def test_first_pruning_uses_actual_keys_and_rejects_inconsistent_layers():
    def row(raw, selected):
        return {"trace": {"layers": [{"layer": i, "raw_spatial_keys": raw, "selected_spatial_keys": selected} for i in range(4)]}}
    receipts = {3: {"trace": {"layers": []}}, 4: row(4096,4096), 5: row(8192,8192), 6: row(12288,12287)}
    assert first_pruning(receipts) == 6
    receipts[6]["trace"]["layers"][2]["selected_spatial_keys"] = 12286
    with pytest.raises(ValueError, match="Inconsistent"):
        first_pruning(receipts)


def native_state():
    mask = torch.full((1, 1, 256, 256), -1.)
    mask[0, 0, 20, 20] = 1.
    def item(frame):
        features = torch.full((1, 64, 64, 64), float(frame))
        return {"pred_masks": mask.clone(), "maskmem_features": features,
                "maskmem_pos_enc": [torch.zeros_like(features)]}
    return {"output_dict_per_obj": {0: {"cond_frame_outputs": {3: item(3)},
                                       "non_cond_frame_outputs": {f:item(f) for f in (4,5,6)}}}}


def test_pulse_bank_exact_random_context_dense_endpoints_and_causality():
    from modeling.matched_region_memory import random_cells
    from scripts.run_v2_pulse import pulse_bank
    state = native_state()
    regional = pulse_bank(state, 3, 6, "seq2", "regional")
    random = pulse_bank(state, 3, 6, "seq2", "random", 20261025)
    assert list(regional) == [3,4,5]  # Current frame6 excluded even if present in state.
    assert regional[3]["keep"].all() and regional[5]["keep"].all()
    assert int(regional[4]["keep"].sum()) == 4095
    assert torch.equal(random[4]["keep"], random_cells(4095, "seq2", 4, 20261025))
    assert regional[4]["features"].data_ptr() == state["output_dict_per_obj"][0]["non_cond_frame_outputs"][4]["maskmem_features"].data_ptr()
    with pytest.raises(ValueError):
        pulse_bank(state, 3, 6, "seq2", "random")


def test_one_pulse_writes_normally_and_restores_native_forward(monkeypatch):
    from scripts import run_v2_pulse as runner
    blocks = [SimpleNamespace(cross_attn_image=SimpleNamespace()) for _ in range(4)]
    writes, active = [], []
    model = SimpleNamespace(memory_attention=SimpleNamespace(layers=blocks),
                            _encode_new_memory=lambda: writes.append(True))

    class Reader:
        def __init__(self, model, prompt, mode):
            self.model = model
        def __enter__(self):
            for b in self.model.memory_attention.layers:
                b.cross_attn_image.forward = object()
            frames = list(self.bank)
            count = sum(4096 if i in (0,len(frames)-1) else int(self.bank[f]["keep"].sum()) for i,f in enumerate(frames))
            self.layers = [{"layer": i, "frames": frames, "selected_spatial_keys": count} for i in range(4)]
            return self
        def __exit__(self, *exc):
            for b in self.model.memory_attention.layers:
                del b.cross_attn_image.forward

    monkeypatch.setattr(runner, "SparseMemoryRead", Reader)
    def predictions():
        for index in (5,6,7):
            active.append("forward" in blocks[0].cross_attn_image.__dict__)
            model._encode_new_memory()
            yield index, [1], torch.zeros(1)
    iterator = predictions()
    receipts = []
    for index in (5,6,7):
        _, pulse = runner.next_at_target(iterator, model, native_state(), index, 3, 6, "seq2", "regional")
        receipts.append(pulse)
    assert active == [False, True, False]
    assert len(writes) == 3
    assert [r is not None for r in receipts] == [False,True,False]
    assert receipts[1]["native_memory_writes"] == 1
    runner.assert_native_forward(model)


def test_leaked_sparse_attributes_are_rejected():
    from scripts.run_v2_pulse import assert_native_forward
    model = SimpleNamespace(memory_attention=SimpleNamespace(layers=[
        SimpleNamespace(cross_attn_image=SimpleNamespace(_sparse_mask=torch.ones(1))) ]))
    with pytest.raises(RuntimeError, match="outside the pulse"):
        assert_native_forward(model)


def test_terminal_empty_distinguishes_absent_success_and_present_miss():
    from scripts.analyze_v2_trajectories import prediction_states
    records = [{"frame": i+10, "present": present,
                "arms": {"v2": {"prediction_nonempty": nonempty, "removed_tokens": removed}}}
               for i,(present,nonempty,removed) in enumerate([(True,True,2),(False,True,3),(False,False,1),(True,False,0)])]
    assert prediction_states(records)["v2"] == {"empty_predictions_on_present": 1, "false_alarms": 1,
                                                 "terminal_empty_start": 12, "last_pruning_frame": 12}

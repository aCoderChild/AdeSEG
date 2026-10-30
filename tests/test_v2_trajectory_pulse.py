import pytest
from types import SimpleNamespace

import torch

from scripts.analyze_v2_trajectories import longest_lower_run, first_pruning


def test_transition_schedules_freeze_cutoff_and_pulse():
    from scripts.run_v2_pulse import intervention_frames
    assert intervention_frames("transition_pulse", 10) == [50]
    assert intervention_frames("stop_before_transition", 10) == list(range(10,50))
    assert 50 not in intervention_frames("stop_before_transition", 10)
    assert intervention_frames("early_pulse", 6) == [6]
    with pytest.raises(ValueError):
        intervention_frames("unknown", 10)


def test_archive_rejects_empty_members(tmp_path):
    from scripts.run_v2_pulse import write_verified_archive
    folder = tmp_path/"data"
    folder.mkdir()
    (folder/"inputs.json").write_bytes(b"")
    with pytest.raises(ValueError, match="empty archive member"):
        write_verified_archive(folder, tmp_path/"bad.zip")


def test_input_preflight_requires_independent_receipts_and_actual_hashes(tmp_path):
    import json
    import zipfile
    from scripts.run_v2_pulse import verified_inputs, author
    replay, earlier = tmp_path/"replay", tmp_path/"earlier"
    (replay/"full/seq13").mkdir(parents=True)
    (earlier/"seq13").mkdir(parents=True)
    image, mask = tmp_path/"image", tmp_path/"mask"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    frames = [SimpleNamespace(frame_index=0, image_path=image, mask_path=mask)]
    data = [{"frame": 0, "image_sha256": author.digest(image), "mask_sha256": author.digest(mask)}]
    for path in [replay/"full/seq13/inputs.json", earlier/"seq13/inputs.json"]:
        path.write_text(json.dumps(data))
    with zipfile.ZipFile(replay/"seq13.zip", "w") as z:
        z.writestr("inputs.json", b"")
    found, receipt = verified_inputs(replay, earlier, "seq13", frames)
    assert found == data and receipt["archive_input_bytes"] == 0
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image/GT/chronology"):
        verified_inputs(replay, earlier, "seq13", frames)
    (earlier/"seq13/inputs.json").write_text("[]")
    with pytest.raises(ValueError, match="receipts disagree"):
        verified_inputs(replay, earlier, "seq13", frames)


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


@pytest.mark.parametrize("condition,seed,matched", [("regional",None,None), ("random",20261025,[4096,123,4096])])
def test_one_pulse_writes_normally_and_restores_native_forward(monkeypatch, condition, seed, matched):
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
        _, pulse = runner.next_at_target(iterator, model, native_state(), index, 3, 6, "seq2", condition, seed, matched)
        receipts.append(pulse)
    assert active == [False, True, False]
    assert len(writes) == 3
    assert [r is not None for r in receipts] == [False,True,False]
    assert receipts[1]["native_memory_writes"] == 1
    if matched is not None:
        assert receipts[1]["per_frame_counts"] == matched
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


import copy
import torch
import pytest
from scripts.bank_integrity import fingerprint
from scripts.v2_writeback_control import FrameWritebackControl, CORE


def example():
    output = {k:torch.tensor([1.]) for k in ('pred_masks','maskmem_features','obj_ptr','object_score_logits')}
    output['maskmem_pos_enc']=[torch.tensor([2.])]
    output['additional_author_field']={'nested':torch.tensor([3.])}
    state={'output_dict_per_obj':{0:{'cond_frame_outputs':{7:{'prompt':torch.tensor([4.])}},
            'non_cond_frame_outputs':{50:output}}},'cached_features':{50:torch.tensor([5.])}}
    return state


def core(state):
    return {'prediction_sha256':'prediction','low_res_sha256':'low',
            'bank_sha256':fingerprint(state['output_dict_per_obj'])['sha256'],
            'cache_sha256':fingerprint(state['cached_features'])['sha256']}


def test_whole_output_replacement_preserves_emitted_mask_and_all_fields(tmp_path):
    native=example(); capture=FrameWritebackControl(tmp_path/'native.pt')
    capture(native,50,core(native),{'layers':[1]})
    changed=copy.deepcopy(native)
    changed['output_dict_per_obj'][0]['non_cond_frame_outputs'][50]['pred_masks'].fill_(-1)
    emitted=changed['output_dict_per_obj'][0]['non_cond_frame_outputs'][50]['pred_masks']
    before=core(changed); pulse={'layers':[1]}
    control=FrameWritebackControl(tmp_path/'native.pt',{**before,'pulse':pulse});control.snapshot=capture.snapshot
    receipt=control(changed,50,before,pulse)
    assert core(changed)==core(native)
    assert emitted.item()==-1
    assert 'additional_author_field' in receipt['stored_keys']
    changed['output_dict_per_obj'][0]['non_cond_frame_outputs'][50]['additional_author_field']['nested'].fill_(99)
    assert capture.snapshot['output']['additional_author_field']['nested'].item()==3
    assert native['output_dict_per_obj'][0]['non_cond_frame_outputs'][50]['additional_author_field']['nested'].item()==3
    assert control(changed,51,{},None) is None


def test_rejects_nonmatching_pulse_before_mutation(tmp_path):
    state=example(); before=core(state)
    control=FrameWritebackControl(tmp_path/'native.pt',{**before,'prediction_sha256':'wrong','pulse':{}})
    control.snapshot={'output':copy.deepcopy(state['output_dict_per_obj'][0]['non_cond_frame_outputs'][50]),'core':before}
    with pytest.raises(RuntimeError,match='pre-replacement pulse'):
        control(state,50,before,{})
    assert core(state)==before


def test_replacement_cannot_hide_other_frame_or_cache_change(tmp_path):
    native=example(); capture=FrameWritebackControl(tmp_path/'native.pt'); capture(native,50,core(native),{})
    changed=copy.deepcopy(native);changed['cached_features'][50].fill_(8)
    before=core(changed)
    control=FrameWritebackControl(tmp_path/'native.pt',{**before,'pulse':{}});control.snapshot=capture.snapshot
    with pytest.raises(RuntimeError,match='bank/cache'):
        control(changed,50,before,{})

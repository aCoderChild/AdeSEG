import numpy as np
import pytest
import torch

from modeling.contiguous_exclusion import topology
from scripts.run_v2_target_replay import selection, verify_layers, summarize


def bank():
    keep = torch.ones(64, 64, dtype=torch.bool)
    keep[20:30, 20] = False
    keep[20:30, 29] = False
    keep[20, 20:30] = False
    keep[29, 20:30] = False
    return {f: {"keep": keep.flatten(), "features": torch.tensor([f])} for f in (0, 8, 9)}


@pytest.mark.parametrize("condition", ["regional", "random", "rigid"])
def test_control_count_dense_protection_repeatability_and_no_mutation(condition):
    original = bank()
    before = {f: item["keep"].clone() for f, item in original.items()}
    first, meta = selection(original, "seq2", 10, condition, 20261025)
    second, again = selection(original, "seq2", 10, condition, 20261025)
    assert meta == again
    for f in original:
        assert torch.equal(original[f]["keep"], before[f])
        assert first[f]["features"] is original[f]["features"]
        assert torch.equal(first[f]["keep"], second[f]["keep"])
    assert first[0]["keep"].all() and first[9]["keep"].all()
    assert first[8]["keep"].sum() == before[8].sum()
    if condition != "random":
        assert topology((~first[8]["keep"]).reshape(64, 64)) == topology((~before[8]).reshape(64, 64))


def test_layer_gate_rejects_wrong_pointers_counts_frames_and_missing_layers():
    selected, _ = selection(bank(), "seq2", 10, "regional")
    count = int(selected[8]["keep"].sum())+8192
    layers = [{"layer": i, "frames": [0, 8, 9], "selected_spatial_keys": count,
               "raw_spatial_keys": 12288, "pointer_keys": 16, "temporal_slots": [6, 1, 0]} for i in range(4)]
    assert verify_layers(layers, [0, 8, 9], selected, layers) == [4096, count-8192, 4096]
    for change in ({"pointer_keys": 12}, {"selected_spatial_keys": 1}, {"frames": [0, 7, 9]}):
        bad = [{**r, **change} for r in layers]
        with pytest.raises(RuntimeError):
            verify_layers(bad, [0, 8, 9], selected, layers)
    with pytest.raises(RuntimeError):
        verify_layers(layers[:3], [0, 8, 9], selected)


def test_summary_averages_seeds_then_targets_then_sequences_excludes_invariant_rigid():
    rows = []
    for seq, frames in (("seq2", [1, 2]), ("seq3", [1])):
        for frame in frames:
            for condition, scores in (("regional", [.5]), ("random", [.1, .3, .5]), ("rigid", [.4, .5, .6])):
                for score in scores:
                    rows.append({"sequence": seq, "frame": frame, "condition": condition,
                                 "present": True, "scores": {"dice": score},
                                 "baseline": {"dice": .6}, "control_informative": seq == "seq2"})
    result = summarize(rows)
    assert np.isclose(result["regional_minus_native"]["mean_difference"], -.1)
    assert np.isclose(result["regional_minus_random"]["mean_difference"], .2)
    assert result["regional_minus_rigid"]["n_sequences"] == 1
    assert np.isclose(result["regional_minus_rigid"]["mean_difference"], 0)


def test_tracking_calls_match_shared_required_arguments():
    import ast
    import inspect
    from scripts import run_v2_target_replay as runner
    tree = ast.parse(inspect.getsource(runner))
    checked = set()
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in {
            "start_wandb", "log_results_artifact", "finalize_tracking"
        }:
            inspect.signature(getattr(runner, call.func.id)).bind(
                *[None for _ in call.args], **{k.arg: None for k in call.keywords})
            checked.add(call.func.id)
    assert checked == {"start_wandb", "log_results_artifact", "finalize_tracking"}

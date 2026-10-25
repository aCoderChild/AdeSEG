from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modeling.matched_region_memory import MatchedRegionMemory, random_cells
from scripts.run_matched_region_experiment import load_reference, primary_statistics


@pytest.mark.parametrize("count", [0, 1, 31, 2048, 4095, 4096])
def test_author_random_exact_count_reproducible_and_rng_isolated(count):
    before = torch.get_rng_state().clone()
    mask = random_cells(count, "seq2", 13, 20261025)
    assert torch.equal(torch.get_rng_state(), before)
    assert mask.dtype == torch.bool and mask.shape == (4096,)
    assert int(mask.sum()) == count
    assert torch.equal(mask, random_cells(count, "seq2", 13, 20261025))


def test_seeds_frames_and_sequences_are_distinct_not_selected_by_outcome():
    first = random_cells(1000, "seq2", 13, 20261025)
    for seq, frame, seed in [("seq3", 13, 20261025), ("seq2", 14, 20261025), ("seq2", 13, 20261026)]:
        assert not torch.equal(first, random_cells(1000, seq, frame, seed))


def setup_matched():
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    reference = {i: {"frame": i, "retained_tokens": n, "allocation": None}
                 for i, n in enumerate([17, 31, 61, 50])}
    reference[3]["allocation"] = {"frames": [0, 1, 2], "per_frame_spatial_keys": [4096, 31, 4096],
                                  "pointer_keys": 4, "selected_spatial_keys": 8223}
    controller = MatchedRegionMemory(model, 0, "seq2", reference=reference, seed=20261025)
    controller.frame = 3
    controller.assembled_frames = [0, 1, 2]
    controller.bank = {i: {"features": torch.full((1, 64, 64, 64), float(i)),
                           "pos": torch.zeros(1, 64, 64, 64),
                           "keep": random_cells(reference[i]["retained_tokens"], "seq2", i, 20261025)} for i in range(3)}
    memory = torch.cat([controller.bank[i]["features"].flatten(2).transpose(1, 2) for i in range(3)]+[torch.zeros(1, 4, 64)], dim=1)
    kwargs = {"q": torch.zeros(1, 4, 256), "v": memory, "num_k_exclude_rope": 4}
    return controller, kwargs, reference


def test_c3_matches_c2_per_frame_counts_all_layers_and_protects_dense_frames():
    controller, kwargs, reference = setup_matched()
    attention = SimpleNamespace(training=False)
    for layer in range(4):
        controller._before(layer, attention, (), kwargs)
        assert attention._sparse_mask.shape == (4096,)
        assert int(attention._sparse_mask.sum()) == 31
        trace = controller.layers[-1]
        assert {k: trace[k] for k in reference[3]["allocation"]} == reference[3]["allocation"]


@pytest.mark.parametrize("field,value", [("frames", [0, 2]), ("per_frame_spatial_keys", [4096, 32, 4096]),
                                         ("pointer_keys", 8), ("selected_spatial_keys", 8224)])
def test_any_count_or_frame_mismatch_fails(field, value):
    controller, kwargs, reference = setup_matched()
    reference[3]["allocation"][field] = value
    with pytest.raises(RuntimeError, match="differ from C2"):
        controller._before(0, SimpleNamespace(training=False), (), kwargs)


def test_c3_uses_past_reference_counts_not_current_prediction_or_gt():
    controller, kwargs, reference = setup_matched()
    # Current C2 selector cannot affect the current read: only previous frames.
    reference[3]["retained_tokens"] = 4000
    attention = SimpleNamespace(training=False)
    for layer in range(4):
        controller._before(layer, attention, (), kwargs)
    features = torch.full((1, 64, 64, 64), 3.)
    stored = {"maskmem_features": features, "maskmem_pos_enc": [torch.zeros_like(features)],
              "pred_masks": torch.full((1, 1, 256, 256), -10.)}
    state = {"output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {3: stored}}}}
    record = controller.capture(state, 3)
    assert record["region_counts"] == [4096, 0, 0]  # own native prediction is empty
    assert record["retained_tokens"] == 4000  # reference count, not own occupancy
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(record["keep_bits_hex"]), dtype=np.uint8), bitorder="big")
    assert len(bits) == 4096 and int(bits.sum()) == 4000
    assert record["allocation"] == reference[3]["allocation"]


def test_invalid_random_count_rejected():
    for count in (-1, 4097, 0.5):
        with pytest.raises(ValueError):
            random_cells(count, "seq2", 1, 20261025)


def test_reference_requires_complete_unique_frames_and_valid_selector_bits(tmp_path):
    import json
    def row(index, count):
        mask = random_cells(count, "seq2", index, 20261025).numpy()
        return {"frame": index, "trace": {"frame": index, "condition": "C2", "retained_tokens": count,
                "keep_bits_hex": np.packbits(mask).tobytes().hex()}}
    path = tmp_path/"integrity.jsonl"
    rows = [row(3, 20), row(4, 0), row(5, 100)]
    path.write_text("\n".join(map(json.dumps, rows)))
    assert sorted(load_reference(path)) == [3, 4, 5]
    for invalid in ([rows[0], rows[2]], [rows[0], rows[0]]):
        path.write_text("\n".join(map(json.dumps, invalid)))
        with pytest.raises(ValueError):
            load_reference(path)
    rows[0]["trace"]["retained_tokens"] = 21
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="bits/count"):
        load_reference(path)


def test_primary_statistics_average_all_seeds_with_sequence_as_unit():
    seeds = [20261025, 20261026, 20261027]
    baseline = {"results": {"C0": {"a": {"present_dice": .7}, "b": {"present_dice": .9}}}, "C1_present_dice": .75}
    results = {"C2": {"a": {"present_dice": .8}, "b": {"present_dice": .85}}, "C3": {}}
    for seed, pair in zip(seeds, [(.5, .5), (.7, .8), (.9, .8)]):
        results["C3"][str(seed)] = {s: {"present_dice": score} for s, score in zip(("a", "b"), pair)}
    summary = primary_statistics(results, baseline, seeds)
    assert summary["C2_present_dice"] == pytest.approx(.825)
    assert summary["C3_seed_mean_present_dice"] == pytest.approx(.7)
    assert [r["mean_difference"] for r in summary["contrasts"]] == pytest.approx([.025, .125])
    assert all(r["n_sequences"] == 2 and r["holm2_p"] >= r["wilcoxon_p"] for r in summary["contrasts"])
    del results["C3"][str(seeds[0])]["a"]
    with pytest.raises(ValueError, match="Incomplete"):
        primary_statistics(results, baseline, seeds)

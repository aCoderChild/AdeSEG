import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from datasets.video import ManifestFrame
from modeling.region_tokens import region_memory_mask, reliability_bias
from modeling.utility_memory import UtilityController, UtilityHead
from scripts.analyze_region_experiments import (
    category_weights,
    method_comparisons,
    paired_effects,
    persistent_effect_rows,
    freeze_method,
    episode_frames,
)
from scripts.run_region_experiments import (
    lean_anchor_similarity,
    require_gate,
    rollout,
    validate_method_split,
)
from scripts.run_region_token_pilot import swap_replacements
from external.MedSAM2.sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)


def test_corruption_is_cumulative_at_memory_resolution_and_never_changes_prompt():
    mask = np.zeros((128, 128), bool)
    mask[20:108, 20:108] = True
    assert np.array_equal(region_memory_mask(mask, "c2", 2, True), mask)
    assert np.array_equal(region_memory_mask(mask, "c0", 2), mask)
    c1 = region_memory_mask(mask, "c1", 2)
    assert c1.shape == (64, 64)
    assert c1.sum() == (44 - 6) ** 2
    c2 = region_memory_mask(mask, "c2", 2)
    assert c2.sum() == c1.sum()
    assert not np.array_equal(c1, c2)


def test_swaps_exclude_same_sequence_and_preserve_exact_position_and_budget(tmp_path):
    labels = np.full(4096, 3)
    labels[[2, 8]] = 0
    values = np.arange(4096 * 64, dtype=np.float32).reshape(4096, 64)
    donors = []
    for video, categories in (("target", labels), ("other", labels.copy())):
        path = tmp_path / f"{video}.npz"
        np.savez(path, values=values, categories=categories.reshape(64, 64))
        donors.append({"path": str(path), "video": video, "relative_time": 0.5})
    units = [{"start": 0, "end": 4096, "frame_idx": 4}]
    result, used = swap_replacements(
        units, np.array([2, 8]), labels, donors, "target", 0.5
    )
    mask, replacement = result[4]
    assert used == ["other"]
    assert mask.sum() == 2
    assert torch.equal(replacement[8], torch.from_numpy(values[8]))
    assert (
        swap_replacements(units, np.array([2, 8]), labels, donors[:1], "target", 0.5)[0]
        is None
    )


def test_memory_swap_changes_keys_and_values_but_not_positions_or_pointer_tokens():
    class Recorder(nn.Module):
        def forward(self, q, k, v, **kwargs):
            self.key, self.value = k.detach().clone(), v.detach().clone()
            return torch.zeros_like(q)

    layer = MemoryAttentionLayer(
        "relu", Recorder(), 4, 8, 0, False, True, False, Recorder()
    )
    module = MemoryAttention(4, True, layer, 1, batch_first=True).eval()
    controller = UtilityController(UtilityHead(4, 4, 8))
    controller.enabled = False
    module.utility_controller = controller
    memory = torch.randn(6, 1, 4)
    position = torch.randn_like(memory)
    query = torch.randn(4, 1, 4)
    units = [
        {
            "start": 0,
            "end": 4,
            "frame_idx": 0,
            "conditioning": True,
            "temporal_distance": 1,
        }
    ]
    controller.token_memory_replacements = {
        0: (torch.tensor([False, True, False, False]), torch.full((4, 4), 7.0))
    }
    module(
        query,
        memory,
        torch.zeros_like(query),
        position,
        memory_units=units,
        utility_query_context={"global": query.mean(0), "foreground": query.mean(0)},
    )
    recorder = module.layers[0].cross_attn_image
    expected = memory.transpose(0, 1).clone()
    expected[:, 1] = 7
    assert torch.equal(recorder.value, expected)
    assert torch.equal(recorder.key, expected + position.transpose(0, 1))
    assert torch.equal(recorder.value[:, 4:], memory.transpose(0, 1)[:, 4:])


def test_reliability_bias_random_control_preserves_norm_and_lambda_zero_is_noop():
    unit = {
        "frame_idx": 0,
        "predicted_iou": 0.8,
        "object_score": 2.0,
        "temporal_distance": 2,
    }
    maps = {
        0: {
            "categories": np.arange(4).reshape(2, 2),
            "artifact": np.array([[0, 1], [0, 0]], bool),
        }
    }
    weights = [2.0, 1.0, 0.5, 0.25]
    full = reliability_bias([unit], maps, weights, 1)[0]
    random = reliability_bias([unit], maps, weights, 1, "random", seed=2)[0]
    assert np.array_equal(np.sort(full), np.sort(random))
    assert reliability_bias([unit], maps, weights, 0) == {}
    no_artifact = reliability_bias([unit], maps, weights, 1, "artifact_off")[0]
    assert full[1] < no_artifact[1]
    del unit["predicted_iou"]
    with pytest.raises(ValueError, match="requires recorded"):
        reliability_bias([unit], maps, weights, 1)


def test_gate_rejects_duplicated_noop_as_restore_and_incomplete_e1(tmp_path):
    (tmp_path / "sanity.jsonl").write_text(
        json.dumps({"passed": True, "zero_bias_exact": True}) + "\n"
    )
    (tmp_path / "decision.json").write_text(
        json.dumps(
            {
                "stage": "E1",
                "all_required_sequences": True,
                "gate_pass": True,
                "decision": "proceed_to_E2",
            }
        )
    )
    with pytest.raises(ValueError, match="exactness"):
        require_gate(tmp_path)


def test_method_split_and_weight_selection_reject_test_outcomes():
    with pytest.raises(ValueError, match="dev/test"):
        validate_method_split("polypgen", False, ["seq16"], {"seq16": "seq16"}, [])
    with pytest.raises(ValueError, match="dev/test"):
        validate_method_split("cholecseg8k", False, ["clip"], {"clip": "video17"}, [])
    with pytest.raises(ValueError, match="dev data"):
        category_weights([{"video": "seq16", "dataset": "polypgen"}])


def test_lambda_freezing_requires_the_complete_grid_and_completed_dev_trials(tmp_path):
    (tmp_path / "lambda_0.json").write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "evaluation_ready": False,
                "scale": 0.0,
                "weights": [1, 1, 1, 1],
            }
        )
    )
    with pytest.raises(ValueError, match="full fixed lambda grid"):
        freeze_method([], tmp_path)


def test_component_delivery_rejects_incomplete_matched_controls(tmp_path):
    from scripts.collect_region_results import validate_study_shard

    study = tmp_path / "study"
    unit = study / "seq2__object1"
    unit.mkdir(parents=True)
    for folder in (tmp_path, study):
        (folder / "tracking.json").write_text(
            json.dumps({"mode": "online", "run_id": "test"})
        )
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete"}))
    (study / "status.json").write_text(
        json.dumps({"state": "complete", "stage": "E2", "completed_units": [unit.name]})
    )
    (unit / "status.json").write_text(
        json.dumps({"state": "complete", "fresh_e0_pass": True, "rows": 1})
    )
    (unit / "frames.jsonl").write_text(
        json.dumps(effect_row("seq2", 7, "boundary", 0.1)) + "\n"
    )
    receipt = validate_study_shard(tmp_path)
    assert not receipt["delivery_valid"]
    assert not receipt["full_study_complete"]
    assert any("matched-control matrix" in problem for problem in receipt["problems"])


def effect_row(video, frame, category, harm, seed=0):
    return {
        "video": video,
        "frame": frame,
        "category": category,
        "seed": seed,
        "bank": "recent",
        "dataset": "polypgen",
        "state": "complete",
        "k": 10,
        "mode": "single_step",
        "harm": {
            "dice": harm,
            "boundary_f": harm,
            "iou": harm,
            "hd95": None,
            "low_resolution_dice": harm,
        },
        "harm_per_token": {"boundary_f": harm / 10},
        "baseline": {"dice": 0.8},
    }


def test_e2_pairs_feasible_frames_and_averages_frames_before_sequences():
    rows = []
    for video, frames in (("seq2", [0, 1, 2]), ("seq3", [0])):
        for frame in frames:
            rows.append(effect_row(video, frame, "boundary", 0.1))
            for seed in range(5):
                rows.append(effect_row(video, frame, "random", 0.0, seed))
    # No boundary comparator exists for frame 999: random-only rows must not
    # change either sequence mean or the null on matched frames.
    for seed in range(5):
        rows.append(effect_row("seq3", 999, "random", 0.99, seed))
    effects = paired_effects(rows)
    assert effects
    assert all(
        e["mean_excess_harm"] == pytest.approx(0.1) and e["n_sequences"] == 2
        for e in effects.values()
    )
    with pytest.raises(ValueError, match="Duplicate"):
        paired_effects(rows + [copy.deepcopy(rows[0])])


def test_method_failure_episodes_are_fixed_by_baseline_for_all_comparators():
    rows = []
    for video in ("seq16", "seq17"):
        for t, dice in enumerate([0.8, 0.4, 0.3, 0.2, 0.9]):
            for arm, gain in (("baseline", 0), ("full", 0.1), ("SAM2Long", 0.02)):
                rows.append(
                    {
                        "video": video,
                        "frame": t,
                        "category": arm,
                        "state": "complete",
                        "bank": None,
                        "dataset": "polypgen",
                        "metrics": {
                            "dice": dice + gain,
                            "boundary_f": dice + gain,
                            "iou": dice + gain,
                        },
                    }
                )
    result = method_comparisons(rows)
    full_vs_long = [
        v for k, v in result.items() if '"SAM2Long", "failure_episode", "dice"' in k
    ]
    assert len(full_vs_long) == 1
    assert full_vs_long[0]["mean_gain"] == pytest.approx(0.08)
    assert full_vs_long[0]["n_sequences"] == 2


def test_missing_annotation_breaks_failure_episodes_and_is_not_recovery():
    from evaluation.utility_retrieval import temporal_summary

    assert episode_frames({1: 0.1, 2: 0.1, 4: 0.1}) == set()
    assert episode_frames({1: 0.1, 2: 0.1, 3: 0.1, 5: 0.9}) == {1, 2, 3}
    result = temporal_summary([0.1, 0.1, 0.1, None])
    assert result["failure_rate"] == 1.0
    assert result["recovery_rate"] == 0.0
    assert temporal_summary([None])["recovery_rate"] is None


def test_official_lean_maxcos_methods_are_reused_without_a_second_sam2_import():
    memory = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]])
    units = [
        {"frame_idx": 0, "conditioning": True, "start": 0, "end": 2},
        {"frame_idx": 1, "conditioning": False, "start": 2, "end": 4},
    ]
    result = lean_anchor_similarity(
        memory, units, {0: np.array([0, 1]), 1: np.array([0, 1])}
    )
    assert result["recent"]["boundary"] == pytest.approx(2**-0.5)
    assert result["recent"]["foreground"] == 0


def test_persistent_effects_exclude_infeasible_arms_without_reducing_budget():
    baseline = {
        "video": "seq2",
        "frame": 7,
        "dataset": "polypgen",
        "category": "baseline",
        "metrics": {
            "dice": 0.8,
            "boundary_f": 0.7,
            "iou": 0.6,
            "hd95": 1.0,
            "low_resolution_dice": 0.8,
        },
    }
    altered = {
        **baseline,
        "bank": "recent",
        "category": "boundary",
        "k": 10,
        "feasible": False,
        "metrics": {**baseline["metrics"], "dice": 0.6},
    }
    row = persistent_effect_rows([baseline, altered])[0]
    assert row["state"] == "skipped" and row["k"] == 10
    assert row["harm"]["dice"] == pytest.approx(0.2)


def test_persistent_rollout_uses_independent_states_and_reuses_features(
    tmp_path, monkeypatch
):
    import scripts.run_region_experiments as study

    image = np.zeros((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 1
    frames = []
    for t in range(9):
        ip, mp = tmp_path / f"{t}.jpg", tmp_path / f"{t}.png"
        Image.fromarray(image).save(ip)
        Image.fromarray(mask).save(mp)
        frames.append(ManifestFrame("seq2", t, ip, mp))
    controller = UtilityController(UtilityHead(4, 2, 8))
    controller.enabled = False

    class Predictor:
        device = "cpu"
        memory_attention = SimpleNamespace(capture_attention=False)
        feature_computations = 0
        states = []

        def _run_single_frame_inference(self, **kwargs):
            raise AssertionError("The zero-bias arm should use native propagation")

        def _get_image_feature(self, state, frame_idx, batch_size):
            if frame_idx not in state["cached_features"]:
                self.feature_computations += 1
                state["cached_features"] = {
                    frame_idx: (torch.ones(1), {"features": torch.ones(1)})
                }
            return state["cached_features"][frame_idx]

        def init_state(self, **kwargs):
            state = {
                "cached_features": {},
                "output_dict": {"non_cond_frame_outputs": {}},
            }
            self.states.append(state)
            return state

        def add_new_mask(self, *args, **kwargs):
            pass

        def propagate_in_video(self, state, start_frame_idx):
            for t in range(9):
                self._get_image_feature(state, t, 1)
                if t >= 7 and controller.token_bias_policy:
                    assert (
                        controller.token_bias_policy(
                            [{"frame_idx": t - 1, "temporal_distance": 1}]
                        )
                        == {}
                    )
                logits = torch.from_numpy(mask.astype(np.float32) * 20 - 10)[None, None]
                state["output_dict"]["non_cond_frame_outputs"][t] = {
                    "pred_masks": logits
                }
                yield t, [1], logits

        def reset_state(self, state):
            state["reset"] = True

    predictor = Predictor()
    monkeypatch.setattr(study, "log_results_artifact", lambda *args: None)
    monkeypatch.setattr(
        study,
        "metrics",
        lambda logits, target, void, decoder_logits: {
            "dice": 1.0,
            "boundary_f": 1.0,
            "iou": 1.0,
        },
    )
    config = {
        "stage": "E6_dev",
        "dataset": "polypgen",
        "label_id": 1,
        "label_ids": [1],
        "video": "seq2",
        "split": "test",
        "category_source": "deployable",
        "intervention": "attention_mask",
        "weights": [1, 1, 1, 1],
        "scale": 0.0,
    }
    run = SimpleNamespace(log=lambda *args: None)
    arms = rollout(
        predictor,
        controller,
        frames,
        {"initialization": "first_labeled_frame"},
        config,
        tmp_path,
        run,
    )
    assert arms == 2
    assert len(predictor.states) == 2 and predictor.states[0] is not predictor.states[1]
    assert all(s["reset"] for s in predictor.states)
    assert predictor.feature_computations == 9
    assert not list(tmp_path.glob("region_feature_cache_*"))

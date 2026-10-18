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
from modeling.memory_attention import UtilityController
from scripts.analyze_region_experiments import (
    category_weights,
    method_comparisons,
    paired_effects,
    persistent_effect_rows,
    freeze_method,
    episode_frames,
    dev_row,
    attention_agreement,
)
from scripts.run_region_experiments import (
    lean_anchor_similarity,
    require_gate,
    rollout,
    validate_method_split,
    cholec_method_split,
)
from scripts.run_region_token_pilot import (
    swap_replacements,
    capture_memory_read,
    mutate_spatial_memory,
    memory_read_checks,
)
from external.MedSAM2.sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)


def test_analysis_creates_new_report_parent_before_recording(tmp_path, monkeypatch):
    import sys
    from scripts import analyze_region_experiments as analyzer
    from scripts import run_region_experiments as runner

    source = tmp_path / "frames.jsonl"
    source.write_text("")
    output = tmp_path / "new_report" / "analysis.json"
    monkeypatch.setattr(runner, "validate_output", lambda path: path)
    monkeypatch.setattr(analyzer, "write_tables_and_figures", lambda *args: None)
    publications = []
    monkeypatch.setattr(
        analyzer, "publish_analysis", lambda *args: publications.append(args)
    )
    monkeypatch.setattr(
        sys, "argv", ["analyzer", "--input", str(source), "--output", str(output)]
    )
    analyzer.main()
    assert output.is_file()
    assert json.loads(output.read_text())["full_study_complete"] is False
    assert publications[0][0] == output.parent


def test_restore_control_observes_changed_memory_when_absence_gate_hides_masks():
    module = nn.Identity()
    bank = {"cond_frame_outputs": {0: {"maskmem_features": torch.ones(1, 4, 2, 2)}}}
    saved = copy.deepcopy(bank)

    def inference():
        module(bank["cond_frame_outputs"][0]["maskmem_features"])
        # SAM2's native absent-object branch can return fixed mask logits.
        return torch.full((1, 1, 4, 4), -1024.0)

    prediction, read = capture_memory_read(module, inference)
    assert not module._forward_hooks
    reads = {
        name: {"read": read.clone()}
        for name in (
            "baseline",
            "bare",
            "empty",
            "zero",
            "value_noop",
            "memory_noop",
            "full_tokens",
            "full_frame",
        )
    }
    mutate_spatial_memory(bank)
    altered, reads["mutated"] = capture_memory_read(module, inference)
    reads["mutated"] = {"read": reads["mutated"]}
    assert torch.equal(prediction, altered)
    assert torch.equal(
        saved["cond_frame_outputs"][0]["maskmem_features"], torch.ones(1, 4, 2, 2)
    )
    bank = copy.deepcopy(saved)
    restored, read = capture_memory_read(module, inference)
    reads["restored"] = {"read": read}
    assert torch.equal(restored, prediction)
    assert all(memory_read_checks(reads).values())
    reads["restored"] = reads["mutated"]
    assert not memory_read_checks(reads)["independent_memory_restore_exact"]
    reads["mutated"] = reads["baseline"]
    assert not memory_read_checks(reads)["mutation_changed_memory_read"]


def test_memory_read_capture_removes_hooks_on_failure_and_rejects_unused_memory():
    module = nn.Identity()

    def broken():
        raise ValueError("inference failed")

    with pytest.raises(ValueError, match="inference failed"):
        capture_memory_read(module, broken)
    assert not module._forward_hooks
    with pytest.raises(RuntimeError, match="observed 0"):
        capture_memory_read(module, lambda: torch.ones(1))
    assert not module._forward_hooks


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
    controller = UtilityController()
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


def test_gate_v2_requires_readout_mutation_and_restoration_not_final_mask_change(
    tmp_path,
):
    decision = {
        "stage": "E1",
        "all_required_sequences": True,
        "gate_pass": True,
        "decision": "proceed_to_E2",
    }
    (tmp_path / "decision.json").write_text(json.dumps(decision))
    check = dict.fromkeys(
        (
            "passed",
            "detached_noop_exact",
            "empty_token_mask_exact",
            "zero_bias_exact",
            "independent_restore_exact",
            "full_token_equals_frame",
            "noop_memory_reads_exact",
            "mutation_changed_memory_read",
            "independent_memory_restore_exact",
            "full_token_memory_equals_frame",
        ),
        True,
    )
    check.update(
        restore_control_version=2,
        mutation_changed_prediction=False,
        attention_reconstruction_max_abs=0.0,
    )
    sanity = tmp_path / "sanity.jsonl"
    sanity.write_text(json.dumps(check) + "\n")
    assert require_gate(tmp_path)["sanity_files"]
    for key, value in (
        ("mutation_changed_memory_read", False),
        ("independent_memory_restore_exact", False),
        ("noop_memory_reads_exact", False),
        ("full_token_memory_equals_frame", False),
        ("attention_reconstruction_max_abs", float("nan")),
    ):
        sanity.write_text(json.dumps({**check, key: value}) + "\n")
        with pytest.raises(ValueError, match="exactness"):
            require_gate(tmp_path)


def test_method_split_and_weight_selection_reject_test_outcomes():
    with pytest.raises(ValueError, match="dev/test"):
        validate_method_split("polypgen", False, ["seq16"], {"seq16": "seq16"}, [])
    with pytest.raises(ValueError, match="dev/test"):
        validate_method_split("cholecseg8k", False, ["clip"], {"clip": "video55"}, [])
    with pytest.raises(ValueError, match="dev data"):
        category_weights([{"video": "seq16", "dataset": "polypgen"}])


def test_cholec_actual_source_split_is_exhaustive_and_clip_safe():
    actual = {
        "video01",
        "video09",
        "video12",
        "video17",
        "video18",
        "video20",
        "video24",
        "video25",
        "video26",
        "video27",
        "video28",
        "video35",
        "video37",
        "video43",
        "video48",
        "video52",
        "video55",
    }
    split = cholec_method_split()
    dev = set(split["dev_source_videos"])
    test = set(split["test_source_videos"])
    assert len(dev) == 12 and len(test) == 5
    assert not dev & test
    assert dev | test == actual
    assert sorted(actual)[:12] == split["dev_source_videos"]
    for source in sorted(actual):
        clips = [f"{source}_clip0", f"{source}_clip1"]
        sources = dict.fromkeys(clips, source)
        validate_method_split("cholecseg8k", source in test, clips, sources, [])
        with pytest.raises(ValueError, match="dev/test"):
            validate_method_split("cholecseg8k", source in dev, clips, sources, [])
        for label in (1, 5, 12):
            assert dev_row(
                {"dataset": "cholecseg8k", "source_video": source, "label_id": label}
            ) == (source in dev)


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
    from evaluation.statistics import temporal_summary

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


def test_combined_persistent_shards_pair_only_with_their_own_baseline():
    rows = []
    for operator, score in (("attention_mask", 0.8), ("swap", 0.6)):
        for band in (0, 1):
            baseline = {
                "video": "seq2",
                "frame": 7,
                "dataset": "polypgen",
                "experiment_stage": "E2",
                "rollout_operator": operator,
                "band": band,
                "category": "baseline",
                "bank": None,
                "metrics": {"dice": score + band * 0.1},
            }
            rows += [
                baseline,
                {
                    **baseline,
                    "category": "boundary",
                    "bank": "recent",
                    "k": 10,
                    "feasible": True,
                    "metrics": {"dice": score + band * 0.1 - 0.05},
                },
            ]
    effects = persistent_effect_rows(rows)
    assert len(effects) == 4
    assert all(r["harm"]["dice"] == pytest.approx(0.05) for r in effects)
    assert method_comparisons(rows) == {}
    with pytest.raises(ValueError, match="Duplicate persistent baseline"):
        persistent_effect_rows(rows + [rows[0]])
    with pytest.raises(ValueError, match="no matching baseline"):
        persistent_effect_rows(rows[1:])


def test_attention_agreement_keeps_direct_and_accumulated_effects_separate():
    from modeling.region_tokens import CATEGORIES

    rows = []
    mass = {name: i + 1.0 for i, name in enumerate(CATEGORIES)}
    attention = [{q: mass for q in ("foreground", "boundary", "background")}]
    for mode, sign in (("single_step", 1), ("persistent", -1)):
        for name in CATEGORIES:
            rows.append(
                {
                    "video": "seq2",
                    "frame": 7,
                    "state": "complete",
                    "bank": "recent",
                    "category": name,
                    "mode": mode,
                    "attention": attention,
                    "category_counts": dict.fromkeys(CATEGORIES, 1),
                    "harm_per_token": {"boundary_f": sign * mass[name]},
                }
            )
    result = attention_agreement(rows)["agreement"]
    assert result
    for key, value in result.items():
        assert value["rho"] == pytest.approx(1 if "single_step" in key else -1)
    assert any("single_step" in key for key in result)
    assert any("persistent" in key for key in result)


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
    controller = UtilityController()
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

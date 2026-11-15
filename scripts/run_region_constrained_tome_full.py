"""Run the frozen native/generic/region-ToMe PolypGen exploratory cohort once.

Requires the seq2/seq3 engineering gate. All experiment artifacts stay on Drive.
"""

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.false_alarm_statistics import propagation_summary
from scripts.run_region_constrained_tome_gate import (
    CALIBRATED_CONDITIONS,
    CONDITIONS,
    PROTOCOL,
    matched_plan_from_regional,
    native_reference,
    run_condition,
)
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write


def dump_plan(path, plan):
    serializable = {
        str(target): {
            str(frame): {str(window): int(count) for window, count in windows.items()}
            for frame, windows in memory_frames.items()
        }
        for target, memory_frames in plan.items()
    }
    encoded = json.dumps(serializable, separators=(",", ":")).encode()
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(encoded)


def load_plan(path):
    with gzip.open(path, "rt") as stream:
        serialized = json.load(stream)
    return {
        int(target): {
            int(frame): {int(window): int(count) for window, count in windows.items()}
            for frame, windows in memory_frames.items()
        }
        for target, memory_frames in serialized.items()
    }


def compact_rows(rows):
    compact = []
    for row in rows:
        result = {
            key: row[key]
            for key in (
                "frame", "present", "scores", "absent_fp", "prediction_sha256",
                "stored_sha256", "object_score",
            )
            if key in row
        }
        reads = row.get("attention_reads")
        if reads:
            result["attention_summary"] = [
                {
                    "layer": call["layer"],
                    "spatial_keys_before": call["spatial_keys_before"],
                    "spatial_keys_after": call["spatial_keys_after"],
                    "pointer_keys": call["pointer_keys"],
                    "spatial_key_mass_sum": call["spatial_key_mass_sum"],
                    "key_mass_sum": call["key_mass_sum"],
                    "memory_frames": [item["frame"] for item in call["frames"]],
                }
                for call in reads
            ]
        compact.append(result)
    return compact


def validate_generic_against_plan(rows, plan):
    """Verify the generic arm used C2's frozen budget in every target/layer/window."""
    checks = 0
    for row in rows:
        target = row["frame"]
        if target not in plan:
            raise RuntimeError(f"No regional budget was frozen for target {target}")
        calls = row.get("attention_reads", [])
        if [call["layer"] for call in calls] != [0, 1, 2, 3]:
            raise RuntimeError(f"Generic control did not use all four layers at frame {target}")
        expected_frames = plan[target]
        for call in calls:
            observed_frames = {}
            for memory_frame in call["frames"]:
                window_rows = memory_frame["windows"]
                expected_windows = expected_frames.get(memory_frame["frame"])
                if expected_windows is None:
                    raise RuntimeError("Generic control read a memory frame absent from the regional budget")
                observed = {item["window"]: item["merges"] for item in window_rows}
                if observed != expected_windows:
                    raise RuntimeError(
                        f"Generic per-window merge counts differ at target {target}, "
                        f"memory frame {memory_frame['frame']}, layer {call['layer']}"
                    )
                observed_frames[memory_frame["frame"]] = observed
                checks += len(observed)
            if observed_frames.keys() != expected_frames.keys():
                raise RuntimeError(f"Generic memory-frame coverage differs at target {target}")
    return checks


def score_summary(rows):
    present = [row for row in rows if row["present"]]
    absent = [row for row in rows if not row["present"]]
    propagation = propagation_summary(rows)
    detected_returns = [event["delay_sampled_frames"] for event in propagation["returns"] if event["detected"]]
    return {
        "targets": len(rows),
        "present_targets": len(present),
        "gt_empty_targets": len(absent),
        "present_dice": sum(row["scores"]["dice"] for row in present) / len(present) if present else None,
        "present_iou": sum(row["scores"]["iou"] for row in present) / len(present) if present else None,
        "present_boundary_f": sum(row["scores"]["boundary_f"] for row in present) / len(present) if present else None,
        "gt_empty_false_alarms": sum(bool(row["absent_fp"]) for row in absent),
        "gt_empty_false_alarm_rate": sum(bool(row["absent_fp"]) for row in absent) / len(absent) if absent else None,
        "return_count": propagation["return_count"],
        "undetected_returns": propagation["undetected_returns"],
        "mean_detected_return_delay_sampled_frames": (
            sum(detected_returns) / len(detected_returns) if detected_returns else None
        ),
    }


def load_completed(folder, expected_targets, setup_sha, *, require_plan=False):
    summary_path = folder / "summary.json"
    frames_path = folder / "frames.json"
    plan_path = folder / "merge_plan.json.gz"
    if not summary_path.is_file() or not frames_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if (
        not summary.get("complete")
        or summary.get("setup_sha256") != setup_sha
        or summary.get("targets") != expected_targets
    ):
        return None
    rows = json.loads(frames_path.read_text())
    if len(rows) != expected_targets:
        raise RuntimeError(f"Completed output has wrong target count: {folder}")
    if require_plan and not plan_path.is_file():
        return None
    return rows


def sequence_summary(rows, run_summary):
    metrics = score_summary(rows)
    metrics.update({
        "wall_seconds": run_summary["wall_seconds"],
        "parameter_digest": run_summary["parameter_digest"],
        "parameters_unchanged": run_summary["parameters_unchanged"],
    })
    if rows and "attention_summary" in rows[0]:
        reads = [read for row in rows for read in row.get("attention_summary", [])]
        before = sum(read["spatial_keys_before"] for read in reads)
        after = sum(read["spatial_keys_after"] for read in reads)
        metrics.update({
            "memory_attention_layer_calls": len(reads),
            "mean_spatial_keys_before": before / len(reads),
            "mean_spatial_keys_after": after / len(reads),
            "spatial_key_retention_fraction": after / before,
            "mean_pointer_keys": sum(read["pointer_keys"] for read in reads) / len(reads),
        })
    return metrics


def make_final_summary(sequence_results, sequence_names, matched_checks, arms=CONDITIONS):
    means = {
        arm: sum(sequence_results[name][arm]["present_dice"] for name in sequence_names) / len(sequence_names)
        for arm in arms
    }
    contrasts = {
        "regional_minus_native": {
            name: sequence_results[name]["region_constrained_tome"]["present_dice"]
            - sequence_results[name]["native"]["present_dice"]
            for name in sequence_names
        },
        "regional_minus_generic": {
            name: sequence_results[name]["region_constrained_tome"]["present_dice"]
            - sequence_results[name]["generic_local_tome"]["present_dice"]
            for name in sequence_names
        },
    }
    if "calibrated_region_constrained_tome" in arms:
        contrasts.update({
            "calibrated_regional_minus_native": {
                name: sequence_results[name]["calibrated_region_constrained_tome"]["present_dice"]
                - sequence_results[name]["native"]["present_dice"]
                for name in sequence_names
            },
            "calibrated_regional_minus_calibrated_generic": {
                name: sequence_results[name]["calibrated_region_constrained_tome"]["present_dice"]
                - sequence_results[name]["calibrated_generic_local_tome"]["present_dice"]
                for name in sequence_names
            },
        })
    p_values, tests = {}, {}
    for contrast, values in contrasts.items():
        boot = sequence_bootstrap_interval(values, seed=20261105, resamples=10_000)
        test = paired_wilcoxon(
            {name: values[name] for name in sequence_names},
            {name: 0.0 for name in sequence_names},
        )
        p_values[contrast] = test["p"]
        tests[contrast] = {
            "n_sequences": len(values),
            "mean_difference": sum(values.values()) / len(values),
            "sequence_bootstrap_95ci": boot,
            "wilcoxon_two_sided_p": test["p"],
            "positive_sequence_wins": sum(value > 0 for value in values.values()),
            "ties": sum(value == 0 for value in values.values()),
        }
    adjusted = holm_adjust(p_values)
    for contrast in tests:
        tests[contrast]["holm_adjusted_p"] = adjusted[contrast]

    pooled_empty = {}
    for arm in arms:
        false_alarms = sum(sequence_results[name][arm]["gt_empty_false_alarms"] for name in sequence_names)
        empty_targets = sum(sequence_results[name][arm]["gt_empty_targets"] for name in sequence_names)
        pooled_empty[arm] = {
            "false_alarms": false_alarms,
            "gt_empty_targets": empty_targets,
            "rate": false_alarms / empty_targets if empty_targets else None,
        }
    primary = "calibrated_regional_minus_native" if "calibrated_region_constrained_tome" in arms else "regional_minus_native"
    control = "calibrated_regional_minus_calibrated_generic" if "calibrated_region_constrained_tome" in arms else "regional_minus_generic"
    c2_native, c2_generic = tests[primary], tests[control]
    treatment = "calibrated_region_constrained_tome" if "calibrated_region_constrained_tome" in arms else "region_constrained_tome"
    meaningful = (
        c2_native["mean_difference"] >= 0.01
        and c2_generic["mean_difference"] > 0
        and c2_native["positive_sequence_wins"] >= 14
        and c2_native["sequence_bootstrap_95ci"][0] > 0
        and c2_generic["sequence_bootstrap_95ci"][0] > 0
        and c2_native["holm_adjusted_p"] < 0.05
        and c2_generic["holm_adjusted_p"] < 0.05
        and pooled_empty[treatment]["false_alarms"] <= pooled_empty["native"]["false_alarms"]
    )
    return {
        "scope": "Exploratory PolypGen full cohort; sequences are the paired statistical units.",
        "sequences": sequence_names,
        "n_sequences": len(sequence_names),
        "conditions": list(arms),
        "equal_sequence_mean_present_dice": means,
        "paired_contrasts": tests,
        "pooled_gt_empty_false_alarms": pooled_empty,
        "matched_budget_checks": matched_checks,
        "meaningful_screen_passed": bool(meaningful),
        "predeclared_screen": "Protocol criterion applied without tuning; failure means this method did not demonstrate the planned meaningful gain on this cohort.",
        "per_sequence": sequence_results,
    }


def write_results_markdown(path, summary):
    names = summary["sequences"]
    arms = summary["conditions"]
    labels = {
        "native": "Native SAM2",
        "region_constrained_tome": "Region-constrained ToMe",
        "generic_local_tome": "Matched generic ToMe",
        "calibrated_region_constrained_tome": "Calibrated regional ToMe",
        "calibrated_generic_local_tome": "Matched calibrated generic ToMe",
    }
    lines = [
        "# ToMe memory-read calibration: PolypGen exploratory run",
        "",
        "Independent seq2/seq3 engineering gates passed before this full-cohort run. PolypGen is exploratory because prior work informed method development.",
        "",
        "## Equal-sequence Dice",
        "",
        "| Sequence | " + " | ".join(labels[arm] for arm in arms) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(arms)) + "|",
    ]
    for name in names:
        values = [summary["per_sequence"][name][arm]["present_dice"] for arm in arms]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.append("| **Equal-sequence mean** | " + " | ".join(f"**{summary['equal_sequence_mean_present_dice'][arm]:.4f}**" for arm in arms) + " |")
    lines += ["", "## Planned paired contrasts", "", "| Contrast | Mean Dice difference | Sequence wins | 95% sequence bootstrap CI | Holm p |", "|---|---:|---:|---:|---:|"]
    for key, row in summary["paired_contrasts"].items():
        ci = row["sequence_bootstrap_95ci"]
        lines.append(f"| {key.replace('_', ' ')} | {row['mean_difference']:+.4f} | {row['positive_sequence_wins']}/{row['n_sequences']} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['holm_adjusted_p']:.4g} |")
    lines += ["", "## False alarms and compute", "", "| Condition | GT-empty false alarms / frames | Rate | Total model wall time (s) |", "|---|---:|---:|---:|"]
    for arm in arms:
        counts = summary["pooled_gt_empty_false_alarms"][arm]
        arm_times = [summary["per_sequence"][name][arm]["wall_seconds"] for name in names]
        total_time = None if any(value is None for value in arm_times) else sum(arm_times)
        rate = "N/A" if counts["rate"] is None else f"{counts['rate']:.4f}"
        time_text = "reused native outputs" if total_time is None else f"{total_time:.1f}"
        lines.append(f"| {labels[arm]} | {counts['false_alarms']} / {counts['gt_empty_targets']} | {rate} | {time_text} |")
    lines += ["", f"Predeclared meaningful-gain screen: **{'PASS' if summary['meaningful_screen_passed'] else 'FAIL'}**.", "", "No thresholds, confidence gates, or frame-selection rules were used. See `summary.json` and the per-sequence files for full metrics, attention-key counts, and provenance.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stat-calibrated", action="store_true",
        help="Run the frozen five-arm statistical-calibration study after its independent gate.",
    )
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume)
    try:
        conditions = CALIBRATED_CONDITIONS if args.stat_calibrated else CONDITIONS
        protocol_path = (
            ROOT / "protocols/stat_calibrated_tome_gate_v1.json"
            if args.stat_calibrated else ROOT / PROTOCOL
        )
        protocol = json.loads(protocol_path.read_text())
        protocol_sha = digest(protocol_path)
        full_scope_protocol = (
            json.loads((ROOT / PROTOCOL).read_text())
            if args.stat_calibrated else protocol
        )
        sequence_names = list(full_scope_protocol["scope"]["sequences"])
        gate_completion = json.loads((args.gate_dir / "completion.json").read_text())
        gate_result = json.loads((args.gate_dir / "gate.json").read_text())
        gate_setup = json.loads((args.gate_dir / "setup.json").read_text())
        if not gate_completion.get("passed"):
            raise RuntimeError("seq2/seq3 engineering gate did not pass")
        if gate_setup.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("The gate used a different protocol revision")
        if args.stat_calibrated:
            if gate_result.get("calibrated_merge_assignments_identical_to_uncalibrated") is not True:
                raise RuntimeError(
                    "Calibration efficacy blocked: gate did not verify exact cross-arm merge assignments"
                )
            replay = json.loads((args.gate_dir.parent / "independent_replay_receipt.json").read_text())
            if (
                replay.get("status") != "passed"
                or replay.get("fresh_process_frame_record_files") != 10
                or replay.get("identical_frame_record_files") != 10
            ):
                raise RuntimeError("The calibrated method lacks a complete exact independent replay")
        if (
            gate_setup.get("gate_script_sha256") != digest(ROOT / "scripts/run_region_constrained_tome_gate.py")
            or gate_setup.get("method_module_sha256") != digest(ROOT / "modeling/region_constrained_tome.py")
            or gate_setup.get("tome_merge_source_sha256") != digest(ROOT / "external/ToMe/tome/merge.py")
            or (args.stat_calibrated and gate_setup.get("calibration_module_sha256") != digest(ROOT / "modeling/stat_calibrated_tome.py"))
        ):
            raise RuntimeError("Gate code or upstream ToMe source changed after the engineering gate")
        if gate_result.get("sam2_revision") != protocol["model"]["revision"]:
            raise RuntimeError("The engineering gate used a different SAM2 revision")
        checkpoint_sha = digest(args.checkpoint)
        if checkpoint_sha != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from the frozen protocol")
        manifest = ROOT / protocol["scope"]["manifest"]
        manifest_sha = digest(manifest)
        videos = {
            **load_manifest(manifest, "dev"),
            **load_manifest(manifest, "test"),
        }
        if args.stat_calibrated:
            import scripts.run_author_baseline as baseline
            baseline.REPOSITORIES["sam2"] = (
                "external/sam2_diagnostic_reference", protocol["model"]["revision"]
            )
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["revision"]:
            raise RuntimeError("SAM2 author checkout differs from the frozen protocol")
        native_results_path = args.native_run / "native_results.json"
        native_results = json.loads(native_results_path.read_text())
        if native_results.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Native cohort reference checkpoint differs")
        if args.resume:
            setup_path = output / "setup.json"
            if not setup_path.is_file():
                raise RuntimeError("Cannot resume: Drive output has no setup receipt")
            setup = json.loads(setup_path.read_text())
            if any(setup.get(key) != value for key, value in {
                "protocol_sha256": protocol_sha,
                "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "native_results_sha256": digest(native_results_path),
                "gate_sha256": digest(args.gate_dir / "gate.json"),
                "full_runner_sha256": digest(Path(__file__)),
                "gate_script_sha256": digest(ROOT / "scripts/run_region_constrained_tome_gate.py"),
                "method_module_sha256": digest(ROOT / "modeling/region_constrained_tome.py"),
                "calibration_module_sha256": digest(ROOT / "modeling/stat_calibrated_tome.py") if args.stat_calibrated else None,
            }.items()):
                raise RuntimeError("Resume provenance differs from the original run")
            setup_sha = digest(setup_path)
        else:
            setup = {
                "protocol_sha256": protocol_sha,
                "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "native_results_sha256": digest(native_results_path),
                "gate_dir": str(args.gate_dir),
                "gate_sha256": digest(args.gate_dir / "gate.json"),
                "sam2_revision": gate_result["sam2_revision"],
                "tome_revision": protocol["prior_work"]["repository_revision"],
                "tome_merge_source_sha256": protocol["prior_work"]["merge_source_sha256"],
                "full_runner_sha256": digest(Path(__file__)),
                "gate_script_sha256": digest(ROOT / "scripts/run_region_constrained_tome_gate.py"),
                "method_module_sha256": digest(ROOT / "modeling/region_constrained_tome.py"),
                "calibration_module_sha256": digest(ROOT / "modeling/stat_calibrated_tome.py") if args.stat_calibrated else None,
                "device": "cpu",
                "dtype": "float32",
                "sequences": sequence_names,
                "conditions": list(conditions),
            }
            write(output / "protocol.json", protocol)
            write(output / "setup.json", setup)
            setup_sha = digest(output / "setup.json")

        sequence_results = {name: {} for name in sequence_names}
        matched_checks = 0
        for name in sequence_names:
            if name not in videos:
                raise RuntimeError(f"Sequence missing from manifest: {name}")
            frames = videos[name]
            prompt, reference, ref_receipt = native_reference(
                args.native_run, native_results, protocol, name, frames
            )
            expected_targets = len(reference)
            merge_plan = None
            for condition in conditions:
                folder = output / condition / name
                folder.mkdir(parents=True, exist_ok=True)
                plan_path = folder / "merge_plan.json.gz"
                if condition == "calibrated_region_constrained_tome":
                    merge_plan = None
                needs_plan = condition in ("region_constrained_tome", "calibrated_region_constrained_tome")
                completed = load_completed(
                    folder, expected_targets, setup_sha,
                    require_plan=needs_plan,
                )
                if completed is not None:
                    rows = completed
                    run_meta = json.loads((folder / "summary.json").read_text())
                    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
                        raise RuntimeError(f"Completed outputs have incorrect frame coverage: {name}/{condition}")
                    if condition == "native":
                        for row in rows:
                            expected_row = reference[row["frame"]]
                            if any(row[key] != expected_row[key] for key in (
                                "prediction_sha256", "stored_sha256", "present", "scores", "absent_fp"
                            )):
                                raise RuntimeError(f"Resumed native replay differs from archived reference: {name}/{row['frame']}")
                    if condition in ("region_constrained_tome", "calibrated_region_constrained_tome"):
                        merge_plan = load_plan(plan_path)
                        if len(merge_plan) != expected_targets:
                            raise RuntimeError(f"Saved regional plan incomplete for {name}")
                        if run_meta.get("merge_plan_sha256") != digest(plan_path):
                            raise RuntimeError(f"Saved regional merge-plan digest differs for {name}")
                    if condition in ("generic_local_tome", "calibrated_generic_local_tome"):
                        matched_checks += int(run_meta.get("matched_budget_checks", 0))
                else:
                    if condition in ("generic_local_tome", "calibrated_generic_local_tome"):
                        if merge_plan is None:
                            raise RuntimeError("Generic ToMe requires the completed regional merge plan")
                    if args.stat_calibrated and condition == "native":
                        # Reuse the already audited full-video native trajectory;
                        # no duplicate native propagation or checkpoint load.
                        rows = [reference[frame] for frame in sorted(reference)]
                        run_meta = {
                            "sequence": name,
                            "condition": condition,
                            "wall_seconds": None,
                            "parameter_digest": None,
                            "parameters_unchanged": None,
                            "reused_archived_native_trajectory": True,
                        }
                    else:
                        rows, run_meta = run_condition(
                            name,
                            condition,
                            frames,
                            prompt,
                            reference,
                            args.checkpoint,
                            protocol["model"]["configuration"],
                            repo,
                            folder,
                            merge_plan_by_target=merge_plan,
                        )
                    if condition in ("region_constrained_tome", "calibrated_region_constrained_tome"):
                        merge_plan = matched_plan_from_regional(rows)
                        dump_plan(plan_path, merge_plan)
                        run_meta["merge_plan_sha256"] = digest(plan_path)
                    budget_checks = 0
                    if condition in ("generic_local_tome", "calibrated_generic_local_tome"):
                        budget_checks = validate_generic_against_plan(rows, merge_plan)
                        matched_checks += budget_checks
                    rows = compact_rows(rows)
                    run_meta.update({
                        **score_summary(rows),
                        "matched_budget_checks": budget_checks,
                        "complete": True,
                        "sequence": name,
                        "condition": condition,
                        "setup_sha256": setup_sha,
                        "reference_receipt": ref_receipt,
                    })
                    write(folder / "frames.json", rows)
                    write(folder / "summary.json", run_meta)
                sequence_results[name][condition] = run_meta
                write(output / "progress.json", {
                    "sequence": name,
                    "condition": condition,
                    "targets_completed": expected_targets,
                    "sequence_count": len(sequence_results),
                    "sequence_total": len(sequence_names),
                })
            # Store sequence-level budget/provenance check receipt after all arms.
            (output / "sequence_completion").mkdir(parents=True, exist_ok=True)
            write(output / "sequence_completion" / f"{name}.json", {
                "sequence": name,
                "prompt": prompt,
                "targets": expected_targets,
                "conditions": list(conditions),
                "complete": True,
                "setup_sha256": setup_sha,
            })
        final = make_final_summary(sequence_results, sequence_names, matched_checks, conditions)
        write(output / "summary.json", final)
        write_results_markdown(output / "results.md", final)
        (output / "failure.json").unlink(missing_ok=True)
        write(output / "completion.json", {
            "passed": True,
            "scope": "full PolypGen exploratory cohort",
            "sequences": sequence_names,
            "conditions": list(conditions),
            "meaningful_screen_passed": final["meaningful_screen_passed"],
        })
    except BaseException as error:
        write(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()

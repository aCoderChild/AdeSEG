"""Run the frozen native/generic/soft-mask-ToMe PolypGen cohort after its gate."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.region_constrained_tome import author_tome
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_region_constrained_tome_full import load_completed, score_summary, sequence_summary
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_mask_aware_tome_gate import CONDITIONS, run_condition


PROTOCOL = "protocols/soft_mask_aware_tome_polypgen_v1.json"
GATE_SCRIPT = "scripts/run_soft_mask_aware_tome_gate.py"
METHOD_MODULE = "modeling/soft_mask_aware_tome.py"
TOME_SOURCE = "external/ToMe/tome/merge.py"


def compact_rows(rows):
    compact = []
    for row in rows:
        item = {
            key: row[key]
            for key in (
                "frame", "present", "scores", "absent_fp", "prediction_sha256",
                "stored_sha256", "object_score",
            )
            if key in row
        }
        reads = row.get("attention_reads", [])
        if reads:
            item["memory_mask_sources"] = reads[0]["mask_sources"]
            item["attention_summary"] = [
                {
                    "layer": call["layer"],
                    "spatial_keys_before": call["spatial_keys_before"],
                    "spatial_keys_after": call["spatial_keys_after"],
                    "pointer_keys": call["pointer_keys"],
                    "spatial_key_mass_sum": call["spatial_key_mass_sum"],
                    "key_mass_sum": call["key_mass_sum"],
                    "memory_frames": [memory["frame"] for memory in call["frames"]],
                }
                for call in reads
            ]
        compact.append(item)
    return compact


def budget_signature(rows):
    """Hash and validate every target/layer/frame/window budget in a run."""
    records = []
    for row in rows:
        reads = row.get("attention_reads", [])
        if [call["layer"] for call in reads] != [0, 1, 2, 3]:
            raise RuntimeError(f"Missing attention layer at target frame {row['frame']}")
        for call in reads:
            for memory in call["frames"]:
                for window in memory["windows"]:
                    if (
                        window["original_spatial_keys"] != 16
                        or window["merges"] != 8
                        or window["retained_spatial_keys"] != 8
                    ):
                        raise RuntimeError("Observed a memory window outside the frozen 8/16 budget")
                    records.append((
                        int(row["frame"]), int(call["layer"]), int(memory["frame"]),
                        int(window["window"]), int(window["merges"]),
                    ))
    records.sort()
    payload = json.dumps(records, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "target_layer_window_records": len(records),
        "all_windows_used_fixed_8_of_16_merges": True,
    }


def compare_budget_signatures(generic, aware):
    if generic["sha256"] != aware["sha256"]:
        raise RuntimeError("Generic and mask-aware arms used different per-window budgets")
    if generic["target_layer_window_records"] != aware["target_layer_window_records"]:
        raise RuntimeError("Generic and mask-aware budget audit coverage differs")
    return generic["target_layer_window_records"]


def make_final_summary(sequence_results, sequence_names, budget_checks):
    arms = CONDITIONS
    means = {
        arm: sum(sequence_results[name][arm]["present_dice"] for name in sequence_names)
        / len(sequence_names)
        for arm in arms
    }
    contrasts = {
        "soft_mask_aware_minus_native": {
            name: sequence_results[name]["soft_mask_aware_tome"]["present_dice"]
            - sequence_results[name]["native"]["present_dice"]
            for name in sequence_names
        },
        "soft_mask_aware_minus_generic": {
            name: sequence_results[name]["soft_mask_aware_tome"]["present_dice"]
            - sequence_results[name]["generic_local_tome"]["present_dice"]
            for name in sequence_names
        },
    }
    tests = {}
    for contrast, values in contrasts.items():
        bootstrap = sequence_bootstrap_interval(values, seed=20261105, resamples=10_000)
        test = paired_wilcoxon(
            {name: values[name] for name in sequence_names},
            {name: 0.0 for name in sequence_names},
        )
        tests[contrast] = {
            "n_sequences": len(values),
            "mean_difference": sum(values.values()) / len(values),
            "sequence_bootstrap_95ci": bootstrap,
            "wilcoxon_two_sided_p": test["p"],
            "positive_sequence_wins": sum(value > 0 for value in values.values()),
            "ties": sum(value == 0 for value in values.values()),
        }
    adjusted = holm_adjust({key: value["wilcoxon_two_sided_p"] for key, value in tests.items()})
    for contrast in tests:
        tests[contrast]["holm_adjusted_p"] = adjusted[contrast]

    pooled_empty = {}
    for arm in arms:
        false_alarms = sum(
            sequence_results[name][arm]["gt_empty_false_alarms"] for name in sequence_names
        )
        empty_targets = sum(
            sequence_results[name][arm]["gt_empty_targets"] for name in sequence_names
        )
        pooled_empty[arm] = {
            "false_alarms": false_alarms,
            "gt_empty_targets": empty_targets,
            "rate": false_alarms / empty_targets if empty_targets else None,
        }

    vs_native = tests["soft_mask_aware_minus_native"]
    vs_generic = tests["soft_mask_aware_minus_generic"]
    meaningful = (
        vs_native["mean_difference"] >= 0.01
        and vs_generic["mean_difference"] > 0
        and vs_native["positive_sequence_wins"] >= 14
        and vs_native["sequence_bootstrap_95ci"][0] > 0
        and vs_generic["sequence_bootstrap_95ci"][0] > 0
        and vs_native["holm_adjusted_p"] < 0.05
        and vs_generic["holm_adjusted_p"] < 0.05
        and pooled_empty["soft_mask_aware_tome"]["false_alarms"]
        <= pooled_empty["native"]["false_alarms"]
    )
    return {
        "scope": "Exploratory PolypGen cohort; sequences are paired statistical units.",
        "sequences": sequence_names,
        "n_sequences": len(sequence_names),
        "equal_sequence_mean_present_dice": means,
        "paired_contrasts": tests,
        "pooled_gt_empty_false_alarms": pooled_empty,
        "matched_budget_window_records": budget_checks,
        "meaningful_screen_passed": bool(meaningful),
        "predeclared_screen": (
            "At least +0.01 equal-sequence Dice over native, positive mean over generic, "
            "at least 14/21 sequence wins over native, positive lower 95% bounds and "
            "Holm-adjusted p<0.05 for both contrasts, and no increase in pooled false alarms. "
            "Applied once without retuning."
        ),
        "per_sequence": sequence_results,
    }


def write_results_markdown(path, summary):
    names = summary["sequences"]
    labels = {
        "native": "Native SAM2",
        "generic_local_tome": "Generic local ToMe",
        "soft_mask_aware_tome": "Soft mask-aware ToMe",
    }
    lines = [
        "# Soft mask-aware ToMe: PolypGen exploratory run",
        "",
        "The seq2/seq3 engineering gate passed before this run. This is exploratory; "
        "PolypGen informed method development.",
        "",
        "## Equal-sequence present-frame Dice",
        "",
        "| Sequence | Native SAM2 | Generic local ToMe | Soft mask-aware ToMe |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        values = [
            summary["per_sequence"][name][arm]["present_dice"]
            for arm in CONDITIONS
        ]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.append(
        "| **Mean across sequences** | "
        + " | ".join(
            f"**{summary['equal_sequence_mean_present_dice'][arm]:.4f}**" for arm in CONDITIONS
        )
        + " |"
    )
    lines += [
        "",
        "## Paired contrasts",
        "",
        "| Contrast | Mean Dice difference | Wins | 95% sequence bootstrap CI | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, result in summary["paired_contrasts"].items():
        low, high = result["sequence_bootstrap_95ci"]
        lines.append(
            f"| {key.replace('_', ' ')} | {result['mean_difference']:+.4f} | "
            f"{result['positive_sequence_wins']}/{result['n_sequences']} | "
            f"[{low:+.4f}, {high:+.4f}] | {result['holm_adjusted_p']:.4g} |"
        )
    lines += [
        "",
        "## False alarms and compute",
        "",
        "| Condition | False alarms / GT-empty frames | Rate | Total wall time (s) |",
        "|---|---:|---:|---:|",
    ]
    for arm in CONDITIONS:
        counts = summary["pooled_gt_empty_false_alarms"][arm]
        seconds = sum(summary["per_sequence"][name][arm]["wall_seconds"] for name in names)
        rate = "N/A" if counts["rate"] is None else f"{counts['rate']:.4f}"
        lines.append(f"| {labels[arm]} | {counts['false_alarms']} / {counts['gt_empty_targets']} | {rate} | {seconds:.1f} |")
    lines += [
        "",
        f"Predeclared meaningful-gain screen: **{'PASS' if summary['meaningful_screen_passed'] else 'FAIL'}**.",
        "",
        "This screen is not a substitute for independent-dataset confirmation. See `summary.json` "
        "for the sequence-level metrics and provenance.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume)
    try:
        torch.set_num_threads(2)
        protocol_path = ROOT / PROTOCOL
        protocol = json.loads(protocol_path.read_text())
        protocol_sha = digest(protocol_path)
        gate_completion = json.loads((args.gate_dir / "completion.json").read_text())
        gate_result = json.loads((args.gate_dir / "gate.json").read_text())
        gate_setup = json.loads((args.gate_dir / "setup.json").read_text())
        if not gate_completion.get("passed") or not gate_result.get("passed"):
            raise RuntimeError("The soft-ToMe seq2/seq3 engineering gate did not pass")
        if gate_setup.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("Gate and full cohort use different frozen protocols")
        expected_hashes = {
            "gate_script_sha256": digest(ROOT / GATE_SCRIPT),
            "method_module_sha256": digest(ROOT / METHOD_MODULE),
            "tome_merge_source_sha256": digest(ROOT / TOME_SOURCE),
        }
        if any(gate_setup.get(key) != value for key, value in expected_hashes.items()):
            raise RuntimeError("The method, gate, or pinned ToMe source changed after the gate")
        shared_source_digests = {
            source: digest(ROOT / source)
            for source in protocol["engineering_gate"]["source_files_to_hash"]
        }
        if gate_setup.get("source_file_digests") != shared_source_digests:
            raise RuntimeError("Shared data, scoring, or SAM2 integration sources changed after the gate")

        checkpoint_sha = digest(args.checkpoint)
        if checkpoint_sha != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from the frozen protocol")
        manifest_path = ROOT / protocol["scope"]["manifest"]
        manifest_sha = digest(manifest_path)
        native_path = args.native_run / "native_results.json"
        native_results = json.loads(native_path.read_text())
        if native_results.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Native cohort reference uses a different checkpoint")
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["revision"] or gate_result.get("sam2_revision") != revision:
            raise RuntimeError("SAM2 revision differs from the gate or protocol")
        author_tome()
        videos = {**load_manifest(manifest_path, "dev"), **load_manifest(manifest_path, "test")}

        provenance = {
            "protocol_sha256": protocol_sha,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "native_results_sha256": digest(native_path),
            "gate_sha256": digest(args.gate_dir / "gate.json"),
            "gate_script_sha256": expected_hashes["gate_script_sha256"],
            "method_module_sha256": expected_hashes["method_module_sha256"],
            "tome_merge_source_sha256": expected_hashes["tome_merge_source_sha256"],
            "shared_source_digests": shared_source_digests,
            "full_runner_sha256": digest(Path(__file__)),
            "sam2_revision": revision,
            "tome_revision": protocol["prior_work"]["repository_revision"],
            "device": "cpu",
            "dtype": "float32",
            "sequences": protocol["scope"]["sequences"],
            "conditions": list(CONDITIONS),
        }
        if args.resume:
            setup_path = output / "setup.json"
            if not setup_path.is_file():
                raise RuntimeError("Cannot resume without the original setup receipt")
            setup = json.loads(setup_path.read_text())
            if any(setup.get(key) != value for key, value in provenance.items()):
                raise RuntimeError("Resume provenance differs from the original run")
        else:
            write(output / "protocol.json", protocol)
            write(output / "setup.json", provenance)
        setup_sha = digest(output / "setup.json")

        sequence_names = list(protocol["scope"]["sequences"])
        sequence_results = {name: {} for name in sequence_names}
        matched_window_records = 0
        for name in sequence_names:
            if name not in videos:
                raise RuntimeError(f"Sequence absent from manifest: {name}")
            frames = videos[name]
            prompt, reference, receipt = native_reference(
                args.native_run, native_results, protocol, name, frames
            )
            expected_targets = len(reference)
            budget_by_condition = {}
            for condition in CONDITIONS:
                folder = output / condition / name
                folder.mkdir(parents=True, exist_ok=True)
                completed = load_completed(folder, expected_targets, setup_sha)
                if completed is not None:
                    rows = completed
                    run_meta = json.loads((folder / "summary.json").read_text())
                    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
                        raise RuntimeError(f"Resumed frame coverage differs: {name}/{condition}")
                    if condition == "native":
                        for row in rows:
                            expected = reference[row["frame"]]
                            if any(row[key] != expected[key] for key in (
                                "prediction_sha256", "stored_sha256", "present", "scores", "absent_fp"
                            )):
                                raise RuntimeError(f"Resumed native replay mismatch: {name}/{row['frame']}")
                    if condition != "native":
                        budget_by_condition[condition] = run_meta.get("budget_signature")
                        if not budget_by_condition[condition]:
                            raise RuntimeError("Resumed method output has no full budget audit")
                    sequence_results[name][condition] = run_meta
                    continue

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
                )
                if condition != "native":
                    budget_by_condition[condition] = budget_signature(rows)
                compact = compact_rows(rows)
                metrics = sequence_summary(compact, run_meta)
                run_meta.update({
                    **metrics,
                    "complete": True,
                    "setup_sha256": setup_sha,
                    "reference_receipt": receipt,
                    "condition": condition,
                })
                if condition != "native":
                    run_meta["budget_signature"] = budget_by_condition[condition]
                write(folder / "frames.json", compact)
                write(folder / "summary.json", run_meta)
                sequence_results[name][condition] = run_meta
                del rows
                gc.collect()

            matched_window_records += compare_budget_signatures(
                budget_by_condition["generic_local_tome"],
                budget_by_condition["soft_mask_aware_tome"],
            )
            (output / "sequence_completion").mkdir(parents=True, exist_ok=True)
            write(output / "sequence_completion" / f"{name}.json", {
                "sequence": name,
                "prompt": prompt,
                "targets": expected_targets,
                "conditions": list(CONDITIONS),
                "matched_budget_window_records": budget_by_condition["generic_local_tome"]["target_layer_window_records"],
                "complete": True,
                "setup_sha256": setup_sha,
            })
            write(output / "progress.json", {
                "sequence": name,
                "sequence_count": len([
                    candidate for candidate in sequence_names
                    if (output / "sequence_completion" / f"{candidate}.json").is_file()
                ]),
                "sequence_total": len(sequence_names),
            })

        final = make_final_summary(sequence_results, sequence_names, matched_window_records)
        write(output / "summary.json", final)
        write_results_markdown(output / "results.md", final)
        (output / "failure.json").unlink(missing_ok=True)
        write(output / "completion.json", {
            "passed": True,
            "scope": "Full PolypGen exploratory cohort",
            "sequences": sequence_names,
            "conditions": list(CONDITIONS),
            "meaningful_screen_passed": final["meaningful_screen_passed"],
        })
    except BaseException as error:
        write(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()

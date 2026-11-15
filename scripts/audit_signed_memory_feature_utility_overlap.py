#!/usr/bin/env python3
"""Verify the completed exact-pair attribution diagnostic and write its summary."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest


def tensor_sha(value):
    value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def audit(output):
    protocol = json.loads((output / "protocol.json").read_text())
    setup = json.loads((output / "setup.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    captures = json.loads((output / "capture_receipts.json").read_text())
    replays = json.loads((output / "replay_results.json").read_text())
    alignment = json.loads((output / "utility_alignment.json").read_text())

    checks = []
    if not completion.get("engineering_passed") or completion.get("efficacy_evaluated"):
        raise RuntimeError("Completion receipt does not describe an engineering-only pass")
    protocol_source = ROOT / "protocols/signed_memory_feature_utility_overlap_v1.json"
    if digest(protocol_source) != setup["protocol_sha256"]:
        raise RuntimeError("Frozen source protocol hash differs from run setup")
    if json.loads(protocol_source.read_text()) != protocol:
        raise RuntimeError("Archived protocol content differs from the frozen source")
    if len(captures) != len(protocol["pairs"]) or len(replays) != len(captures):
        raise RuntimeError("Capture/replay target coverage is incomplete")
    if {r["replay_process_pid"] for r in replays} == {setup["capture_process_pid"]}:
        raise RuntimeError("Replay did not occur in a separate process")
    checks.append("frozen protocol, source hashes, and separate replay process")

    for name, expected in setup["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Run source changed since capture: {name}")
    for pair, receipt, replay in zip(protocol["pairs"], captures, replays, strict=True):
        key = (pair["sequence"], pair["source_frame"], pair["target_frame"])
        if key != (receipt["sequence"], receipt["source_frame"], receipt["target_frame"]):
            raise RuntimeError(f"Capture pair mismatch: {key}")
        if receipt["native_prediction_sha256"] != pair["expected_native_prediction_sha256"]:
            raise RuntimeError(f"Native target output mismatch: {key}")
        if replay["native_prediction_sha256"] != receipt["native_prediction_sha256"]:
            raise RuntimeError(f"Replay target output mismatch: {key}")
        if replay["native_replay_max_abs_error"] != 0.0:
            raise RuntimeError(f"Native raw decoder replay is not exact: {key}")
        if replay["attention_layers_per_checked_forward"] != [0, 1, 2, 3]:
            raise RuntimeError(f"Four-layer memory attention not verified: {key}")
        if not (replay["parameters_unchanged"] and replay["memory_and_pointers_unchanged"]):
            raise RuntimeError(f"Frozen model or memory invariant failed: {key}")
        if receipt["memory_sources"] != pair["expected_memory_sources"] and [s["frame"] for s in receipt["memory_sources"]] != pair["expected_memory_sources"]:
            raise RuntimeError(f"Native memory source IDs changed: {key}")
        selected = [s["frame"] for s in receipt["memory_sources"]]
        if pair["source_frame"] not in selected or receipt["pointer_count"] < 0:
            raise RuntimeError(f"Selected source or pointer metadata invalid: {key}")
        capture_path = Path(receipt["capture_path"])
        if digest(capture_path) != receipt["capture_sha256"]:
            raise RuntimeError(f"Serialized capture hash mismatch: {key}")
        capture = torch.load(capture_path, map_location="cpu", weights_only=True)
        memory = capture["memory"]["memory"]
        spatial_count = memory.shape[0] - capture["memory"]["num_obj_ptr_tokens"]
        if tensor_sha(memory) != receipt["memory_input_sha256"]:
            raise RuntimeError(f"Serialized memory changed: {key}")
        if tensor_sha(memory[spatial_count:]) != receipt["pointer_content_sha256"]:
            raise RuntimeError(f"Pointer tail changed: {key}")
        archive = Path(replay["attribution_archive"])
        if digest(archive) != replay["attribution_sha256"]:
            raise RuntimeError(f"Attribution archive hash mismatch: {key}")
        with np.load(archive, allow_pickle=False) as maps:
            if not all(np.isfinite(maps[name]).all() for name in maps.files if name != "memory_sources_json"):
                raise RuntimeError(f"Nonfinite attribution map: {key}")
    checks.append("4/4 archived native outputs, memory sources, pointers, weights, gradients, and exact replay")

    finite_differences = [fd for result in replays for fd in result["finite_differences"]]
    expected_fd = 2 * 2 * len(protocol["attribution"]["finite_difference_tiles"]) * len(protocol["attribution"]["finite_difference_epsilons"])
    if len(finite_differences) != expected_fd or not all(
        np.isfinite(fd["absolute_error"]) and np.isfinite(fd["normalized_error"])
        for fd in finite_differences
    ):
        raise RuntimeError("Finite-difference coverage or values are invalid")
    fd_summary = {}
    for epsilon in protocol["attribution"]["finite_difference_epsilons"]:
        rows = [fd for fd in finite_differences if fd["epsilon"] == epsilon]
        abs_errors = [fd["absolute_error"] for fd in rows]
        norm_errors = [fd["normalized_error"] for fd in rows]
        fd_summary[str(epsilon)] = {
            "n": len(rows),
            "median_absolute_error": float(np.median(abs_errors)),
            "max_absolute_error": float(np.max(abs_errors)),
            "median_normalized_error": float(np.median(norm_errors)),
            "max_normalized_error": float(np.max(norm_errors)),
        }
    checks.append(f"{len(finite_differences)} fixed finite-difference comparisons; errors reported, not efficacy-gated")

    utility_root = Path(protocol["utility_archive"]["root"])
    csv_path = utility_root / protocol["utility_archive"]["pairs_csv"]
    if digest(csv_path) != protocol["utility_archive"]["pairs_csv_sha256"]:
        raise RuntimeError("Prior utility CSV hash mismatch")
    with csv_path.open(newline="") as handle:
        utility_rows = list(csv.DictReader(handle))
    summary = []
    for replay, aligned in zip(replays, alignment["pairs"], strict=True):
        sequence, source, target = replay["sequence"], replay["source_frame"], replay["target_frame"]
        utility_row = next(r for r in utility_rows if r["sequence"] == sequence and int(r["source_frame"]) == source and int(r["target_frame"]) == target)
        if aligned["sequence"] != sequence or aligned["source_frame"] != source or aligned["target_frame"] != target:
            raise RuntimeError("Utility alignment pair differs from replay pair")
        result_path = utility_root / "scientific_run_20261108" / "sequences" / sequence / "results.json"
        expected_hash = protocol["utility_archive"]["sequence_result_sha256"][sequence]
        if digest(result_path) != expected_hash:
            raise RuntimeError(f"Prior utility receipts changed for {sequence}")
        archived = json.loads(result_path.read_text())
        record = next(r for r in archived["main"] if r.get("anchor_target") == target and r.get("source_frame") == source)
        arms = record["arms"]
        native = arms["native"]
        if native["prediction_sha256"] != replay["native_prediction_sha256"]:
            raise RuntimeError("Attribution replay and utility-native target hashes differ")
        tile_utilities = []
        for tile in range(16):
            arm = arms[f"tile_{tile:02d}"]
            selected_sources = [layer.get("source_frame") for layer in arm["read"]["selection_layers"]]
            if arm["source_frame_index"] != target or arm["read"]["frame"] != target:
                raise RuntimeError("A saved tile intervention used another target frame")
            if selected_sources != [source] * 4:
                raise RuntimeError("A saved tile intervention used another source frame")
            tile_utilities.append(float(native["scores"]["dice"]) - float(arm["scores"]["dice"]))
        if not np.allclose(tile_utilities, aligned["tile_removal_utility"], rtol=0, atol=1e-12):
            raise RuntimeError("Saved tile utility values differ from their archived receipts")
        source_scores = next(r for r in replay["per_source_signed_tile_scores"] if r["frame"] == source)
        correlations = {}
        for signal in ("native_mask_support", "object_presence"):
            x = [source_scores[signal]["tiles"][f"tile_{tile:02d}"]["signed_input_x_gradient_sum"] for tile in range(16)]
            rho = spearmanr(x, tile_utilities).statistic
            rho = float(rho) if np.isfinite(rho) else None
            saved = aligned[signal]["signed_spearman_rho"]
            if rho is None or saved is None:
                if rho is not saved:
                    raise RuntimeError("Saved descriptive correlation differs")
            elif not np.isclose(rho, saved, rtol=0, atol=1e-12):
                raise RuntimeError("Saved descriptive correlation differs")
            correlations[signal] = rho
        summary.append({
            "sequence": sequence,
            "source_frame": source,
            "target_frame": target,
            "source_dice": float(utility_row["source_dice"]),
            "target_native_dice": float(utility_row["target_native_dice"]),
            "signed_rho_native_mask_support": correlations["native_mask_support"],
            "signed_rho_object_presence": correlations["object_presence"],
            "useful_tiles": int(sum(value > 0 for value in tile_utilities)),
            "harmful_tiles": int(sum(value < 0 for value in tile_utilities)),
            "zero_effect_tiles": int(sum(value == 0 for value in tile_utilities)),
        })
    checks.append("Exact-source 16-tile utility and descriptive correlations independently recomputed from pinned receipts")
    if setup["source_hashes"] != {name: digest(ROOT / name) for name in setup["source_hashes"]}:
        raise RuntimeError("Frozen code changed after run completion")

    audit_result = {
        "passed": True,
        "scope": "Engineering-valid, exploratory exact-pair attribution-to-utility diagnostic; no full-video or efficacy result",
        "checks": checks,
        "pairs": summary,
        "finite_difference_summary": fd_summary,
        "limits": [
            "Four post-hoc selected source/target pairs; correlations across 16 tiles per pair are descriptive and not independent evidence.",
            "Input-times-gradient measures local sensitivity to feature scaling; saved utility measures actual tile removal and Dice.",
            "No inference-time token selector, ToMe modification, or full-video Dice evaluation was tested.",
        ],
    }
    write(output / "audit.json", json.dumps(audit_result, indent=2) + "\n")
    lines = [
        "# Signed memory-feature attribution vs. saved tile utility",
        "",
        "Engineering-valid exploratory diagnostic; four post-hoc exact source/target pairs. No segmentation method or full-video efficacy run was tested.",
        "",
        "| Source → target | Source Dice | Target native Dice | Signed ρ: mask-logit attribution | Signed ρ: object-presence attribution | Tiles helpful / harmful / zero |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['sequence']} {row['source_frame']} → {row['target_frame']} | {row['source_dice']:.6f} | {row['target_native_dice']:.6f} | {row['signed_rho_native_mask_support']:+.3f} | {row['signed_rho_object_presence']:+.3f} | {row['useful_tiles']} / {row['harmful_tiles']} / {row['zero_effect_tiles']} |"
        )
    lines += [
        "",
        "Utility is native Dice minus tile-removed Dice: positive means removing that tile harmed the saved target prediction. Correlations are Spearman ranks across 16 tiles within each pair.",
        "",
        "**Result:** no consistent positive relationship. The low-quality seq17 source (Dice 0.003777) has mixed tile effects, but signed attribution correlates weakly negatively with utility for both tested decoder scalars. This does not refute local mixed utility; it shows these confidence/logit gradients are not a reliable proxy for the saved Dice-removal effects in these four examples.",
        "",
        "Native target hashes matched 4/4 archived outputs; independent raw decoder replay had zero max absolute error; all four memory-attention layers executed; weights, memory and pointer-tail checks passed. Finite-difference checks are descriptive, with error statistics in `audit.json`.",
        "",
        "Feature scaling is not token deletion: deletion also changes attention normalization. The sample was selected post hoc, uses only four pairs, and gives no generalization or full-video Dice claim.",
        "",
        f"Artifacts: `{output}`",
    ]
    write(output / "results.md", "\n".join(lines) + "\n")
    print(json.dumps({"passed": True, "pairs": len(summary), "fd": fd_summary, "results": str(output / 'results.md')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output_dir.resolve())


if __name__ == "__main__":
    main()

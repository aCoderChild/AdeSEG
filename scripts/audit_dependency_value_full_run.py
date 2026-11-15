"""Offline integrity audit for the frozen dependency-aware V full-video run."""

import hashlib
import json
import math
from pathlib import Path

from scripts.run_native_value_consolidation_full import SEQUENCES, compare, summarize


ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/tome_region_identification_v1/"
    "post_value_source_review_20261112/dependency_value_intervention_v1"
)
PROTOCOL = DRIVE / "full_protocol.json"
RUN = DRIVE / "full_attempt01"
ARMS = ("native", "value", "joint")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(left, right, *, atol=1e-12):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def compare_tree(actual, expected, path="comparison"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                f"Field mismatch at {path}")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                f"Length mismatch at {path}")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_tree(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        require(close(actual, expected), f"Numeric mismatch at {path}: {actual} != {expected}")
    else:
        require(actual == expected, f"Value mismatch at {path}: {actual} != {expected}")


def main():
    protocol = json.loads(PROTOCOL.read_text())
    expected = tuple(f"seq{i}" for i in range(2, 24) if i != 7)
    require(SEQUENCES == expected, "Auditor sequence definition differs from protocol cohort")
    require(tuple(protocol["sequences"]) == expected, "Protocol does not specify the 21-sequence cohort")
    require(protocol["excluded"] == ["seq1", "seq7"], "Unexpected excluded sequences")

    completion = json.loads((RUN / "completion.json").read_text())
    require(completion.get("complete") is True, "Full run has no successful completion receipt")
    require(tuple(completion.get("sequences", ())) == expected, "Completion cohort differs")
    result_path = RUN / "results.json"
    require(sha256(result_path) == completion.get("results_sha256"), "Results digest mismatch")
    pin_receipt = json.loads((RUN / "external_pins_final.json").read_text())
    require(pin_receipt.get("passed") is True and not pin_receipt.get("mismatches"),
            "Final external input pin audit failed")

    for name, expected_hash in protocol.get("source_hashes", {}).items():
        require(sha256(ROOT / name) == expected_hash, f"Repository source changed: {name}")
    for name, expected_hash in protocol.get("input_hashes", {}).items():
        require(sha256(name) == expected_hash, f"Experiment input changed: {name}")
    for name, expected_hash in protocol.get("external_source_hashes", {}).items():
        require(sha256(name) == expected_hash, f"Pinned external source changed: {name}")

    result = json.loads(result_path.read_text())
    rows = result["frame_rows"]
    saved_summaries = result["sequence_summaries"]
    runtime = result["runtime"]
    require(set(rows) == set(ARMS) and set(saved_summaries) == set(ARMS), "Arm set mismatch")
    require(all(set(rows[arm]) == set(expected) for arm in ARMS), "A sequence is missing from an arm")

    recomputed_summaries = {arm: {} for arm in ARMS}
    sequence_audit = {}
    for sequence in expected:
        native = rows["native"][sequence]
        expected_frames = [row["frame"] for row in native]
        require(expected_frames == sorted(set(expected_frames)), f"Native frame order invalid: {sequence}")
        present_frames = [row["frame"] for row in native if row["present"]]
        require(present_frames, f"No present GT frames in {sequence}")
        for arm in ARMS:
            current = rows[arm][sequence]
            require([row["frame"] for row in current] == expected_frames,
                    f"Target frame set differs: {arm}/{sequence}")
            require([row["present"] for row in current] == [row["present"] for row in native],
                    f"GT presence differs: {arm}/{sequence}")
            dice = [row["scores"]["dice"] for row in current if row["present"]]
            require(len(dice) == len(present_frames), f"Present-frame denominator differs: {arm}/{sequence}")
            recomputed = summarize(current)
            recomputed_summaries[arm][sequence] = recomputed
            saved = saved_summaries[arm][sequence]
            require(close(saved["present_dice"], sum(dice) / len(dice)),
                    f"Saved present Dice does not recompute: {arm}/{sequence}")
            require(close(saved["boundary_f"], recomputed["boundary_f"]),
                    f"Saved Boundary F does not recompute: {arm}/{sequence}")
            if arm != "native":
                rt = runtime[arm][sequence]
                require(rt["parameter_hash_unchanged"] and rt["temporal_slots_restored"],
                        f"Parameter or temporal invariant failed: {arm}/{sequence}")
                require(rt["attention_layer_calls"] == 4 * len(current),
                        f"Attention layer read count differs: {arm}/{sequence}")
                for row in current:
                    reads = row.get("attention_reads", [])
                    require([read["layer"] for read in reads] == [0, 1, 2, 3],
                            f"Missing native attention layer: {arm}/{sequence}/{row['frame']}")
        value = rows["value"][sequence]
        joint = rows["joint"][sequence]
        for vr, jr in zip(value, joint, strict=True):
            for key in ("pointer_keys", "frames", "spatial_keys_before", "spatial_keys_after"):
                require(all(vr["attention_reads"][layer][key] == jr["attention_reads"][layer][key]
                            for layer in range(4)),
                        f"Matched V-control schedule differs: {sequence}/{vr['frame']}/{key}")
        sequence_audit[sequence] = {
            "present_frames": len(present_frames),
            "empty_frames": sum(not row["present"] for row in native),
            "dice": {arm: recomputed_summaries[arm][sequence]["present_dice"] for arm in ARMS},
        }

    recomputed_comparison = compare(recomputed_summaries)
    compare_tree(result["comparison"], recomputed_comparison)
    report = {
        "status": "passed",
        "cohort": list(expected),
        "results_sha256": sha256(result_path),
        "final_external_pins_passed": True,
        "all_arm_frame_sets_match": True,
        "all_gt_presence_sets_match": True,
        "sequence_summary_dice_recomputed": True,
        "runtime_and_attention_receipts_valid": True,
        "matched_control_schedules_valid": True,
        "comparison_statistics_recomputed": True,
        "cohort_comparison": recomputed_comparison,
        "sequences": sequence_audit,
    }
    (RUN / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Independent full-run audit",
        "",
        "Status: **PASSED**",
        "",
        f"Results SHA-256: `{report['results_sha256']}`",
        "",
        "| Arm | Mean present Dice | Mean Boundary F | False alarms / GT-empty frames | Undetected returns |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in recomputed_comparison["cohort"].items():
        lines.append(
            f"| {arm} | {row['present_dice']:.6f} | {row['boundary_f']:.6f} | "
            f"{row['gt_empty_false_alarm_count']} / {row['gt_empty_targets']} | {row['undetected_returns']} |"
        )
    lines += [
        "",
        f"Frozen continuation screen: **{'PASS' if recomputed_comparison['screen']['passed'] else 'FAIL'}**.",
        "",
        "This audit validates saved full-run arithmetic, frame coverage, source pins, and recorded invariants; it does not establish untouched-data generalization.",
    ]
    (RUN / "independent_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

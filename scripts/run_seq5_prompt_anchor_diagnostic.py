"""Run the frozen four-arm Seq5 prompt-anchor ToMe diagnostic after its gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_pointer_conditioned_tome_full as full
import scripts.run_region_constrained_tome_gate as gate
from modeling.pointer_conditioned_tome import NativePointerConditionedToMe
from modeling.prompt_anchor_tome import NativePromptAnchorToMe
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from scripts.memory_token_io import digest, validate_output
from scripts.run_regional_refinement import write


PROTOCOL = ROOT / "protocols/seq5_prompt_anchor_diagnostic_v1.json"
DRIVE_ROOT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment"
)
ACTIVE = DRIVE_ROOT / "pointer_conditioned_tome_polypgen_v1/full_attempt01"
NATIVE_RUN = DRIVE_ROOT / "143_official_native_feature_audit_v1/full"
CHECKPOINT = DRIVE_ROOT / "assets/sam2.1_hiera_tiny.pt"
GATE_DIR = ACTIVE / "prompt_anchor_seq5_diagnostic/gate_v2"
COHORT_AUDIT = ACTIVE / "audits/final_full_cohort_integrity_audit.json"


class _DiagnosticManagerFactory:
    treatment = "prompt_anchor"

    def __new__(cls, model, *, mode, merge_plan_by_target=None, **kwargs):
        if mode == "regional":
            manager = (NativePromptAnchorToMe if cls.treatment == "prompt_anchor"
                       else NativePointerConditionedToMe)
            return manager(model, merge_plan_by_target=merge_plan_by_target, **kwargs)
        return NativeRegionConstrainedToMe(
            model, mode=mode, merge_plan_by_target=merge_plan_by_target, **kwargs
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("execution", {}).get("launch_authorized") is not True:
        raise RuntimeError("Protocol does not authorize launch; complete prerequisites and freeze an amendment")
    if digest(PROTOCOL) != PROTOCOL.with_suffix(".json.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol hash mismatch")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, expected in protocol["input_hashes"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    completion = ACTIVE / "completion.json"
    if not completion.is_file():
        raise RuntimeError("Active full cohort is not complete")
    cohort = json.loads(completion.read_text())
    expected_sequences = [f"seq{i}" for i in range(2, 24) if i != 7]
    if not cohort.get("passed") or cohort.get("sequences") != expected_sequences:
        raise RuntimeError("Active full cohort completion receipt is incomplete or failed")
    if not COHORT_AUDIT.is_file():
        raise RuntimeError("Independent full-cohort integrity audit receipt is missing")
    audit = json.loads(COHORT_AUDIT.read_text())
    if audit.get("passed") is not True or audit.get("sequences") != expected_sequences:
        raise RuntimeError("Independent full-cohort integrity audit did not pass")
    gate_receipt = GATE_DIR / "completion.json"
    if not gate_receipt.is_file():
        raise RuntimeError("Prompt-anchor engineering gate is not complete")
    gate_result = json.loads(gate_receipt.read_text())
    independent_replay = GATE_DIR / "independent_replay.json"
    if (not gate_result.get("engineering_passed")
            or not gate_result.get("native_noop_exact")
            or not gate_result.get("matched_generic_budget")
            or not independent_replay.is_file()
            or json.loads(independent_replay.read_text()).get("byte_identical") is not True):
        raise RuntimeError("Prompt-anchor gate and independent replay must both pass")

    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        setup = {
            "protocol_sha256": digest(PROTOCOL),
            "runner_sha256": digest(Path(__file__)),
            "prompt_anchor_sha256": digest(ROOT / "modeling/prompt_anchor_tome.py"),
            "pointer_tome_sha256": digest(ROOT / "modeling/pointer_conditioned_tome.py"),
            "active_cohort_completion_sha256": digest(completion),
            "full_cohort_integrity_audit_sha256": digest(COHORT_AUDIT),
            "engineering_gate_completion_sha256": digest(gate_receipt),
            "engineering_gate_independent_replay_sha256": digest(independent_replay),
            "source_hashes": protocol["source_hashes"],
            "input_hashes": protocol["input_hashes"],
            "device": "cpu",
            "dtype": "float32",
        }
        write(output / "setup.json", setup)
        gate.NativeRegionConstrainedToMe = _DiagnosticManagerFactory
        full.PROTOCOL = "protocols/pointer_conditioned_tome_efficacy_v1.json"
        repo, revision = full.activate_author_repo()
        gate_protocol = json.loads((ROOT / "protocols/prompt_anchor_tome_gate_v2.json").read_text())
        if revision != gate_protocol["sam2_revision"]:
            raise RuntimeError("SAM2 checkout differs from the engineering-gate revision")
        import subprocess
        import torch
        tome_repo = ROOT / "external/ToMe"
        tome_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tome_repo, text=True
        ).strip()
        if tome_revision != gate_protocol["tome_revision"]:
            raise RuntimeError("ToMe checkout differs from the engineering-gate revision")
        torch.set_num_threads(2)
        torch.manual_seed(protocol["execution"]["seed"])
        torch.use_deterministic_algorithms(True)
        native_protocol = json.loads((ROOT / full.PROTOCOL).read_text())
        native_results = json.loads((NATIVE_RUN / "native_results.json").read_text())
        if native_results.get("checkpoint_sha256") != digest(CHECKPOINT):
            raise RuntimeError("Native reference checkpoint differs from the frozen candidate")
        videos = {**full.load_manifest(ROOT / "data/polypgen_sequence.jsonl", "dev"),
                  **full.load_manifest(ROOT / "data/polypgen_sequence.jsonl", "test")}
        sequence = protocol["scope"]["sequence"]
        if sequence not in videos:
            raise RuntimeError("Seq5 is missing from the frozen manifest")
        frames = videos[sequence]
        prompt, reference, native_receipt = gate.native_reference(
            NATIVE_RUN, native_results, native_protocol, sequence, frames
        )
        if prompt != 0 or len(reference) != 249:
            raise RuntimeError("Seq5 native reference differs from frozen prompt/target coverage")
        results = {}
        merge_plan = None
        for arm in ("native", "prompt_anchor", "same_frame_pointer", "generic"):
            folder = output / arm
            folder.mkdir()
            if arm == "prompt_anchor":
                _DiagnosticManagerFactory.treatment = "prompt_anchor"
                condition = "region_constrained_tome"
                plan = None
            elif arm == "same_frame_pointer":
                _DiagnosticManagerFactory.treatment = "same_frame_pointer"
                condition = "region_constrained_tome"
                plan = merge_plan
            elif arm == "generic":
                condition = "generic_local_tome"
                plan = merge_plan
            else:
                condition = "native"
                plan = None
            rows, metadata = gate.run_condition(
                sequence, condition, frames, prompt, reference, CHECKPOINT,
                native_protocol["model"]["configuration"],
                repo, folder,
                merge_plan_by_target=plan,
            )
            if [row["frame"] for row in rows] != sorted(reference):
                raise RuntimeError(f"Incomplete or misordered frame rows in {arm}")
            if arm == "native":
                for row in rows:
                    ref = reference[row["frame"]]
                    if any(row[key] != ref[key] for key in (
                        "prediction_sha256", "stored_sha256", "present", "scores", "absent_fp"
                    )):
                        raise RuntimeError(f"Native replay mismatch at Seq5 frame {row['frame']}")
            if arm == "prompt_anchor":
                for row in rows:
                    calls = row.get("attention_reads", [])
                    if [call.get("layer") for call in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Prompt-anchor arm missed a layer at frame {row['frame']}")
                    refs = {(call.get("pointer_reference_frame"),
                             call.get("pointer_reference_sha256")) for call in calls}
                    if len(refs) != 1 or None in next(iter(refs)) or next(iter(refs))[0] != prompt:
                        raise RuntimeError(f"Prompt anchor identity changed at frame {row['frame']}")
                merge_plan = full.matched_plan_from_regional(rows)
                write(output / "matched_window_plan.json", {
                    str(target): {str(frame): {str(window): count
                        for window, count in windows.items()}
                        for frame, windows in memory.items()}
                    for target, memory in merge_plan.items()
                })
            if arm in ("same_frame_pointer", "generic"):
                full.validate_generic_against_plan(rows, merge_plan)
            compact = full.compact_rows(rows)
            summary = {
                **full.score_summary(compact),
                "complete": True,
                "sequence": sequence,
                "arm": arm,
                "runtime": metadata.get("wall_seconds"),
                "native_reference": native_receipt,
            }
            write(folder / "frames.json", compact)
            write(folder / "summary.json", summary)
            results[arm] = summary
            write(output / "progress.json", {"completed_arms": list(results), "current_arm": arm})
        write(output / "results.json", results)
        for name, expected in protocol["source_hashes"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Source changed during diagnostic: {name}")
        write(output / "completion.json", {
            "complete": True,
            "scope": "Seq5 development diagnostic only",
            "arms": list(results),
            "results_sha256": digest(output / "results.json"),
            "claim_limit": protocol["report"]["interpretation_limit"],
        })
    except Exception as exc:
        write(output / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

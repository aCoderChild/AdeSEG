"""Real-video engineering gate for the ten-map ToMe allocation study."""

import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    SAM2_REFERENCE,
    SEQUENCES as GATE_SEQUENCES,
    _assert_read_integrity,
    activate_reference_repo,
    run_condition,
    verify_native_setup,
)


PROTOCOL = ROOT / "protocols/ten_map_allocation_tome_polypgen_v1.json"
SEQUENCES = GATE_SEQUENCES
ARMS = (
    ("uniform_generic", "global", "uniform"),
    ("uniform_regional", "regional", "uniform"),
    ("recent_dense_generic", "global", "recent_dense"),
    ("recent_dense_regional", "regional", "recent_dense"),
)
MAX_NONCOND_MAPS = 10
SOURCES = (
    "protocols/ten_map_allocation_tome_polypgen_v1.json",
    "modeling/wider_history_tome.py",
    "modeling/region_constrained_tome.py",
    "modeling/prediction_conditioned_affinity.py",
    "modeling/soft_role_affinity.py",
    "scripts/run_ten_map_allocation_tome_gate.py",
    "scripts/run_wider_history_tome_gate.py",
    "scripts/run_region_constrained_tome_gate.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_soft_role_affinity_cv.py",
    "scripts/run_regional_refinement.py",
    "scripts/bank_integrity.py",
    "scripts/memory_token_io.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
)


def _span_ids(read):
    return [(row["frame"], row["t_pos"]) for row in read["frames"]]


def _assert_matched_design(rows_by_arm, allocation):
    generic = rows_by_arm[f"{allocation}_generic"]
    regional = rows_by_arm[f"{allocation}_regional"]
    if [row["frame"] for row in generic] != [row["frame"] for row in regional]:
        raise RuntimeError(f"Matched frame coverage differs for {allocation}")
    for g_row, r_row in zip(generic, regional, strict=True):
        if len(g_row["attention_reads"]) != len(r_row["attention_reads"]):
            raise RuntimeError("Generic/regional attention-read count differs")
        for g_read, r_read in zip(
            g_row["attention_reads"], r_row["attention_reads"], strict=True
        ):
            _assert_read_integrity(g_read, expected_max_noncond=MAX_NONCOND_MAPS)
            _assert_read_integrity(r_read, expected_max_noncond=MAX_NONCOND_MAPS)
            if _span_ids(g_read) != _span_ids(r_read):
                raise RuntimeError("Generic/regional arms read different memory frame IDs")
            if g_read["frames"] != r_read["frames"]:
                raise RuntimeError("Generic/regional arms have different local quotas")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    protocol = json.loads(PROTOCOL.read_text())
    if "engineering gate required" not in protocol["status"]:
        raise RuntimeError("Protocol status does not require an engineering gate")
    if protocol["fixed_design"]["history"].find("ten") < 0:
        raise RuntimeError("Protocol does not specify the ten-map fixed history")
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    if digest(args.checkpoint) != "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69":
        raise RuntimeError("Checkpoint digest differs from the verified native checkpoint")
    if digest(manifest) != "80d83e6a4099fc727e90bf06757ae81b3292d24dc5651daf1457f4a54e248c7f":
        raise RuntimeError("PolypGen manifest differs from the archived native run")
    activate_reference_repo()
    verify_native_setup(args.checkpoint, manifest)
    from modeling.region_constrained_tome import author_tome

    tome = author_tome()
    if tome.__name__ != "_aadeseg_author_tome_merge":
        raise RuntimeError("Did not load the pinned upstream ToMe matcher")

    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    native_refs, prompts, receipts = {}, {}, {}
    for name in SEQUENCES:
        prompt, rows, receipt = native_reference(
            NATIVE_RESULTS.parent, native_results, native_protocol, name, videos[name]
        )
        native_refs[name], prompts[name], receipts[name] = rows, prompt, receipt

    source_hashes = {source: digest(ROOT / source) for source in SOURCES}
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "native_setup_sha256": digest(NATIVE_SETUP),
        "native_results_sha256": digest(NATIVE_RESULTS),
        "source_hashes": source_hashes,
        "sequences": list(SEQUENCES),
        "arms": [name for name, _, _ in ARMS],
        "max_nonconditioning_maps": MAX_NONCOND_MAPS,
        "device": "cpu",
        "native_receipts": receipts,
        "status": "ten-map engineering gate in progress; no efficacy conclusion",
    }
    write(args.output_dir / "setup.json", setup)
    results, summaries = {name: {} for name, _, _ in ARMS}, {}
    for sequence in SEQUENCES:
        arm_rows = {}
        for name, matching, allocation in ARMS:
            rows, summary = run_condition(
                matching,
                sequence,
                videos[sequence],
                prompts[sequence],
                native_refs[sequence],
                args.checkpoint,
                args.output_dir,
                max_noncond_maps=MAX_NONCOND_MAPS,
                allocation=allocation,
                condition_name=name,
            )
            results[name][sequence] = rows
            arm_rows[name] = rows
            summaries[f"{name}/{sequence}"] = summary
            write(args.output_dir / "summaries.json", summaries)
        _assert_matched_design(arm_rows, "uniform")
        _assert_matched_design(arm_rows, "recent_dense")
        reference = arm_rows["uniform_generic"]
        for name in ("recent_dense_generic", "uniform_regional", "recent_dense_regional"):
            candidate = arm_rows[name]
            if [row["frame"] for row in candidate] != [row["frame"] for row in reference]:
                raise RuntimeError(f"Frame coverage differs across arms for {sequence}")
            for ref_row, candidate_row in zip(reference, candidate, strict=True):
                for ref_read, candidate_read in zip(
                    ref_row["attention_reads"], candidate_row["attention_reads"], strict=True
                ):
                    if _span_ids(ref_read) != _span_ids(candidate_read):
                        raise RuntimeError(f"Memory source frame IDs differ at {sequence}")

    write(args.output_dir / "results.json", {
        "classification": "engineering gate only; not a Dice efficacy result",
        "arms": results,
        "summaries": summaries,
    })
    write(args.output_dir / "completion.json", {
        "passed": True,
        "classification": "real-video engineering gate; no efficacy claim",
        "sequences": list(SEQUENCES),
        "arms": [name for name, _, _ in ARMS],
        "native_exact_replay_through_six_history_maps": True,
        "same_memory_source_frames_across_arms": True,
        "matched_generic_regional_quotas": True,
        "fixed_spatial_budget_and_pointer_preservation": True,
        "sam2_frozen_and_temporal_slots_restored": True,
        "full_cohort_authorized_next": True,
        "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    })
    write(args.output_dir / "progress.json", {"status": "complete", "passed": True})
    print(f"PASS: ten-map engineering gate saved to {args.output_dir}")


if __name__ == "__main__":
    main()

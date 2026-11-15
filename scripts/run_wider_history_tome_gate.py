"""Drive-only real-video engineering gate for fixed-budget wider-history ToMe."""

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.wider_history_tome import FixedBudgetLongHistoryToMe
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_region_constrained_tome_gate import load_shape, native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame


PROTOCOL = ROOT / "protocols/wider_history_tome_polypgen_v1.json"
NATIVE_PROTOCOL = ROOT / "protocols/region_constrained_tome_polypgen_v1.json"
NATIVE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
NATIVE_SETUP = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/143_official_native_feature_audit_v1/full/setup.json"
)
NATIVE_RESULTS = NATIVE_SETUP.parent / "native_results.json"
CHECKPOINT_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/assets/sam2.1_hiera_tiny.pt"
)
SAM2_REFERENCE = ROOT / "external/sam2_diagnostic_reference"
CONDITIONS = ("global", "regional")
SEQUENCES = ("seq2", "seq3")
SOURCES = (
    "protocols/wider_history_tome_polypgen_v1.json",
    "modeling/wider_history_tome.py",
    "modeling/region_constrained_tome.py",
    "modeling/prediction_conditioned_affinity.py",
    "modeling/soft_role_affinity.py",
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
    "evaluation/davis2017/metrics.py",
)


def activate_reference_repo():
    import subprocess

    revision = subprocess.check_output(
        ["git", "-C", str(SAM2_REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != NATIVE_REVISION:
        raise RuntimeError(f"SAM2 reference revision mismatch: {revision}")
    subprocess.run(
        ["git", "-C", str(SAM2_REFERENCE), "diff", "--exit-code", "HEAD"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(SAM2_REFERENCE), "diff", "--cached", "--exit-code"],
        check=True,
        capture_output=True,
    )
    path = str(SAM2_REFERENCE)
    if path not in sys.path:
        sys.path.insert(0, path)
    return revision


def verify_native_setup(checkpoint, manifest):
    setup = json.loads(NATIVE_SETUP.read_text())
    if setup.get("checkpoint_sha256") != digest(checkpoint):
        raise RuntimeError("Gate checkpoint differs from the archived native checkpoint")
    if setup.get("manifest_sha256") != digest(manifest):
        raise RuntimeError("Gate manifest differs from the archived native manifest")
    if setup.get("model_config") != "configs/sam2.1/sam2.1_hiera_t.yaml":
        raise RuntimeError("Archived native model configuration differs")
    for source, expected in setup.get("sources", {}).items():
        if not source.startswith("external/sam2/"):
            continue
        relative = source.removeprefix("external/sam2/")
        actual = digest(SAM2_REFERENCE / relative)
        if actual != expected:
            raise RuntimeError(f"SAM2 reference source differs from native setup: {relative}")
    return setup


def _compact_read(call):
    frame_rows = []
    for row in call["frames"]:
        windows = row.get("windows", [])
        retained = [int(window["retained_tokens"]) for window in windows]
        if retained and sum(retained) != int(row["target_tokens"]):
            raise RuntimeError("Local retained-token quotas do not sum to the frame quota")
        canonical = json.dumps(
            [(index, value) for index, value in enumerate(retained)],
            separators=(",", ":"),
        ).encode()
        frame_rows.append({
            "frame": int(row["frame"]),
            "t_pos": int(row["t_pos"]),
            "source_tokens": int(row["source_tokens"]),
            "target_tokens": int(row["target_tokens"]),
            "merged_tokens": int(row["merged_tokens"]),
            "windows": len(retained),
            "window_quota_sha256": hashlib.sha256(canonical).hexdigest(),
        })
    return {
        "layer": int(call["layer"]),
        "spatial_keys_before": int(call["spatial_keys_before"]),
        "spatial_keys_after": int(call["spatial_keys_after"]),
        "pointer_keys": int(call["pointer_keys"]),
        "spatial_key_mass_sum": float(call["spatial_key_mass_sum"]),
        "key_mass_sum": float(call["key_mass_sum"]),
        "frames": frame_rows,
    }


def _assert_read_integrity(read, expected_max_noncond=12):
    if read["spatial_key_mass_sum"] != float(read["spatial_keys_before"]):
        raise RuntimeError("Spatial cluster mass is not conserved")
    if read["key_mass_sum"] != float(read["spatial_keys_before"] + read["pointer_keys"]):
        raise RuntimeError("Total attention mass including pointers is not conserved")
    frames = read["frames"]
    cond = [row for row in frames if row["t_pos"] == 0]
    noncond = [row for row in frames if row["t_pos"] > 0]
    if len(cond) != 1 or cond[0]["target_tokens"] != 4096:
        raise RuntimeError("The single conditioning map must remain dense")
    if len(noncond) > expected_max_noncond:
        raise RuntimeError(
            f"The read exceeded the declared {expected_max_noncond}-frame history"
        )
    expected_tokens = 4096 + 4096 * len(noncond)
    if read["spatial_keys_before"] != expected_tokens:
        raise RuntimeError("Uncompressed spatial key count differs from selected memory spans")
    expected_compressed = (
        expected_tokens if len(noncond) <= 6 else 4096 + 24576
    )
    if sum(row["target_tokens"] for row in frames) != expected_compressed:
        raise RuntimeError("Per-frame quotas violate the fixed native spatial budget")
    if read["spatial_keys_after"] != expected_compressed:
        raise RuntimeError("SDPA spatial token count differs from the planned budget")


def assert_replay_row(row, archived, sequence):
    fields = ("frame","prediction_sha256","stored_sha256","present","scores","absent_fp","object_score")
    if archived is None or any(row.get(key) != archived.get(key) for key in fields):
        raise RuntimeError(f"Archived treatment replay differs: {sequence}/{row['frame']}")


@torch.inference_mode()
def run_condition(
    condition, sequence, frames, prompt, native_rows, checkpoint, output_dir,
    *, max_noncond_maps=12, allocation="uniform", condition_name=None,
    end_frame=None, replay_reference=None, save_replay_bank=False,
    memory_reader_factory=None, native_exact_required=None,
):
    final_frame = len(frames)-1 if end_frame is None else end_frame
    if type(final_frame) is not int or not prompt < final_frame < len(frames):
        raise ValueError("Replay endpoint must follow the prompt within the original video")
    if save_replay_bank and replay_reference is None:
        raise ValueError("Bank capture requires an archived replay reference")
    revision = activate_reference_repo()
    args = SimpleNamespace(
        method="sam2",
        device="cpu",
        checkpoint=str(checkpoint),
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    model = build_author_predictor(args, SAM2_REFERENCE)
    if model.num_maskmem != 7 or len(model.memory_attention.layers) != 4:
        raise RuntimeError("SAM2 native memory configuration differs from the gate protocol")
    if model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
        raise RuntimeError("SAM2 evaluation behavior differs from the native reference")
    initial_parameter_hash = parameter_digest(model)
    initial_tpos = model.maskmem_tpos_enc.detach().clone()
    initial_num_maskmem = model.num_maskmem
    result_rows, calls, elapsed = [], [], 0.0
    native_exact_checks = 0
    condition_name = condition_name or condition
    sequence_dir = output_dir / condition_name / sequence
    sequence_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix=f"wider_{sequence}_", dir=output_dir) as temp:
        state, observed_prompt = prepare(model, frames, Path(temp))
        if observed_prompt != prompt:
            raise RuntimeError("Prompt differs from the archived native reference")
        iterator = None
        started = time.monotonic()
        manager = (memory_reader_factory(model) if memory_reader_factory is not None
                   else FixedBudgetLongHistoryToMe(
                       model,mode=condition,window_size=4,
                       max_noncond_maps=max_noncond_maps,allocation=allocation))
        try:
            with manager as observer:
                expanded_tpos = model.maskmem_tpos_enc.detach()
                expected_rows = max_noncond_maps + 1
                if expanded_tpos.shape[0] != expected_rows or model.num_maskmem != expected_rows:
                    raise RuntimeError(
                        f"Extended {max_noncond_maps}-frame time-slot table was not installed"
                    )
                if not torch.equal(expanded_tpos[:6], initial_tpos[:6]):
                    raise RuntimeError("Native time slots for ages one through six changed")
                if not torch.equal(
                    expanded_tpos[6:max_noncond_maps],
                    initial_tpos[5:6].expand(max_noncond_maps - 6, *initial_tpos.shape[1:]),
                ):
                    raise RuntimeError("Older frames do not reuse the native oldest time slot")
                if not torch.equal(
                    expanded_tpos[max_noncond_maps:max_noncond_maps + 1], initial_tpos[6:7]
                ):
                    raise RuntimeError("Conditioning-frame time slot changed")

                iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                expected_frame = prompt
                while expected_frame <= final_frame:
                    frame, object_ids, logits = next(iterator)
                    if frame != expected_frame or object_ids != [1]:
                        raise RuntimeError(f"Unexpected frame/object order: {sequence}/{frame}")
                    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
                        raise RuntimeError(f"Unexpected logits shape at {sequence}/{frame}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite logits at {sequence}/{frame}")
                    expected_frame += 1
                    if frame == prompt:
                        continue

                    stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"].get(frame)
                    if stored is None:
                        raise RuntimeError(f"Missing stored output at {sequence}/{frame}")
                    if tuple(logits.shape) != (1, 1, *load_shape(frames[frame].mask_path)):
                        raise RuntimeError(f"Propagated mask does not match video resolution at {sequence}/{frame}")
                    if stored["pred_masks"].ndim != 4 or stored["pred_masks"].shape[:2] != (1, 1):
                        raise RuntimeError(f"Stored memory mask has an unexpected shape at {sequence}/{frame}")
                    scored = score_frame(logits, frames, frame)
                    row = {
                        **scored,
                        "stored_sha256": fingerprint(stored)["sha256"],
                        "object_score": float(stored["object_score_logits"].detach().cpu().item()),
                        "condition_name": condition_name,
                        "allocation": allocation,
                    }
                    if replay_reference is not None:
                        assert_replay_row(row, replay_reference.get(frame), sequence)
                    noncond_count = sum(int(span["t_pos"] > 0) for span in observer.spans)
                    check_native = (noncond_count <= 6 if native_exact_required is None
                                    else native_exact_required)
                    if check_native:
                        native = native_rows.get(frame)
                        if native is None or any(
                            row[key] != native[key]
                            for key in ("prediction_sha256", "stored_sha256", "present", "scores", "absent_fp")
                        ):
                            raise RuntimeError(
                                f"Native exact replay failed before expanded history is used: {sequence}/{frame}"
                            )
                    if check_native:
                        native_exact_checks += 1
                    frame_calls = [call for call in observer.calls if call["frame"] == frame]
                    if [call["layer"] for call in frame_calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Expected one read per layer at {sequence}/{frame}")
                    compact_calls = []
                    for call in frame_calls:
                        compact = _compact_read(call)
                        _assert_read_integrity(compact, expected_max_noncond=max_noncond_maps)
                        compact_calls.append(compact)
                    row["available_noncond_frames"] = noncond_count
                    row["attention_reads"] = compact_calls
                    calls.extend(compact_calls)
                    result_rows.append(row)
                    write(sequence_dir / "frames.json", result_rows)
                    write(output_dir / "progress.json", {
                        "status": "running",
                        "sequence": sequence,
                        "condition": condition_name,
                        "frame": frame,
                        "targets_done": len(result_rows),
                        "targets_total": final_frame - prompt,
                    })
                    print(f"[{condition_name}] {sequence}: frame {frame}/{len(frames)-1}", flush=True)
                if save_replay_bank:
                    payload = {"schema":1,"object_id":1,"num_frames":state["num_frames"],
                        "prompt":prompt,"outputs":state["output_dict_per_obj"][0]}
                    bank_path = sequence_dir / "replay_bank.pt"
                    torch.save(payload,bank_path)
                    write(sequence_dir / "replay_bank_receipt.json", {
                        "scope":"bounded archived-treatment replay; incomplete future timeline",
                        "through_frame":final_frame,"prompt":prompt,"num_frames":state["num_frames"],
                        "object_id":1,"bank_sha256":digest(bank_path),
                        "payload_sha256":fingerprint(payload)["sha256"],
                        "checkpoint_sha256":digest(Path(checkpoint)),
                        "model_config":args.model_config,
                        "manifest_sha256":digest(ROOT/"data/polypgen_sequence.jsonl"),
                        "all_archived_rows_match":True})
                elapsed = time.monotonic() - started
        finally:
            if iterator is not None:
                iterator.close()
            model.reset_state(state)

    if [row["frame"] for row in result_rows] != list(range(prompt + 1, final_frame+1)):
        raise RuntimeError(f"Incomplete full-video propagation: {sequence}/{condition}")
    if len(calls) != len(result_rows) * 4:
        raise RuntimeError("Not all four native attention layers were audited per frame")
    if model.num_maskmem != initial_num_maskmem or not torch.equal(model.maskmem_tpos_enc.detach(), initial_tpos):
        raise RuntimeError("Native temporal slots were not restored exactly")
    final_parameter_hash = parameter_digest(model)
    if final_parameter_hash != initial_parameter_hash:
        raise RuntimeError("SAM2 parameters changed during the gate")
    del model
    gc.collect()
    return result_rows, {
        "sequence": sequence,
        "condition": condition,
        "sam2_revision": revision,
        "targets": len(result_rows),
        "attention_layer_calls": len(calls),
        "wall_seconds": elapsed,
        "parameter_hash_unchanged": True,
        "temporal_slots_restored": True,
        "native_exact_frames_checked": native_exact_checks,
        "expanded_history_reads": sum(row["available_noncond_frames"] > 6 for row in result_rows),
        "maximum_history_frames": max(row["available_noncond_frames"] for row in result_rows),
    }


def compare_matched_conditions(results):
    for sequence in SEQUENCES:
        generic = {row["frame"]: row for row in results["global"][sequence]}
        regional = {row["frame"]: row for row in results["regional"][sequence]}
        if generic.keys() != regional.keys():
            raise RuntimeError("Generic and regional arms cover different targets")
        for frame in generic:
            for g_read, r_read in zip(
                generic[frame]["attention_reads"], regional[frame]["attention_reads"], strict=True
            ):
                if g_read["layer"] != r_read["layer"]:
                    raise RuntimeError("Matched conditions differ in attention layer order")
                if g_read["pointer_keys"] != r_read["pointer_keys"]:
                    raise RuntimeError("Matched conditions changed the pointer count")
                if g_read["frames"] != r_read["frames"]:
                    raise RuntimeError(
                        f"Matched per-frame/local token budget differs at {sequence}/{frame}"
                    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    protocol = json.loads(PROTOCOL.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint digest differs from the frozen protocol")
    if digest(manifest) != "80d83e6a4099fc727e90bf06757ae81b3292d24dc5651daf1457f4a54e248c7f":
        raise RuntimeError("PolypGen manifest differs from the archived native run")
    native_setup = verify_native_setup(args.checkpoint, manifest)
    for source, expected in native_setup.get("sources", {}).items():
        if not source.startswith("external/sam2/"):
            continue
        relative = source.removeprefix("external/sam2/")
        if digest(SAM2_REFERENCE / relative) != expected:
            raise RuntimeError(f"Clean SAM2 checkout does not reproduce native source {relative}")
    from modeling.region_constrained_tome import author_tome
    tome = author_tome()
    if tome.__name__ != "_aadeseg_author_tome_merge":
        raise RuntimeError("Did not load the pinned upstream ToMe matcher")

    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_refs = {}
    prompts = {}
    receipts = {}
    for name in SEQUENCES:
        frames = videos[name]
        prompt, rows, receipt = native_reference(
            NATIVE_RESULTS.parent, native_results, native_protocol, name, frames
        )
        if len(frames) - prompt - 1 != len(rows):
            raise RuntimeError(f"Native reference does not cover {name}")
        native_refs[name], prompts[name], receipts[name] = rows, prompt, receipt

    source_hashes = {source: digest(ROOT / source) for source in SOURCES}
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "native_setup_sha256": digest(NATIVE_SETUP),
        "native_results_sha256": digest(NATIVE_RESULTS),
        "sam2_revision": NATIVE_REVISION,
        "sam2_reference": str(SAM2_REFERENCE),
        "source_hashes": source_hashes,
        "sequences": list(SEQUENCES),
        "conditions": list(CONDITIONS),
        "device": "cpu",
        "threshold": "native 0-logit evaluation convention; no method confidence gate",
        "native_receipts": receipts,
        "status": "engineering gate in progress",
    }
    write(args.output_dir / "setup.json", setup)
    all_rows, summaries = {condition: {} for condition in CONDITIONS}, {}
    for name in SEQUENCES:
        for condition in CONDITIONS:
            frames = videos[name]
            rows, summary = run_condition(
                condition, name, frames, prompts[name], native_refs[name],
                args.checkpoint, args.output_dir,
            )
            all_rows[condition][name] = rows
            summaries[f"{condition}/{name}"] = summary
            write(args.output_dir / "summaries.json", summaries)

    matched = compare_matched_conditions(all_rows)
    for condition in CONDITIONS:
        for name in SEQUENCES:
            if summaries[f"{condition}/{name}"]["expanded_history_reads"] <= 0:
                raise RuntimeError(f"Sequence never exercised expanded history: {condition}/{name}")
    results = {
        "matched_per_frame_and_local_budget": matched,
        "summary": summaries,
        "sequence_targets": {
            condition: {name: rows for name, rows in seqs.items()}
            for condition, seqs in all_rows.items()
        },
    }
    write(args.output_dir / "results.json", results)
    write(args.output_dir / "completion.json", {
        "passed": True,
        "classification": "real-video engineering gate; not a Dice efficacy result",
        "sequences": list(SEQUENCES),
        "conditions": list(CONDITIONS),
        "matched_budget": True,
        "native_exact_replay_through_six_frames": True,
        "sam2_frozen": True,
        "temporal_slots_restored": True,
        "full_cohort_authorized_next": True,
        "completion_utc_local_clock": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    })
    write(args.output_dir / "progress.json", {"status": "complete", "passed": True})
    print(f"PASS: real-video engineering gate saved to {args.output_dir}")


if __name__ == "__main__":
    main()

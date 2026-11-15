"""Read-only fixed-bank token-group diagnostic at Seq5 frames 157 and 184.

Uses the already-gated WithinFrameTokenRead implementation. It does not train,
write memory, change SAM2 weights, or select a token group for deployment.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.cfbi_training import capture_training_read, clone_tensors
from modeling.seq5_failure_window_utility import (
    FailureWindowTokenRead,
    TARGETS,
    TILE_NAMES,
    summarize_target,
    target_sources,
    validate_target_sources,
)
from modeling.complementary_memory.integration import _tensor_sha256
from scripts import run_complementary_memory_pilot as base
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_replay import check_exact, check_layout
from scripts.run_regional_token_selection import replay, scored
from scripts.run_wider_history_tome_gate import SAM2_REFERENCE, activate_reference_repo


PROTOCOL_PATH = "protocols/seq5_failure_window_token_utility_v1.json"
NATIVE_REFERENCE = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/pointer_conditioned_tome_polypgen_v1/"
    "full_attempt01/native/seq5/frames.json"
)
_PROJECT_SOURCES = set((
    PROTOCOL_PATH,
    "modeling/seq5_failure_window_utility.py",
    "modeling/within_frame_token_utility.py",
    "modeling/cfbi_training.py",
    "modeling/complementary_memory/integration.py",
    "modeling/native_memory_layout.py",
    "scripts/run_seq5_failure_window_token_utility.py",
    "scripts/run_complementary_memory_pilot.py",
    "scripts/run_regional_token_selection.py",
    "scripts/run_complementary_memory_replay.py",
    "scripts/run_author_baseline.py",
    "scripts/run_wider_history_tome_gate.py",
    "scripts/bank_integrity.py",
    "scripts/memory_token_io.py",
    "datasets/io.py",
    "datasets/video.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "external/sam2_diagnostic_reference/sam2/sam2_video_predictor.py",
    "external/sam2_diagnostic_reference/sam2/modeling/sam2_base.py",
    "external/sam2_diagnostic_reference/sam2/modeling/memory_attention.py",
    "external/sam2_diagnostic_reference/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
))
SOURCE_FILES = tuple(sorted(_PROJECT_SOURCES))
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _raw_metrics(model, state, observer, frames, frame, result, trace, output):
    if len(observer.raw_decoder_outputs) != 1:
        raise RuntimeError("Expected one pre-presence decoder capture for each replay")
    raw = observer.raw_decoder_outputs.pop()
    visible = model._get_orig_video_res_output(state, result["pred_masks"])[1]
    raw_tensor = raw["lowres_logits"][None, None]
    raw_full = model._get_orig_video_res_output(state, raw_tensor)[1]
    visible_row = scored(visible, frames, frame, output / "visible", {})
    raw_row = scored(raw_full, frames, frame, output / "raw", {})
    score = float(result["object_score_logits"].detach().float().reshape(-1)[0].cpu())
    if score != raw["object_score_logit"]:
        raise RuntimeError("Decoder object-presence score changed between decoder and track output")
    return {
        "visible": visible_row,
        "raw": raw_row,
        "object_score_logit": score,
        "raw_selected_mask_index": raw["selected_mask_index"],
        "raw_candidate_count": raw["candidate_count"],
        "visible_output_hash": tensor_hash(result["pred_masks"]),
        "raw_output_hash": _tensor_sha256(raw_tensor),
        "visible_foreground_pixels": int((visible > 0).sum()),
        "raw_foreground_pixels": int((raw_full > 0).sum()),
        "trace_layers": [
            {"layer": row["layer"], "frame_ids": row["frame_ids"], "counts": row["counts"],
             "selected_indices_sha256": row["selected_indices_sha256"],
             "pointer_keys": row["pointer_keys"]}
            for row in trace["selection_layers"]
        ],
    }


@torch.inference_mode()
def run_sequence(model, frames, prompt, output, setup):
    observer = FailureWindowTokenRead(model)
    parameter_sha = parameter_digest(model)
    records = []
    with tempfile.TemporaryDirectory(dir=output, prefix="seq5_native_frames_") as scratch:
        state, actual_prompt = base.prepare(model, frames, Path(scratch))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt index changed: {actual_prompt} != {prompt}")
        iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
        visited, pending = [], []
        # Capture the exact native source schedule during propagation. Counterfactual
        # replay is deferred until this observer context exits; nested readers would
        # replace SAM2's preparation hook and are rejected by the existing guard.
        with observer:
            while True:
                observer.arm, observer.source_frame = "native", None
                try:
                    frame, object_ids, original_logits = next(iterator)
                except StopIteration:
                    break
                if object_ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs at {frame}: {object_ids}")
                if frame != prompt + len(visited):
                    raise RuntimeError("Native propagation frame order changed")
                visited.append(frame)
                if frame not in TARGETS:
                    observer.calls.clear()
                    continue
                if frame >= len(frames) or not frames[frame].mask_path:
                    raise RuntimeError(f"Missing target annotation at {frame}")
                collected = clone_tensors(observer.calls[-1])
                sample = capture_training_read(model, state, frame)
                # The cloned prefix in sample is the fixed bank consumed by every
                # replay. The live inference bank will correctly grow as later
                # frames propagate, so it must not be compared to this target-time
                # snapshot after propagation has advanced.
                bank_sha = fingerprint(sample["output_dict"])["sha256"]
                expected = clone_tensors(
                    state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame]
                )
                pending.append({
                    "frame": frame,
                    "original_logits": original_logits.detach().clone(),
                    "collected": collected,
                    "sample": sample,
                    "sample_sha": fingerprint(sample)["sha256"],
                    "bank_sha": bank_sha,
                    "expected": expected,
                })
                observer.calls.clear()
                if frame == max(TARGETS):
                    break
        if visited != list(range(prompt, max(TARGETS) + 1)):
            raise RuntimeError("Incomplete native chronological prefix through final diagnostic target")
        iterator.close()

        completed_interventions = 0
        for captured in pending:
            frame = captured["frame"]
            sample, sample_sha = captured["sample"], captured["sample_sha"]
            bank_sha, original_logits = captured["bank_sha"], captured["original_logits"]
            observer.arm, observer.source_frame = "native", None
            observer.raw_decoder_outputs.clear()
            observer.capture_raw_decoder = True
            try:
                native_result, native_trace = replay(model, sample, observer)
            finally:
                observer.capture_raw_decoder = False
            check_exact(native_result, captured["expected"])
            check_layout(native_trace, captured["collected"])
            replayed_full = model._get_orig_video_res_output(state, native_result["pred_masks"])[1]
            if tensor_hash(replayed_full) != tensor_hash(original_logits):
                raise RuntimeError(f"Native/no-op full-resolution parity failed at {frame}")
            native_metrics = _raw_metrics(
                model, state, observer, frames, frame, native_result, native_trace,
                output / "masks" / f"target_{frame:06d}" / "native",
            )
            sources = target_sources(native_trace)
            if len(sources) != 5:
                raise RuntimeError(f"Expected five intermediate native sources; got {sources}")
            target_record = {
                "target_frame": frame,
                "native_source_frames": [span["frame"] for span in native_trace["memory_frames"]],
                "native_memory_trace_sha256": digest_bytes(json.dumps(
                    native_trace["memory_frames"], sort_keys=True, separators=(",", ":")
                ).encode()),
                "native_sample_sha256": sample_sha,
                "native_bank_sha256_before": bank_sha,
                "native": native_metrics,
                "sources": [],
            }
            for source_frame in sources:
                source_record = {"source_frame": source_frame, "tiles": []}
                for tile in TILE_NAMES:
                    observer.arm, observer.source_frame = tile, source_frame
                    observer.raw_decoder_outputs.clear()
                    observer.capture_raw_decoder = True
                    try:
                        result, trace = replay(model, sample, observer)
                    finally:
                        observer.capture_raw_decoder = False
                    check_layout(trace, native_trace)
                    if fingerprint(sample)["sha256"] != sample_sha:
                        raise RuntimeError("Read-only intervention mutated captured native inputs")
                    if fingerprint(sample["output_dict"])["sha256"] != bank_sha:
                        raise RuntimeError("Read-only intervention mutated the captured target-time memory bank")
                    expected_counts = [4096 - 256 if int(fid) == source_frame else 4096
                                       for fid in [s["frame"] for s in trace["memory_frames"]]]
                    if any(layer["counts"] != expected_counts for layer in trace["selection_layers"]):
                        raise RuntimeError("Ablation count/source mismatch in one or more memory layers")
                    metrics = _raw_metrics(
                        model, state, observer, frames, frame, result, trace,
                        output / "masks" / f"target_{frame:06d}" / f"source_{source_frame:06d}" / tile,
                    )
                    source_record["tiles"].append({
                        "tile": tile,
                        "removed_spatial_tokens_per_layer": 256,
                        "metrics": metrics,
                    })
                    completed_interventions += 1
                    write_json(output / "progress.json", {
                        "phase": "counterfactual_reads",
                        "pid": os.getpid(),
                        "target_frame": frame,
                        "source_frame": source_frame,
                        "tile": tile,
                        "completed_interventions": completed_interventions,
                        "expected_interventions": len(TARGETS) * len(sources) * len(TILE_NAMES),
                    })
                target_record["sources"].append(source_record)
            if fingerprint(sample["output_dict"])["sha256"] != bank_sha:
                raise RuntimeError("Captured target-time memory bank changed during counterfactuals")
            validate_target_sources(native_trace, sources)
            target_record["summary"] = summarize_target({
                "target_frame": frame,
                "native": native_metrics,
                "sources": [{"source_frame": s["source_frame"], "tiles": [
                    {"tile": r["tile"], "visible": r["metrics"]["visible"],
                     "raw": r["metrics"]["raw"], "object_score_logit": r["metrics"]["object_score_logit"]}
                    for r in s["tiles"]]} for s in target_record["sources"]],
            })
            records.append(target_record)
            write_json(output / "partial_results.json", {"targets": records})
            observer.calls.clear()
        model.reset_state(state)
    if parameter_digest(model) != parameter_sha:
        raise RuntimeError("SAM2 parameters changed during the read-only diagnostic")
    return {"targets": records, "parameter_sha256": parameter_sha}


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, default=NATIVE_REFERENCE)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol_path = ROOT / PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text())
    if protocol["model"]["sam2_revision"] != SAM2_REVISION:
        raise RuntimeError("Protocol SAM2 revision differs from the clean diagnostic checkout")
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from frozen Seq5 diagnostic protocol")
    if digest(manifest) != protocol["model"]["manifest_sha256"]:
        raise RuntimeError("PolypGen manifest differs from frozen diagnostic protocol")
    if not args.native_reference.is_file():
        raise FileNotFoundError(args.native_reference)
    native_rows = json.loads(args.native_reference.read_text())
    native_rows = {int(row["frame"]): row for row in native_rows}
    if not set(TARGETS).issubset(native_rows):
        raise RuntimeError("Frozen native reference lacks a requested target")
    videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
    frames = videos["seq5"]
    if any(frame >= len(frames) for frame in TARGETS):
        raise RuntimeError("Frozen target exceeds Seq5 manifest length")
    if not all((load_label_mask(frames[frame].mask_path, [1]) == 1).any() for frame in TARGETS):
        raise RuntimeError("Failure-window targets must be GT-present")

    revision = activate_reference_repo()
    if revision != SAM2_REVISION or str(SAM2_REFERENCE) not in sys.path:
        raise RuntimeError("Pinned clean SAM2 diagnostic checkout was not activated")
    repo = SAM2_REFERENCE
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    setup = {
        "protocol": protocol,
        "protocol_sha256": digest(protocol_path),
        "source_sha256": source_hashes,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "input_sha256": base.input_hashes(videos, ["seq5"])["seq5"],
        "native_reference": str(args.native_reference.resolve()),
        "native_reference_sha256": digest(args.native_reference),
        "device": "cpu",
        "torch_version": torch.__version__,
        "sequence": "seq5",
        "targets": list(TARGETS),
        "sam2_reference": str(SAM2_REFERENCE),
        "sam2_revision": revision,
        "interpretation": "Exploratory fixed-bank token-group intervention; no selector fitting or efficacy claim.",
    }
    write_json(output / "setup.json", setup)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.write(ROOT / name, arcname=name)
    start = time.monotonic()
    model = base.init_predictor(args.checkpoint, repo)
    # The prompt is pinned from the sequence manifest by the same first-labeled-frame policy.
    from scripts.memory_token_io import prompt_index
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    result = run_sequence(model, frames, prompt, output, setup)
    for frame_record in result["targets"]:
        native = native_rows[frame_record["target_frame"]]
        if native["prediction_sha256"] != frame_record["native"]["visible"]["prediction_sha256"]:
            raise RuntimeError(f"Native archived prediction hash differs at frame {frame_record['target_frame']}")
        if native["object_score"] != frame_record["native"]["object_score_logit"]:
            raise RuntimeError(f"Native archived object score differs at frame {frame_record['target_frame']}")
    if {name: digest(ROOT / name) for name in SOURCE_FILES} != source_hashes:
        raise RuntimeError("A frozen source changed during the run")
    final = {
        **result,
        "sequence": "seq5",
        "targets": result["targets"],
        "completion": "complete",
        "native_reference_exact_at_targets": True,
    }
    write_json(output / "results.json", final)
    write_json(output / "completion.json", {
        "complete": True,
        "target_count": len(result["targets"]),
        "intervention_count": sum(len(s["tiles"]) for t in result["targets"] for s in t["sources"]),
        "results_sha256": digest(output / "results.json"),
        "setup_sha256": digest(output / "setup.json"),
        "native_reference_exact_at_targets": True,
    })
    write_json(output / "runtime.json", {"elapsed_seconds": time.monotonic() - start})
    model = None
    gc.collect()


if __name__ == "__main__":
    main()

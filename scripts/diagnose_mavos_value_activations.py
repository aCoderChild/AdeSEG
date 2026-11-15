"""Post-hoc diagnostic of learned changes to actual SAM2 memory value tokens.

Replays the already-evaluated validation trajectories with the saved C1/C2
adapters and records scalar statistics on each complete stored-frame block
immediately before SAM2's frozen v_proj. It does not read ground truth after
the initial prompt, compute segmentation metrics, or touch the held-out test
split. All output is written to a caller-provided Drive directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.mavos_focal_value_modulation import MAVOSMemoryValueModulation
from scripts.mavos_value_utils import activate_author_repo, prepare, atomic_json
from scripts.memory_token_io import digest, parameter_digest, prompt_index
from scripts.run_author_baseline import build_author_predictor
from scripts.run_mavos_value_contrastive import (
    MANIFEST, PROTOCOL, VALIDATION, make_adapters,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sequence_summary(rows: list[dict]) -> dict:
    result = {}
    for arm in ("segmentation_only", "segmentation_plus_contrastive"):
        for sequence in VALIDATION:
            selected = [r for r in rows if r["arm"] == arm and r["sequence"] == sequence]
            if not selected:
                continue
            for layer in range(4):
                values = [r for r in selected if r["layer"] == layer]
                relative = [r["relative_l2_mean"] for r in values]
                result[f"{arm}/{sequence}/layer{layer}"] = {
                    "memory_frame_blocks": len(values),
                    "relative_l2_mean_across_blocks": statistics.mean(relative),
                    "relative_l2_median_across_blocks": statistics.median(relative),
                    "relative_l2_p95_across_blocks": sorted(relative)[
                        min(len(relative) - 1, math.ceil(0.95 * len(relative)) - 1)
                    ],
                    "delta_rms_mean_across_blocks": statistics.mean(
                        r["delta_rms"] for r in values
                    ),
                    "cosine_mean_across_blocks": statistics.mean(
                        r["cosine_mean"] for r in values
                    ),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pilot-output", required=True, type=Path)
    parser.add_argument("--diagnostic-output", required=True, type=Path)
    args = parser.parse_args()
    out = args.diagnostic_output.resolve()
    if not out.is_dir():
        raise FileNotFoundError("Create the diagnostic Drive directory and freeze protocol.json first")
    os.environ["TMPDIR"] = str(out)
    pilot = args.pilot_output.resolve()
    diagnostic_protocol_path = out / "protocol.json"
    if not diagnostic_protocol_path.is_file():
        raise FileNotFoundError(
            "Freeze diagnostic protocol.json in the Drive output directory before replay"
        )
    diagnostic_protocol = json.loads(diagnostic_protocol_path.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    completion = json.loads((pilot / "completion.json").read_text())
    if completion.get("test_run") is not False:
        raise RuntimeError("Expected the frozen pilot to have withheld its test split")
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the completed frozen pilot")
    if digest(MANIFEST) != protocol["data"]["manifest_sha256_at_freeze"]:
        raise RuntimeError("Manifest differs from the completed frozen pilot")

    arm_paths = {
        "segmentation_only": pilot / "training/segmentation_only/adapters.pt",
        "segmentation_plus_contrastive": pilot / "training/segmentation_plus_contrastive/adapters.pt",
    }
    for arm, path in arm_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained adapter for {arm}: {path}")
        if diagnostic_protocol["adapter_sha256"].get(arm) != digest(path):
            raise RuntimeError(f"Adapter artifact changed after diagnostic freeze: {arm}")
    if diagnostic_protocol["validation_results_sha256"] != digest(pilot / "validation_results.json"):
        raise RuntimeError("Frozen validation results changed after diagnostic protocol was written")
    if diagnostic_protocol["completion_sha256"] != digest(pilot / "completion.json"):
        raise RuntimeError("Frozen completion receipt changed after diagnostic protocol was written")
    if diagnostic_protocol["script_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("Diagnostic script changed after protocol freeze")
    if diagnostic_protocol["hook_source_sha256"] != sha256_file(
        Path(__file__).resolve().parents[1] / "modeling/mavos_focal_value_modulation.py"
    ):
        raise RuntimeError("Activation hook source changed after protocol freeze")
    repo, revision = activate_author_repo()
    if revision != protocol["model"]["sam2_revision"]:
        raise RuntimeError("SAM2 revision differs from the frozen pilot")
    videos = load_manifest(MANIFEST, "dev")
    if not set(VALIDATION).issubset(videos):
        raise RuntimeError("Validation sequences missing from manifest")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = build_author_predictor(argparse.Namespace(
        method="sam2", device="cpu", checkpoint=args.checkpoint,
        model_config=protocol["model"]["configuration"],
    ), repo)
    model.requires_grad_(False)
    model.eval()
    original_model_hash = parameter_digest(model)
    summaries_path = out / "activation_stats.jsonl"
    status_path = out / "run_status.json"
    all_rows: list[dict] = []
    completed_sequences = []
    status = {
        "phase": "prepared",
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_sequences": completed_sequences,
    }
    atomic_json(status_path, status)

    with summaries_path.open("w", encoding="utf-8") as output:
        for arm, adapter_path in arm_paths.items():
            state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            adapters = make_adapters(model)
            adapters.load_state_dict(state, strict=True)
            adapters.eval()
            for sequence in VALIDATION:
                frames = videos[sequence]
                prompt = prompt_index(
                    frames, {"initialization": "first_labeled_frame"}
                )
                sequence_rows = 0
                started = time.monotonic()
                status = {
                    "phase": "replay",
                    "pid": os.getpid(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "arm": arm,
                    "sequence": sequence,
                    "prompt_frame": prompt,
                    "expected_queries": len(frames) - prompt - 1,
                    "completed_sequences": list(completed_sequences),
                }
                atomic_json(status_path, status)
                with tempfile.TemporaryDirectory(
                    prefix=f"activation_{arm}_{sequence}_", dir=out
                ) as temp:
                    state_obj, actual_prompt = prepare(model, frames, Path(temp))
                    if actual_prompt != prompt:
                        raise RuntimeError("Prompt differs from the frozen evaluation setup")
                    model._run_state = state_obj
                    observer = NativeComplementaryRead(
                        model, None, "regional", inference_state=state_obj,
                        trace_only=True,
                    )
                    integration = MAVOSMemoryValueModulation(
                        model, adapters, record_value_receipts=True,
                        record_modulation_stats=True,
                    )
                    iterator = None
                    try:
                        with observer, integration, torch.inference_mode():
                            iterator = model.propagate_in_video(
                                state_obj, start_frame_idx=prompt
                            )
                            for frame, object_ids, _logits in iterator:
                                if object_ids != [1]:
                                    raise RuntimeError(
                                        f"Unexpected persistent object IDs on {sequence}/{frame}"
                                    )
                                if frame == prompt:
                                    continue
                                if len(observer.calls) != sequence_rows + 1:
                                    raise RuntimeError("Missing or repeated memory-read trace")
                                trace = observer.calls[-1]
                                value_calls = integration.value_calls[-4:]
                                total_blocks = sum(call["frame_blocks"] for call in value_calls)
                                all_blocks = (
                                    integration.modulation_stats[-total_blocks:]
                                    if total_blocks else []
                                )
                                if trace.get("layers") != [0, 1, 2, 3]:
                                    raise RuntimeError("Not all native attention layers were observed")
                                if [x["layer"] for x in value_calls] != [0, 1, 2, 3]:
                                    raise RuntimeError("Value hook receipts are not layer aligned")
                                if not all(x["pointer_tail_bit_identical"] for x in value_calls):
                                    raise RuntimeError("Pointer values changed during diagnostic replay")
                                spans = trace["memory_frames"]
                                pending_rows = []
                                block_cursor = 0
                                for layer in range(4):
                                    call = value_calls[layer]
                                    block_count = int(call["frame_blocks"])
                                    blocks = all_blocks[block_cursor:block_cursor + block_count]
                                    block_cursor += block_count
                                    if any(x["layer"] != layer for x in blocks):
                                        raise RuntimeError("Activation summaries are not layer aligned")
                                    if len(blocks) != len(spans):
                                        raise RuntimeError("Memory frame blocks do not align to trace")
                                    if call["frame_blocks"] != len(spans):
                                        raise RuntimeError("Native value frame count differs from observer")
                                    if [x["frame_block"] for x in blocks] != list(range(len(spans))):
                                        raise RuntimeError("Frame-block ordering differs from native memory trace")
                                    for block, span in zip(blocks, spans):
                                        token_count = int(span["stop"]) - int(span["start"])
                                        if token_count != block["token_count"] or token_count != 4096:
                                            raise RuntimeError("Memory span is not one complete 64x64 frame")
                                        row = {
                                            "arm": arm,
                                            "sequence": sequence,
                                            "query_frame": int(frame),
                                            "memory_frame": int(span["frame"]),
                                            "memory_age_sampled_frames": int(frame) - int(span["frame"]),
                                            "layer": layer,
                                            **{k: block[k] for k in (
                                                "token_count", "raw_rms", "delta_rms",
                                                "relative_l2_mean", "relative_l2_median",
                                                "relative_l2_p95", "cosine_mean",
                                                "changed_token_fraction",
                                            )},
                                        }
                                        pending_rows.append(row)
                                        all_rows.append(row)
                                if block_cursor != len(all_blocks):
                                    raise RuntimeError("Unused activation summary blocks remain")
                                output.writelines(
                                    json.dumps(row, sort_keys=True) + "\n"
                                    for row in pending_rows
                                )
                                if sequence_rows % 10 == 9:
                                    output.flush()
                                    os.fsync(output.fileno())
                                sequence_rows += 1
                                if sequence_rows % 25 == 0:
                                    status = {
                                        "phase": "replay", "pid": os.getpid(),
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                        "arm": arm, "sequence": sequence,
                                        "completed_queries": sequence_rows,
                                        "expected_queries": len(frames) - prompt - 1,
                                        "completed_sequences": list(completed_sequences),
                                    }
                                    atomic_json(status_path, status)
                    finally:
                        if iterator is not None:
                            iterator.close()
                        with torch.inference_mode():
                            model.reset_state(state_obj)
                        model._run_state = None
                expected = len(frames) - prompt - 1
                if sequence_rows != expected:
                    raise RuntimeError(
                        f"Incomplete diagnostic replay on {arm}/{sequence}: "
                        f"{sequence_rows}/{expected} query frames"
                    )
                if len(observer.calls) != expected:
                    raise RuntimeError("Final observer trace count differs from query count")
                output.flush()
                os.fsync(output.fileno())
                if parameter_digest(model) != original_model_hash:
                    raise RuntimeError("SAM2 pretrained parameters changed during replay")
                completed_sequences.append(f"{arm}/{sequence}")
                atomic_json(out / f"{arm}_{sequence}_completion.json", {
                    "complete": True, "query_frames": sequence_rows,
                    "memory_frame_block_records": sum(
                        1 for row in all_rows
                        if row["arm"] == arm and row["sequence"] == sequence
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                    "pointer_values_unchanged": True,
                    "layers_observed": [0, 1, 2, 3],
                })
                atomic_json(status_path, {
                    "phase": "replay", "pid": os.getpid(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed_sequences": list(completed_sequences),
                    "last_completed": f"{arm}/{sequence}",
                })
            del adapters

    if parameter_digest(model) != original_model_hash:
        raise RuntimeError("SAM2 pretrained parameters changed before summary")
    summary = {
        "diagnostic_only": True,
        "does_not_update_frozen_validation_metrics": True,
        "test_split_used": False,
        "model_sha256": original_model_hash,
        "adapter_artifacts": {arm: digest(path) for arm, path in arm_paths.items()},
        "memory_frame_block_records": len(all_rows),
        "summary_by_arm_sequence_layer": _sequence_summary(all_rows),
        "interpretation_limit": (
            "These are descriptive activation changes on separate autoregressive replays. "
            "They do not establish causal benefit or predict Dice; the C1 and C2 trajectories "
            "may differ slightly."
        ),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(status_path, {
        "phase": "complete", "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_sequences": list(completed_sequences),
        "memory_frame_block_records": len(all_rows),
    })
    print(json.dumps({"phase": "complete", "records": len(all_rows), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()

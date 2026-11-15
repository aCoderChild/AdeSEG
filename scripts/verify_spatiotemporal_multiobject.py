"""Engineering-only object-bank isolation check on controlled real frames."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from modeling.spatiotemporal_tome_value_exchange import NativeSpatiotemporalValueExchange
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_spatiotemporal_tome_value_exchange_cohort import load_protocol, preflight
from scripts.run_wider_history_tome_gate import CHECKPOINT_DEFAULT, SAM2_REFERENCE, activate_reference_repo

MANIFEST = ROOT / "data/polypgen_sequence.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.gate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.gate.with_suffix(args.gate.suffix + ".sha256").read_text().strip():
        raise RuntimeError("Gate protocol checksum differs")
    gate = json.loads(raw)
    for path, expected in gate["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Gate source differs: {path}")
    base_path = ROOT / gate["base_protocol"]
    if digest(base_path) != gate["base_protocol_sha256"]:
        raise RuntimeError("Base protocol differs")
    base = load_protocol(base_path)
    preflight(base)
    replay_gate = Path(gate["durable_main_gate_receipt"])
    if digest(replay_gate) != gate["durable_main_gate_sha256"] or json.loads(replay_gate.read_text())["status"] != "passed":
        raise RuntimeError("Durable-main replay evidence differs or did not pass")
    output = validate_output(args.output_dir)
    if not output.is_relative_to(Path(gate["output_root"]).resolve()):
        raise RuntimeError("Output is outside the gate root")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(base["runtime"]["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(base["runtime"]["seed"])
    activate_reference_repo()
    model = build_author_predictor(SimpleNamespace(
        method="sam2", device="cpu", checkpoint=CHECKPOINT_DEFAULT,
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
    ), SAM2_REFERENCE)
    model.requires_grad_(False)
    initial_parameters = parameter_digest(model)
    videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}
    frames = videos[gate["sequence"]][gate["start_frame"]:gate["start_frame"] + gate["frame_count"]]
    if len(frames) != gate["frame_count"]:
        raise RuntimeError("Fixture frames missing")
    for frame, expected in zip(frames, gate["frame_hashes"], strict=True):
        if digest(frame.image_path) != expected["image_sha256"] or digest(frame.mask_path) != expected["mask_sha256"]:
            raise RuntimeError("Fixture frame or mask differs")
    mask = load_label_mask(frames[0].mask_path, [1]) == 1
    xs = np.nonzero(mask)[1]
    split = int(np.median(xs))
    columns = np.arange(mask.shape[1])[None, :]
    left, right = mask & (columns < split), mask & (columns >= split)
    right_alt = right & (columns >= int(np.quantile(np.nonzero(right)[1], 0.75)))
    if not left.any() or not right_alt.any() or np.array_equal(right, right_alt):
        raise RuntimeError("Fixture needs two nonempty distinguishable synthetic objects")

    with tempfile.TemporaryDirectory(prefix="fixture_", dir=output) as directory:
        stage_frames(frames, Path(directory))

        def rollout(mode, second_mask):
            from sam2.modeling.position_encoding import PositionEmbeddingSine
            for module in model.modules():
                if isinstance(module, PositionEmbeddingSine):
                    module.cache.clear()
            state = model.init_state(video_path=directory, offload_video_to_cpu=True, offload_state_to_cpu=False)
            model.add_new_mask(state, frame_idx=0, obj_id=7, mask=left)
            model.add_new_mask(state, frame_idx=0, obj_id=42, mask=second_mask)
            reader = None if mode == "native" else NativeSpatiotemporalValueExchange(model, mode=mode)
            if reader is not None:
                reader.object_id_by_output_dict = {
                    id(state["output_dict_per_obj"][index]): object_id
                    for object_id, index in state["obj_id_to_idx"].items()
                }
            records = {}
            with (reader if reader is not None else nullcontext()):
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=len(frames) - 1):
                    for index, object_id in enumerate(ids):
                        bank = state["output_dict_per_obj"][state["obj_id_to_idx"][object_id]]
                        stored = bank["cond_frame_outputs"].get(frame, bank["non_cond_frame_outputs"].get(frame))
                        records[f"{frame}:{object_id}"] = {
                            "prediction": tensor_hash(logits[index:index + 1]),
                            "bank": fingerprint(stored)["sha256"],
                        }
            if parameter_digest(model) != initial_parameters:
                raise RuntimeError("SAM2 parameters changed")
            calls = [] if reader is None else reader.calls
            return records, calls

        with torch.inference_mode():
            native, _ = rollout("native", right)
            noop, noop_calls = rollout("noop", right)
            cross, cross_calls = rollout("cross_frame", right)
            changed, changed_calls = rollout("cross_frame", right_alt)

    expected = {f"{frame}:{object_id}" for frame in range(len(frames)) for object_id in (7, 42)}
    if any(set(records) != expected for records in (native, noop, cross, changed)):
        raise RuntimeError("Object/frame outputs are incomplete")
    checks = {
        "noop_matches_native_both_object_banks": native == noop,
        "object_7_unchanged_when_object_42_prompt_changes": all(cross[f"{i}:7"] == changed[f"{i}:7"] for i in range(len(frames))),
        "object_42_perturbation_changes_output": any(cross[f"{i}:42"] != changed[f"{i}:42"] for i in range(len(frames))),
        "all_attention_calls_identify_object": all(call["object_id"] in (7, 42) for calls in (noop_calls, cross_calls, changed_calls) for call in calls),
        "four_layers_per_object_per_target": all(
            sorted(call["layer"] for call in calls if call["object_id"] == obj and call["frame"] == frame) == [0, 1, 2, 3]
            for calls in (noop_calls, cross_calls, changed_calls)
            for obj in (7, 42) for frame in range(1, len(frames))
        ),
        "cross_frame_groups_observed_for_both_objects": all(any(call["object_id"] == obj and call["cross_frame_groups"] > 0 for call in cross_calls) for obj in (7, 42)),
        "sam2_parameters_unchanged": parameter_digest(model) == initial_parameters,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "scope": "Two synthetic instance IDs derived from one real polyp mask; engineering isolation fixture, not an anatomical multi-object benchmark.",
              "gate_sha256": hashlib.sha256(raw).hexdigest(),
              "call_counts": {"noop": len(noop_calls), "cross": len(cross_calls), "changed": len(changed_calls)},
              "records": {"native": native, "noop": noop, "cross": cross, "changed": changed}}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks, "call_counts": result["call_counts"]}, indent=2), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Multi-object isolation gate failed")


if __name__ == "__main__":
    main()

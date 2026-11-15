"""Real-frame engineering gate for frozen MCTF memory-value consolidation."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.attention_distortion_observer import _sha256_tensor
from modeling.mctf_memory_value import MCTFMemoryValue
from modeling.native_history_cache import cached_read, load_bank
from modeling.native_value_consolidation import NativeValueConsolidation
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_attention_distortion_diagnostic import _compare_prompt
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_native_value_consolidation_gate import track_read, require_same_prediction
from scripts.run_regional_refinement import write


def _audit_comparability(audits):
    base = audits["noop"]
    def source_schedule(row):
        sources = row.get("source_frames", row.get("frames", []))
        return [(int(item["frame"]), int(item["t_pos"])) for item in sources]

    def spatial_count(row):
        return int(row.get("spatial_keys", row.get("spatial_keys_before", -1)))

    for arm in ("tome_value", "mctf_uniform", "mctf_attention"):
        rows = audits[arm]
        for layer in range(4):
            reference, row = base[layer], rows[layer]
            if row["layer"] != layer:
                raise RuntimeError("Missing or repeated memory attention layer")
            if row["qkv_input_sha256"][1:] != reference["qkv_input_sha256"][1:]:
                raise RuntimeError(f"{arm}: native memory K/V changed at layer {layer}")
            if layer == 0 and row["qkv_input_sha256"][0] != reference["qkv_input_sha256"][0]:
                raise RuntimeError(f"{arm}: first-layer query differs from native")
            if source_schedule(row) != source_schedule(reference):
                raise RuntimeError(f"{arm}: native frame schedule changed")
            if row["pointer_keys"] != reference["pointer_keys"]:
                raise RuntimeError(f"{arm}: pointer token count changed")
            if spatial_count(row) != spatial_count(reference):
                raise RuntimeError(f"{arm}: dense spatial token grid changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    checksum = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    if protocol.get("status") != "frozen_for_engineering_gate" or digest(args.protocol) != checksum.read_text().strip():
        raise RuntimeError("Engineering protocol is not frozen or its hash differs")
    out = validate_output(args.output_dir)
    if not out.is_relative_to(Path(protocol["output_root"]).resolve()) or out.exists():
        raise RuntimeError("Use a new child of the frozen Drive output root")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")

    input_path = Path(protocol["input_plan"])
    native_path = Path(protocol["native_plan"])
    if digest(input_path) != protocol["input_plan_sha256"] or digest(native_path) != protocol["native_plan_sha256"]:
        raise RuntimeError("Frozen native-bank input or baseline plan changed")
    inputs = json.loads(input_path.read_text())
    native_plan = json.loads(native_path.read_text())
    config = native_plan["protocol"]
    repo = ROOT / config["sam2_repository"]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != config["sam2_revision"]:
        raise RuntimeError("SAM2 source revision differs from the frozen baseline")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        raise RuntimeError("SAM2 has tracked source changes")
    if "sam2" in sys.modules:
        raise RuntimeError("SAM2 was imported before selecting its pinned repository")
    if digest(Path(config["checkpoint"])) != config["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint changed")
    manifest = ROOT / config["manifest"]
    if digest(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Video manifest changed")
    sys.path.insert(0, str(repo))
    torch.set_num_threads(protocol["runtime"]["threads"])
    torch.use_deterministic_algorithms(protocol["runtime"]["deterministic_algorithms"])
    torch.manual_seed(protocol["runtime"]["seed"])
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    out.mkdir(parents=True)
    write(out / "setup.json", {"protocol": protocol, "protocol_sha256": digest(args.protocol)})

    results = {}
    try:
        for sequence, targets in protocol["targets"].items():
            verified = inputs["banks"][sequence]
            by_frame = {row["frame"]: row for row in verified["targets"]}
            if any(frame not in by_frame for frame in targets):
                raise RuntimeError("A gate target is absent from the verified cached plan")
            for frame in targets:
                if digest(Path(by_frame[frame]["image"])) != by_frame[frame]["image_sha256"]:
                    raise RuntimeError("A cached target image changed")
            folder = Path(verified["cache_directory"])
            if digest(folder / "receipt.json") != verified["receipt_sha256"]:
                raise RuntimeError("Native cached bank receipt changed")
            receipt = json.loads((folder / "receipt.json").read_text())
            payload = load_bank(
                folder / "bank.pt", receipt, config["checkpoint_sha256"],
                config["manifest_sha256"], config["model_config"],
            )
            payload_hash = fingerprint(payload)["sha256"]
            model = build_author_predictor(SimpleNamespace(
                method="sam2", device="cpu", checkpoint=Path(config["checkpoint"]),
                model_config=config["model_config"],
            ), repo)
            weights_hash = parameter_digest(model)
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
                raise RuntimeError("Native SAM2 evaluation setup changed")
            frames = videos[sequence]
            results[sequence] = []
            with tempfile.TemporaryDirectory(dir=out) as temporary:
                with torch.inference_mode():
                    state, prompt = prepare(model, frames, Path(temporary))
                    model.propagate_in_video_preflight(state)
                    if prompt != payload["prompt"] or len(frames) != payload["num_frames"]:
                        raise RuntimeError("Native prompt/video differs from cached bank")
                    object_index = state["obj_id_to_idx"][payload["object_id"]]
                    _compare_prompt(state["output_dict_per_obj"][object_index], payload["outputs"], prompt, sequence)
                    for frame in targets:
                        write(out / "progress.json", {"sequence": sequence, "frame": frame, "status": "running"})
                        sample = cached_read(model, state, frame, payload)
                        bank_hash = fingerprint(sample["output_dict"])["sha256"]
                        native = track_read(model, sample, frame)
                        require_same_prediction(native, {
                            "pred_masks": sample["native_logits"],
                            "object_score_logits": sample["native_object_score"],
                        }, "Native cached replay")
                        arms, audits = {}, {}
                        adapters = {
                            "noop": lambda: NativeValueConsolidation(model, mode="noop"),
                            "tome_value": lambda: NativeValueConsolidation(model, mode="value"),
                            "mctf_uniform": lambda: MCTFMemoryValue(model, importance=False),
                            "mctf_attention": lambda: MCTFMemoryValue(model, importance=True),
                        }
                        for name, factory in adapters.items():
                            with factory() as adapter:
                                prediction = track_read(model, sample, frame)
                            if [row["layer"] for row in adapter.calls] != [0, 1, 2, 3]:
                                raise RuntimeError(f"{name}: all four SAM2 attention layers were not observed once")
                            if name == "noop":
                                require_same_prediction(prediction, native, "Native/no-op parity")
                            if name.startswith("mctf") and any(not row["key_unchanged"] for row in adapter.calls):
                                raise RuntimeError(f"{name}: SAM2 memory keys changed")
                            arms[name] = {
                                "mask_sha256": _sha256_tensor(prediction["pred_masks"]),
                                "presence_sha256": _sha256_tensor(prediction["object_score_logits"]),
                            }
                            audits[name] = adapter.calls
                        _audit_comparability(audits)
                        if fingerprint(sample["output_dict"])["sha256"] != bank_hash:
                            raise RuntimeError("Cached native source bank changed during read")
                        if model.num_maskmem != 7 or parameter_digest(model) != weights_hash:
                            raise RuntimeError("MCTF changed SAM2 parameters or temporal slots")
                        results[sequence].append({
                            "frame": frame,
                            "native_replay_exact": True,
                            "noop_parity_exact": True,
                            "source_bank_unchanged": True,
                            "arms": arms,
                            "audits": audits,
                        })
                        write(out / "results.json", results)
                        del sample, native, prediction, adapter
                    model.reset_state(state)
            if fingerprint(payload)["sha256"] != payload_hash or parameter_digest(model) != weights_hash:
                raise RuntimeError("Input cache or SAM2 weights changed")
            del model, state, payload
            gc.collect()
        if any(digest(ROOT / name) != expected for name, expected in protocol["source_hashes"].items()):
            raise RuntimeError("A frozen local source file changed during the gate")
        write(out / "completion.json", {
            "passed": True,
            "scope": "cached real-frame engineering only; not Dice efficacy",
            "targets": sum(map(len, results.values())),
            "layers_per_target": 4,
            "conditions": ["noop", "ToMe value-only", "MCTF uniform", "MCTF attention-aware"],
            "native_keys_pointers_and_frame_schedule": "preserved",
            "SAM2_weights_and_memory_writes": "unchanged",
            "protocol_sha256": digest(args.protocol),
            "results_sha256": digest(out / "results.json"),
        })
        write(out / "progress.json", {"status": "complete"})
    except Exception as exc:
        write(out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

"""Real-frame engineering gate for video-ToMe cross-frame value sharing.

This gate replays verified native banks and reports no Dice or efficacy claim.
"""

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
from modeling.native_history_cache import cached_read, load_bank
from modeling.spatiotemporal_tome_value_exchange import NativeSpatiotemporalValueExchange
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_attention_distortion_diagnostic import _compare_prompt
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_native_value_consolidation_gate import require_same_prediction, track_read
from scripts.run_regional_refinement import write


def verify_source_native_inputs(audits):
    for layer in range(4):
        noop = audits["noop"][layer]
        for mode in ("local", "cross_frame"):
            treatment = audits[mode][layer]
            if treatment.get("scope") != mode:
                raise RuntimeError(
                    f"{mode} did not execute its declared matching scope at layer {layer}"
                )
            for field in (
                "eligible_frames",
                "eligible_t_pos",
                "spatial_keys_before",
                "spatial_keys_after",
                "spatial_key_mass_sum",
                "key_mass_sum",
                "pointer_keys",
                "frames",
            ):
                if treatment.get(field) != noop.get(field):
                    raise RuntimeError(f"{mode} changed native layout field {field}")
            if sum(frame["target_tokens"] for frame in treatment["frames"]) != treatment["spatial_keys_after"]:
                raise RuntimeError(f"{mode} frame audit does not cover the native spatial slots")
            if treatment["qkv_input_sha256"][1:] != noop["qkv_input_sha256"][1:]:
                raise RuntimeError(f"{mode} changed native K/V input at layer {layer}")
            if layer == 0 and treatment["qkv_input_sha256"][0] != noop["qkv_input_sha256"][0]:
                raise RuntimeError(f"{mode} changed the first-layer query")
            if mode == "cross_frame":
                if len(set(treatment["eligible_t_pos"])) < 2:
                    raise RuntimeError("Cross-frame ToMe needs at least two past memory maps")
                if treatment["cross_frame_groups"] <= 0:
                    raise RuntimeError(
                        f"No cross-frame token pair was formed at layer {layer}"
                    )
            elif treatment["cross_frame_groups"] != 0:
                raise RuntimeError("The local control formed a cross-frame token group")
        if audits["local"][layer]["merged_pairs"] != audits["cross_frame"][layer]["merged_pairs"]:
            raise RuntimeError("Local and cross-frame conditions use different merge counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    checksum = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    if (
        protocol.get("status") != "frozen_for_engineering_gate"
        or digest(args.protocol) != checksum.read_text().strip()
    ):
        raise RuntimeError("Engineering protocol is not frozen or its hash differs")
    out = validate_output(args.output_dir)
    output_root = Path(protocol["output_root"]).resolve()
    if not out.is_relative_to(output_root) or out.exists():
        raise RuntimeError("Use a new child of the frozen Drive output root")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")

    input_path = Path(protocol["input_plan"])
    native_path = Path(protocol["native_plan"])
    if (
        digest(input_path) != protocol["input_plan_sha256"]
        or digest(native_path) != protocol["native_plan_sha256"]
    ):
        raise RuntimeError("Frozen bank/target plan differs")
    inputs = json.loads(input_path.read_text())
    native_plan = json.loads(native_path.read_text())
    config = native_plan["protocol"]
    sam2_repo = ROOT / config["sam2_repository"]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sam2_repo, text=True).strip() != config["sam2_revision"]:
        raise RuntimeError("SAM2 reference revision changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=sam2_repo,
        text=True,
    ).strip():
        raise RuntimeError("SAM2 reference has tracked changes")
    if "sam2" in sys.modules:
        raise RuntimeError("SAM2 was imported before selecting the clean reference")
    if digest(Path(config["checkpoint"])) != config["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed")
    manifest = ROOT / config["manifest"]
    if digest(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Manifest changed")

    sys.path.insert(0, str(sam2_repo))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol["runtime"]["seed"])
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    out.mkdir(parents=True)
    write(out / "setup.json", {"protocol": protocol, "protocol_sha256": digest(args.protocol)})
    results = {}
    try:
        for sequence, targets in protocol["targets"].items():
            verified = inputs["banks"][sequence]
            selected = {row["frame"]: row for row in verified["targets"]}
            if any(frame not in selected for frame in targets):
                raise RuntimeError("Gate target was not input-verified")
            for frame in targets:
                if digest(Path(selected[frame]["image"])) != selected[frame]["image_sha256"]:
                    raise RuntimeError("A verified target image changed")

            cache_dir = Path(verified["cache_directory"])
            if digest(cache_dir / "receipt.json") != verified["receipt_sha256"]:
                raise RuntimeError("Native bank receipt changed")
            receipt = json.loads((cache_dir / "receipt.json").read_text())
            payload = load_bank(
                cache_dir / "bank.pt",
                receipt,
                config["checkpoint_sha256"],
                config["manifest_sha256"],
                config["model_config"],
            )
            payload_before = fingerprint(payload)["sha256"]
            model = build_author_predictor(
                SimpleNamespace(
                    method="sam2",
                    device="cpu",
                    checkpoint=Path(config["checkpoint"]),
                    model_config=config["model_config"],
                ),
                sam2_repo,
            )
            weights = parameter_digest(model)
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
                raise RuntimeError("Native SAM2 evaluation settings differ")

            frames = videos[sequence]
            results[sequence] = []
            with tempfile.TemporaryDirectory(dir=out) as temporary:
                with torch.inference_mode():
                    state, prompt = prepare(model, frames, Path(temporary))
                    model.propagate_in_video_preflight(state)
                    if prompt != payload["prompt"] or len(frames) != payload["num_frames"]:
                        raise RuntimeError("Native prompt/video differs")
                    object_index = state["obj_id_to_idx"][payload["object_id"]]
                    _compare_prompt(
                        state["output_dict_per_obj"][object_index],
                        payload["outputs"],
                        prompt,
                        sequence,
                    )
                    for frame in targets:
                        write(
                            out / "progress.json",
                            {"sequence": sequence, "frame": frame, "status": "running"},
                        )
                        sample = cached_read(model, state, frame, payload)
                        bank_before = fingerprint(sample["output_dict"])["sha256"]
                        native = track_read(model, sample, frame)
                        require_same_prediction(
                            native,
                            {
                                "pred_masks": sample["native_logits"],
                                "object_score_logits": sample["native_object_score"],
                            },
                            "Native cached replay",
                        )

                        arms, audits = {}, {}
                        for mode in ("noop", "local", "cross_frame"):
                            with NativeSpatiotemporalValueExchange(model, mode=mode) as adapter:
                                prediction = track_read(model, sample, frame)
                            if [entry["layer"] for entry in adapter.calls] != [0, 1, 2, 3]:
                                raise RuntimeError("Missing or duplicate SAM2 memory-attention layer")
                            if mode == "noop":
                                require_same_prediction(prediction, native, "Native/no-op parity")
                            if any(not entry["key_unchanged"] for entry in adapter.calls):
                                raise RuntimeError("Treatment modified native K")
                            arms[mode] = {
                                "mask_sha256": _sha256_tensor(prediction["pred_masks"]),
                                "presence_sha256": _sha256_tensor(prediction["object_score_logits"]),
                            }
                            audits[mode] = adapter.calls

                        verify_source_native_inputs(audits)
                        if fingerprint(sample["output_dict"])["sha256"] != bank_before:
                            raise RuntimeError("Read-time value exchange changed the source bank")
                        if parameter_digest(model) != weights or model.num_maskmem != 7:
                            raise RuntimeError("Read-time value exchange changed SAM2 weights or slots")
                        results[sequence].append(
                            {
                                "frame": frame,
                                "object_id": payload["object_id"],
                                "native_replay_exact": True,
                                "noop_parity_exact": True,
                                "source_bank_unchanged": True,
                                "arms": arms,
                                "audits": audits,
                            }
                        )
                        write(out / "results.json", results)
                        del sample, native, prediction, adapter
                    model.reset_state(state)

            if fingerprint(payload)["sha256"] != payload_before or parameter_digest(model) != weights:
                raise RuntimeError("Bank/model changed during the gate")
            del model, state, payload
            gc.collect()

        for name, expected in protocol["source_hashes"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"A frozen source changed during the gate: {name}")
        write(
            out / "completion.json",
            {
                "passed": True,
                "scope": "real-frame engineering only; no Dice inference",
                "targets": sum(map(len, results.values())),
                "layers_per_target": 4,
                "modes": ["noop", "local", "cross_frame"],
                "result_sha256": digest(out / "results.json"),
            },
        )
        write(out / "progress.json", {"status": "complete"})
    except BaseException as exc:
        write(
            out / "failure.json",
            {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()

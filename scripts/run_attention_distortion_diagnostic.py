"""Shadow-evaluate pinned memory-token mergers on archived native SAM2 reads."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.attention_distortion_observer import AttentionDistortionObserver, DEFAULT_CONDITIONS
from modeling.native_history_cache import cached_read, load_bank
from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_author_matching_diagnostic import decode
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write


def _verify_passed(path, field="passed"):
    receipt = json.loads(Path(path).read_text())
    if not receipt.get(field):
        raise RuntimeError(f"Required verified gate failed: {path}")
    return receipt


def _compare_prompt(current, cached, prompt, sequence):
    for field in ("pred_masks", "maskmem_features", "obj_ptr", "object_score_logits"):
        if not torch.equal(current["cond_frame_outputs"][prompt][field],
                           cached["cond_frame_outputs"][prompt][field]):
            raise RuntimeError(f"Prompt memory differs from archived native bank: {sequence}/{field}")


def observer_settings(protocol):
    calibrated = protocol.get("calibration_shadow", False)
    if type(calibrated) is not bool:
        raise RuntimeError("calibration_shadow must be boolean")
    expected = DEFAULT_CONDITIONS[:2] if calibrated else DEFAULT_CONDITIONS
    if protocol.get("conditions") != [dict(condition) for condition in expected]:
        raise RuntimeError("Frozen condition list differs from the tested observer")
    return {"conditions": expected, "calibration_shadow": calibrated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    protocol_path = args.protocol.resolve()
    native_plan_path = args.native_plan.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_for_bounded_diagnostic":
        raise RuntimeError("Diagnostic protocol is not frozen for execution")
    settings = observer_settings(protocol)
    protocol_checksum_path = protocol_path.with_suffix(protocol_path.suffix + ".sha256")
    if digest(protocol_path) != protocol_checksum_path.read_text().strip():
        raise RuntimeError("Frozen protocol checksum differs")
    if digest(native_plan_path) != protocol.get("native_history_plan_sha256"):
        raise RuntimeError("Native-history plan checksum differs")
    output_root = Path(protocol["output_root"]).resolve()
    if out == output_root or not out.is_relative_to(output_root):
        raise RuntimeError("Diagnostic outputs must be a new child of the frozen Drive output root")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {out}")

    started = time.monotonic()
    try:
        source_hashes = protocol["source_hashes"]
        for relative, expected_hash in source_hashes.items():
            if digest(ROOT / relative) != expected_hash:
                raise RuntimeError(f"Frozen diagnostic source changed: {relative}")

        plan = json.loads(native_plan_path.read_text())
        native_protocol = plan["protocol"]
        if native_protocol["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint pin differs from native-history plan")
        if native_protocol["manifest_sha256"] != protocol["manifest_sha256"]:
            raise RuntimeError("Manifest pin differs from native-history plan")
        if digest(Path(native_protocol["checkpoint"])) != protocol["checkpoint_sha256"]:
            raise RuntimeError("Pinned checkpoint file changed")
        manifest = ROOT / native_protocol["manifest"]
        if digest(manifest) != protocol["manifest_sha256"]:
            raise RuntimeError("Pinned PolypGen manifest changed")

        id_gate = _verify_passed(native_protocol["id_gate"])
        replay_gate = json.loads(Path(native_protocol["matching_replay_gate"]).read_text())
        if not replay_gate.get("passed") or not replay_gate.get("treatment_tensors_bit_identical"):
            raise RuntimeError("Independent-process source-matcher replay gate is missing or failed")
        cache_completion = _verify_passed(protocol["cache_gate_receipt"])
        if cache_completion.get("plan_sha256") != digest(native_plan_path):
            raise RuntimeError("Native-cache gate belongs to a different plan")

        sam2_repo = ROOT / native_protocol["sam2_repository"]
        sam2_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=sam2_repo, text=True
        ).strip()
        if sam2_revision != protocol["sam2_revision"]:
            raise RuntimeError("Clean SAM2 diagnostic reference revision differs")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=sam2_repo, text=True,
        ).strip():
            raise RuntimeError("SAM2 diagnostic reference has tracked modifications")
        if "sam2" in sys.modules:
            raise RuntimeError("SAM2 imported before activating the pinned reference")

        pitome_repo = Path(native_protocol["pitome_repository"])
        sys.path.insert(0, str(sam2_repo))
        torch.set_num_threads(2)
        torch.manual_seed(protocol["seed"])
        torch.use_deterministic_algorithms(True)
        videos = {
            **load_manifest(manifest, "dev"),
            **load_manifest(manifest, "test"),
        }
        out.mkdir(parents=True, exist_ok=False)
        write(out / "setup.json", {
            "protocol": protocol,
            "protocol_sha256": digest(protocol_path),
            "native_plan_sha256": digest(native_plan_path),
            "source_hashes": source_hashes,
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "manifest_sha256": protocol["manifest_sha256"],
            "id_gate": id_gate,
            "matcher_replay_gate": replay_gate,
            "cache_gate": cache_completion,
            "measurement_only": True,
            "ground_truth_read": False,
            "shadow_outputs_propagated": False,
        })

        results = {}
        write(out / "results.json", results)
        expected_targets = []
        for sequence in protocol["sequences"]:
            if sequence not in ("seq2", "seq3"):
                raise RuntimeError("Frozen bounded diagnostic supports only seq2 and seq3")
            specification = plan["sequences"][sequence]
            cache_dir = Path(specification["cache_directory"])
            receipt_path = cache_dir / "receipt.json"
            if digest(receipt_path) != specification["receipt_sha256"]:
                raise RuntimeError(f"Native cache receipt changed: {sequence}")
            cache_receipt = json.loads(receipt_path.read_text())
            payload = load_bank(
                cache_dir / "bank.pt", cache_receipt,
                checkpoint_sha256=protocol["checkpoint_sha256"],
                manifest_sha256=protocol["manifest_sha256"],
                model_config=native_protocol["model_config"],
            )
            payload_before = fingerprint(payload)["sha256"]
            frames = videos[sequence]
            if len(frames) != payload["num_frames"]:
                raise RuntimeError(f"Native cache does not cover the full video: {sequence}")
            prompt = payload["prompt"]
            offsets = protocol["target_offsets_from_prompt"]
            targets = [prompt + offset for offset in offsets]
            if len(set(targets)) != len(targets) or max(targets) >= len(frames):
                raise RuntimeError(f"Frozen targets are unavailable or duplicated: {sequence}")
            expected_targets.extend((frame, payload["object_id"]) for frame in targets)

            model = build_author_predictor(SimpleNamespace(
                method="sam2", device=protocol["device"],
                checkpoint=Path(native_protocol["checkpoint"]),
                model_config=native_protocol["model_config"],
            ), sam2_repo)
            if (model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1
                    or model.fill_hole_area != 0):
                raise RuntimeError("SAM2 native frame schedule or postprocessing differs")
            weights_before = parameter_digest(model)
            results[sequence] = []
            folder = out / sequence
            folder.mkdir()

            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                state, current_prompt = prepare(model, frames, Path(temporary))
                model.propagate_in_video_preflight(state)
                if current_prompt != prompt or state["num_frames"] != len(frames):
                    raise RuntimeError("Current prompt/video does not match the archived native bank")
                object_index = state["obj_id_to_idx"][payload["object_id"]]
                original_output_dict = state["output_dict_per_obj"][object_index]
                _compare_prompt(original_output_dict, payload["outputs"], prompt, sequence)

                for frame in targets:
                    write(out / "progress.json", {
                        "phase": "native_shadow_replay",
                        "sequence": sequence,
                        "frame": frame,
                        "completed_targets": sum(map(len, results.values())),
                        "total_targets": len(protocol["sequences"]) * len(offsets),
                        "elapsed_seconds": time.monotonic() - started,
                    })
                    sample = cached_read(model, state, frame, payload)
                    state["output_dict_per_obj"][object_index] = sample["output_dict"]
                    try:
                        with torch.inference_mode():
                            with NativeReadCapture(model) as capture:
                                native = model.track_step(
                                    frame_idx=frame,
                                    is_init_cond_frame=False,
                                    current_vision_feats=sample["vision"],
                                    current_vision_pos_embeds=sample["positions"],
                                    feat_sizes=sample["sizes"],
                                    point_inputs=None,
                                    mask_inputs=None,
                                    output_dict=sample["output_dict"],
                                    num_frames=sample["num_frames"],
                                    track_in_reverse=False,
                                    run_mem_encoder=False,
                                )
                            if not torch.equal(native["pred_masks"], sample["native_logits"]):
                                raise RuntimeError(f"Native mask replay differs: {sequence}/{frame}")
                            if not torch.equal(native["object_score_logits"], sample["native_object_score"]):
                                raise RuntimeError(f"Native presence replay differs: {sequence}/{frame}")
                            _, native_logits = model._get_orig_video_res_output(
                                state, native["pred_masks"]
                            )
                        if capture.layers != [0, 1, 2, 3]:
                            raise RuntimeError("Native capture did not observe every attention layer")

                        roles, spans = memory_role_sequence(
                            model, frame, sample["output_dict"], sample["num_frames"]
                        )
                        bank_before = fingerprint(sample["output_dict"])["sha256"]
                        observer = AttentionDistortionObserver(
                            model, pitome_repository=pitome_repo, **settings,
                        )
                        observer.frame = frame
                        observer.current_object_id = payload["object_id"]
                        observer.memberships = roles
                        observer.spans = spans
                        with torch.inference_mode():
                            with observer:
                                shadow_logits, shadow_presence = decode(model, state, capture)
                        observer.validate_complete([(frame, payload["object_id"])])
                        if not torch.equal(shadow_logits, native_logits):
                            raise RuntimeError(f"Observer changed native decoder mask: {sequence}/{frame}")
                        if not torch.equal(shadow_presence, sample["native_object_score"]):
                            raise RuntimeError(f"Observer changed native presence output: {sequence}/{frame}")
                        if fingerprint(sample["output_dict"])["sha256"] != bank_before:
                            raise RuntimeError(f"Observer changed an archived native bank: {sequence}/{frame}")
                        layer_records = [row for row in observer.records]
                        if [row["layer"] for row in layer_records] != [0, 1, 2, 3]:
                            raise RuntimeError("Observer failed exact once-per-layer coverage")
                        if any(len(row["conditions"]) != len(observer.conditions)
                               for row in layer_records):
                            raise RuntimeError("A shadow condition was missing from a native layer")
                        results[sequence].append({
                            "frame": frame,
                            "object_id": payload["object_id"],
                            "native_mask_sha256": digest_tensor(native_logits),
                            "native_presence_sha256": digest_tensor(sample["native_object_score"]),
                            "native_decoder_parity": True,
                            "native_bank_unchanged": True,
                            "layers": layer_records,
                        })
                        write(out / "results.json", results)
                    finally:
                        state["output_dict_per_obj"][object_index] = original_output_dict
                    del sample, capture, native, observer
                    gc.collect()

                model.reset_state(state)
            if parameter_digest(model) != weights_before:
                raise RuntimeError(f"SAM2 parameters changed: {sequence}")
            if fingerprint(payload)["sha256"] != payload_before:
                raise RuntimeError(f"Archived bank changed: {sequence}")
            if [row["frame"] for row in results[sequence]] != targets:
                raise RuntimeError(f"Target coverage incomplete: {sequence}")
            del model, state, payload
            gc.collect()

        if any(digest(ROOT / relative) != expected
               for relative, expected in source_hashes.items()):
            raise RuntimeError("A frozen source changed during the diagnostic")
        if digest(protocol_path) != protocol_checksum_path.read_text().strip():
            raise RuntimeError("Frozen protocol changed during the diagnostic")
        if digest(native_plan_path) != protocol["native_history_plan_sha256"]:
            raise RuntimeError("Native-history plan changed during the diagnostic")
        write(out / "completion.json", {
            "passed": True,
            "scope": "bounded shadow-attention diagnostic; no efficacy evaluation",
            "targets": len(expected_targets),
            "conditions_per_layer": len(protocol["conditions"]),
            "native_sdpa_outputs_returned": True,
            "native_decoder_and_presence_parity": True,
            "unchanged_parameters_and_banks": True,
            "all_four_layers_once_per_target": True,
            "ground_truth_read": False,
            "elapsed_seconds": time.monotonic() - started,
            "results_sha256": digest(out / "results.json"),
        })
        write(out / "progress.json", {"phase": "completed", "targets": len(expected_targets)})
    except Exception as exc:
        if out.exists():
            write(out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


def digest_tensor(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only, exact-pair attribution audit; not a SAM2 efficacy experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.signed_memory_feature_attribution import SpatialMemoryScalar
from modeling.token_diagnostic_replay import NativeReadCapture, detached_copy
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output


SOURCE_FILES = [
    "protocols/signed_memory_feature_utility_overlap_v1.json",
    "scripts/run_signed_memory_feature_utility_overlap.py",
    "modeling/signed_memory_feature_attribution.py",
    "modeling/token_diagnostic_replay.py",
    "modeling/prediction_conditioned_affinity.py",
    "scripts/run_author_baseline.py",
    "scripts/bank_integrity.py",
    "scripts/memory_token_io.py",
    "datasets/io.py",
    "datasets/video.py",
]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def tensor_sha(value):
    value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def prediction_sha(value):
    array = value.detach().float().cpu().contiguous().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("Nonfinite native prediction")
    return hashlib.sha256(array.tobytes()).hexdigest()


def activate_model(protocol):
    repo = ROOT / protocol["sam2_repository"]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    if revision != protocol["sam2_revision"] or dirty:
        raise RuntimeError("Pinned SAM2 checkout is not at the expected clean revision")
    imported = sys.modules.get("sam2")
    if imported is not None:
        module_path = Path(imported.__file__).resolve()
        if not module_path.is_relative_to(repo.resolve()):
            raise RuntimeError("A different sam2 package was imported first")
    elif str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.run_author_baseline import build_author_predictor

    model = build_author_predictor(
        SimpleNamespace(
            method="sam2",
            device=protocol["device"],
            checkpoint=Path(protocol["checkpoint"]),
            model_config=protocol["model_config"],
        ),
        repo,
    )
    return model, revision


def verify_pins(protocol):
    if digest(Path(protocol["checkpoint"])) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    manifest = ROOT / protocol["manifest"]
    if digest(manifest) != protocol["manifest_sha256"]:
        raise RuntimeError("Manifest hash mismatch")
    utility = Path(protocol["utility_archive"]["root"])
    csv_path = utility / protocol["utility_archive"]["pairs_csv"]
    if digest(csv_path) != protocol["utility_archive"]["pairs_csv_sha256"]:
        raise RuntimeError("Prior utility-pair CSV differs from the frozen pin")
    for sequence, expected in protocol["utility_archive"]["sequence_result_sha256"].items():
        result_path = utility / "scientific_run_20261108" / "sequences" / sequence / "results.json"
        if digest(result_path) != expected:
            raise RuntimeError(f"Prior utility receipt hash mismatch for {sequence}")
    return manifest, utility


def capture_one(protocol, pair, videos, output):
    frames = videos[pair["sequence"]]
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    target = pair["target_frame"]
    if target <= prompt or target >= len(frames):
        raise RuntimeError(f"Invalid target/prompt timing for {pair['sequence']}")

    model, revision = activate_model(protocol)
    before_parameters = parameter_digest(model)
    sample_folder = output / "captures" / pair["sequence"]
    sample_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="staged_", dir=sample_folder) as staged:
        stage_frames(frames, Path(staged))
        state = model.init_state(
            video_path=staged,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        prompt_mask = load_label_mask(frames[prompt].mask_path, [1]) == 1
        model.add_new_mask(
            state,
            frame_idx=prompt,
            obj_id=1,
            mask=torch.from_numpy(prompt_mask.copy()),
        )
        trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
        captured = None
        for expected_frame in range(prompt, target + 1):
            with NativeReadCapture(model) as native:
                frame, object_ids, logits = next(trajectory)
            if frame != expected_frame or object_ids != [1]:
                raise RuntimeError("Native trajectory order or persistent object ID changed")
            write(output / "progress.json", {
                "phase": "native_capture",
                "sequence": pair["sequence"],
                "frame": frame,
                "target": target,
            })
            if frame == target:
                native_prediction_sha = prediction_sha(logits)
                if native_prediction_sha != pair["expected_native_prediction_sha256"]:
                    raise RuntimeError(
                        f"Archived native prediction mismatch {pair['sequence']}/{target}: {native_prediction_sha}"
                    )
                bank = state["output_dict_per_obj"][0]
                bank_before = fingerprint(bank)["sha256"]
                _, spans = memory_role_sequence(
                    model, target, bank, len(frames), mode="global"
                )
                span_sources = [span["frame"] for span in spans]
                if span_sources != pair["expected_memory_sources"]:
                    raise RuntimeError(
                        f"Native memory source order differs for {pair['sequence']}/{target}: {span_sources}"
                    )
                if pair["source_frame"] not in span_sources:
                    raise RuntimeError("Utility source frame is absent from the native memory read")

                memory = native.memory["memory"]
                pointer_count = native.memory["num_obj_ptr_tokens"]
                spatial_count = memory.shape[0] - pointer_count
                if spans[-1]["stop"] != spatial_count:
                    raise RuntimeError("Spatial memory spans do not cover spatial K/V exactly")
                for span in spans:
                    stored = bank["cond_frame_outputs"].get(span["frame"])
                    if stored is None:
                        stored = bank["non_cond_frame_outputs"].get(span["frame"])
                    if stored is None:
                        raise RuntimeError("A selected source span is absent from the native bank")
                    features = memory[span["start"]:span["stop"], 0]
                    expected = stored["maskmem_features"][0].flatten(1).T
                    if not torch.equal(features, expected):
                        raise RuntimeError("Captured spatial token content differs from its bank source")
                    span["grid"] = list(stored["maskmem_features"].shape[-2:])
                    span["feature_sha256"] = tensor_sha(features)
                bank_after = fingerprint(bank)["sha256"]
                if bank_before != bank_after:
                    raise RuntimeError("Read-only source indexing mutated the native memory bank")
                if native.layers != [0, 1, 2, 3]:
                    raise RuntimeError("Native read did not exercise all four attention layers")
                if memory.shape[1] != 1:
                    raise RuntimeError("This bounded diagnostic expects one persistent object")
                captured = {
                    "memory": cpu_tree(native.memory),
                    "decoder": cpu_tree(native.decoder),
                    "outputs": cpu_tree(native.outputs),
                    "layers": list(native.layers),
                    "metadata": {
                        "sequence": pair["sequence"],
                        "target_frame": target,
                        "source_frame": pair["source_frame"],
                        "prompt_frame": prompt,
                        "object_id": 1,
                        "native_prediction_sha256": native_prediction_sha,
                        "memory_bank_sha256_at_target": bank_before,
                        "memory_sources": spans,
                        "pointer_count": int(pointer_count),
                        "pointer_content_sha256": tensor_sha(memory[spatial_count:]),
                        "pointer_positions_sha256": tensor_sha(native.memory["memory_pos"][-pointer_count:]) if pointer_count else None,
                        "memory_input_sha256": tensor_sha(memory),
                        "native_process_pid": os.getpid(),
                        "sam2_revision": revision,
                    },
                }
        trajectory.close()
        model.reset_state(state)
    if captured is None:
        raise RuntimeError("Requested target frame was not captured")
    if parameter_digest(model) != before_parameters or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("SAM2 parameters changed or accumulated gradients during capture")

    capture_path = output / "captures" / f"{pair['sequence']}_target_{target}.pt"
    temporary = capture_path.with_suffix(".pt.tmp")
    torch.save(captured, temporary)
    checked = torch.load(temporary, map_location="cpu", weights_only=True)
    if checked["metadata"] != captured["metadata"]:
        raise RuntimeError("Safe capture serialization changed metadata")
    os.replace(temporary, capture_path)
    return {
        **captured["metadata"],
        "capture_path": str(capture_path),
        "capture_sha256": digest(capture_path),
        "sam2_parameter_sha256": before_parameters,
    }


def scalar_forward_with_layer_check(scalar, call):
    observed = []
    handles = []
    for index, layer in enumerate(scalar.model.memory_attention.layers):
        handles.append(layer.register_forward_hook(lambda module, args, out, i=index: observed.append(i)))
    try:
        result = call()
    finally:
        for handle in handles:
            handle.remove()
    if observed != [0, 1, 2, 3]:
        raise RuntimeError(f"Replay did not execute all attention layers in order: {observed}")
    return result


def attribution_tiles(attribution, span, grid):
    height, width = grid
    if height != 64 or width != 64 or span["stop"] - span["start"] != height * width:
        raise RuntimeError("Expected a 64x64 SAM2 spatial source map")
    values = attribution[0, span["start"]:span["stop"]]
    signed_map = values.sum(dim=-1).reshape(height, width)
    absolute_map = values.abs().sum(dim=-1).reshape(height, width)
    tile_scores = {}
    for tile_id in range(16):
        row, col = divmod(tile_id, 4)
        ys, xs = slice(row * 16, (row + 1) * 16), slice(col * 16, (col + 1) * 16)
        tile_scores[f"tile_{tile_id:02d}"] = {
            "signed_input_x_gradient_sum": float(signed_map[ys, xs].sum()),
            "absolute_input_x_gradient_sum": float(absolute_map[ys, xs].sum()),
        }
    return signed_map.numpy(), absolute_map.numpy(), tile_scores


def run_finite_differences(scalar, pair, spans, attributions, config):
    source = next(span for span in spans if span["frame"] == pair["source_frame"])
    height, width = source["grid"]
    scores = []
    for target_name in config["scalars"]:
        if target_name == "native_mask_support" and not scalar.support.any():
            continue
        attribution = attributions[target_name]
        for tile_id in config["finite_difference_tiles"]:
            tile_row, tile_col = divmod(tile_id, 4)
            y0, y1 = tile_row * 16, (tile_row + 1) * 16
            x0, x1 = tile_col * 16, (tile_col + 1) * 16
            local = torch.arange(height * width).reshape(height, width)[y0:y1, x0:x1].flatten()
            token_ids = local + source["start"]
            predicted = float(attribution[0, token_ids].sum())
            for epsilon in config["finite_difference_epsilons"]:
                plus = scalar.inputs.clone()
                minus = scalar.inputs.clone()
                plus[:, token_ids] *= 1.0 + epsilon
                minus[:, token_ids] *= 1.0 - epsilon
                with torch.no_grad():
                    fp = scalar(plus, target_name)
                    fm = scalar(minus, target_name)
                observed = float((fp - fm).item() / (2.0 * epsilon))
                error = abs(observed - predicted)
                denominator = max(abs(observed), abs(predicted), 1e-8)
                scores.append({
                    "target": target_name,
                    "source_frame": pair["source_frame"],
                    "tile_id": tile_id,
                    "epsilon": epsilon,
                    "input_x_gradient_sum": predicted,
                    "symmetric_finite_difference": observed,
                    "absolute_error": error,
                    "normalized_error": error / denominator,
                    "status": "measured_flat" if max(abs(observed), abs(predicted)) < 1e-8 else "measured",
                })
    return scores


def replay_one(protocol, capture_path, output, finite_difference):
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    metadata = capture["metadata"]
    pair = next(p for p in protocol["pairs"] if p["sequence"] == metadata["sequence"] and p["target_frame"] == metadata["target_frame"])
    model, revision = activate_model(protocol)
    before_parameters = parameter_digest(model)
    scalar = SpatialMemoryScalar(model, SimpleNamespace(
        memory=capture["memory"], decoder=capture["decoder"], outputs=capture["outputs"], layers=capture["layers"]
    ))
    memory_before = tensor_sha(scalar.memory["memory"])
    pointer_before = tensor_sha(scalar.pointer_tail)

    native_outputs = scalar_forward_with_layer_check(
        scalar, lambda: scalar.raw_outputs(scalar.inputs)
    )
    for expected, actual in zip(capture["outputs"], native_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    attributions = {}
    output_scalars = {}
    for target_name in protocol["attribution"]["scalars"]:
        if target_name == "native_mask_support" and not scalar.support.any():
            output_scalars[target_name] = {"status": "unavailable_empty_native_support"}
            continue
        attribution = scalar_forward_with_layer_check(
            scalar, lambda target=target_name: scalar.attribute(
                protocol["attribution"]["captum_root"], target
            )
        )
        if not torch.isfinite(attribution).all():
            raise RuntimeError("Attribution contains nonfinite values")
        attributions[target_name] = attribution
        output_scalars[target_name] = {"status": "available", "shape": list(attribution.shape)}

    per_source = []
    arrays = {}
    for span in metadata["memory_sources"]:
        source_result = {"frame": span["frame"], "t_pos": span["t_pos"], "start": span["start"], "stop": span["stop"]}
        for target_name, attribution in attributions.items():
            signed_map, absolute_map, tiles = attribution_tiles(attribution, span, span["grid"])
            arrays[f"{target_name}_source_{span['frame']}_signed"] = signed_map
            arrays[f"{target_name}_source_{span['frame']}_absolute"] = absolute_map
            source_result[target_name] = {"tiles": tiles, "total_signed": float(attribution[0, span["start"]:span["stop"]].sum())}
        per_source.append(source_result)

    finite_differences = run_finite_differences(
        scalar, pair, metadata["memory_sources"], attributions,
        {**protocol["attribution"], "finite_difference_tiles": protocol["attribution"]["finite_difference_tiles"]}
    ) if finite_difference else []

    if memory_before != tensor_sha(scalar.memory["memory"]) or pointer_before != tensor_sha(scalar.pointer_tail):
        raise RuntimeError("Replay changed serialized spatial memory or pointer tail")
    if before_parameters != parameter_digest(model) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Replay changed frozen SAM2 weights or accumulated parameter gradients")
    arrays["memory_sources_json"] = np.asarray(json.dumps(metadata["memory_sources"]))
    archive = output / "attributions" / f"{metadata['sequence']}_target_{metadata['target_frame']}.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **arrays)
    result = {
        "sequence": metadata["sequence"],
        "source_frame": pair["source_frame"],
        "target_frame": pair["target_frame"],
        "native_prediction_sha256": metadata["native_prediction_sha256"],
        "native_replay_max_abs_error": max(float((a - e).abs().max()) for a, e in zip(native_outputs, capture["outputs"], strict=True)),
        "attention_layers_per_checked_forward": [0, 1, 2, 3],
        "object_presence_and_mask_support": output_scalars,
        "per_source_signed_tile_scores": per_source,
        "finite_differences": finite_differences,
        "attribution_archive": str(archive),
        "attribution_sha256": digest(archive),
        "replay_process_pid": os.getpid(),
        "sam2_revision": revision,
        "parameters_unchanged": True,
        "memory_and_pointers_unchanged": True,
    }
    del model
    return result


def utility_alignment(protocol, results):
    """Compare GT-free attribution ranks with already-saved tile Dice effects."""
    from scipy.stats import spearmanr

    utility_root = Path(protocol["utility_archive"]["root"])
    rows = []
    for result in results:
        sequence = result["sequence"]
        target = result["target_frame"]
        source = result["source_frame"]
        archived_path = utility_root / "scientific_run_20261108" / "sequences" / sequence / "results.json"
        archived = json.loads(archived_path.read_text())
        record = next((row for row in archived["main"]
                       if row.get("anchor_target") == target and row.get("source_frame") == source), None)
        if record is None:
            raise RuntimeError(f"No exact saved source/target utility record for {sequence}/{source}->{target}")
        arms = record["arms"]
        if arms["native"].get("prediction_sha256") != result["native_prediction_sha256"]:
            raise RuntimeError("Saved utility native prediction differs from this exact replay")
        if not all(f"tile_{tile:02d}" in arms for tile in range(16)):
            raise RuntimeError(f"Incomplete 16-tile utility intervention for {sequence}/{target}")
        native_dice = float(arms["native"]["scores"]["dice"])
        tile_utility = []
        for tile in range(16):
            arm = arms[f"tile_{tile:02d}"]
            read = arm.get("read", {})
            selected_sources = [layer.get("source_frame") for layer in read.get("selection_layers", [])]
            if arm.get("source_frame_index") != target or read.get("frame") != target:
                raise RuntimeError("Saved tile arm is indexed at a different target frame")
            if selected_sources != [source, source, source, source]:
                raise RuntimeError("Saved tile arm altered a different source frame")
            tile_utility.append(native_dice - float(arm["scores"]["dice"]))
        source_row = next(row for row in result["per_source_signed_tile_scores"] if row["frame"] == source)
        row_result = {"sequence": sequence, "source_frame": source, "target_frame": target,
                      "saved_native_dice": native_dice,
                      "utility_definition": "native Dice - tile-removed Dice; positive means this source tile helped under the saved deletion intervention",
                      "tile_removal_utility": tile_utility,
                      "descriptive_only": True}
        for signal in ("native_mask_support", "object_presence"):
            if signal not in source_row:
                row_result[signal] = {"status": "unavailable"}
                continue
            attribution = [source_row[signal]["tiles"][f"tile_{tile:02d}"]["signed_input_x_gradient_sum"] for tile in range(16)]
            absolute = [source_row[signal]["tiles"][f"tile_{tile:02d}"]["absolute_input_x_gradient_sum"] for tile in range(16)]
            rho_signed_value = float(spearmanr(attribution, tile_utility).statistic)
            rho_absolute_value = float(spearmanr(absolute, tile_utility).statistic)
            rho_signed = rho_signed_value if np.isfinite(rho_signed_value) else None
            rho_absolute = rho_absolute_value if np.isfinite(rho_absolute_value) else None
            row_result[signal] = {"signed_spearman_rho": rho_signed,
                                  "absolute_spearman_rho": rho_absolute,
                                  "input_x_gradient_tile_sums": attribution,
                                  "absolute_input_x_gradient_tile_sums": absolute}
        rows.append(row_result)
    return {"status": "exploratory_exact_pair_alignment",
            "pairs": rows,
            "interpretation_limit": "Four post-hoc selected pairs; 16 spatial tiles per pair are dependent, and feature scaling sensitivity is not the same intervention as deleting a key/value entry. Correlations are descriptive and do not establish useful-token prediction or Dice efficacy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/signed_memory_feature_utility_overlap_v1.json")
    parser.add_argument("--phase", choices=("all", "capture", "replay"), default="all")
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    if args.phase == "all":
        output.mkdir(parents=True, exist_ok=False)
        verify_pins(protocol)
        source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        write(output / "setup.json", {
            "protocol_sha256": digest(args.protocol),
            "source_hashes": source_hashes,
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "manifest_sha256": protocol["manifest_sha256"],
            "utility_pairs_csv_sha256": protocol["utility_archive"]["pairs_csv_sha256"],
            "started_unix": time.time(),
            "capture_process_pid": os.getpid(),
            "scope": protocol["scope"],
        })
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"), **load_manifest(ROOT / protocol["manifest"], "test")}
        observed_captures = []
        for pair in protocol["pairs"]:
            observed_captures.append(capture_one(protocol, pair, videos, output))
        write(output / "capture_receipts.json", observed_captures)
        if source_hashes != {name: digest(ROOT / name) for name in SOURCE_FILES}:
            raise RuntimeError("A frozen source changed during capture")
        log = (output / "replay_process.log").open("w")
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--phase", "replay",
            "--output-dir", str(output), "--protocol", str(args.protocol),
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
        log.close()
        if source_hashes != {name: digest(ROOT / name) for name in SOURCE_FILES}:
            raise RuntimeError("A frozen source changed during replay")
        results = json.loads((output / "replay_results.json").read_text())
        alignment = utility_alignment(protocol, results)
        write(output / "utility_alignment.json", alignment)
        write(output / "completion.json", {
            "engineering_passed": True,
            "independent_process_replay": True,
            "exact_native_pair_predictions": len(observed_captures),
            "attribution_pairs": len(results),
            "efficacy_evaluated": False,
            "dice_gain_claim": False,
            "finished_unix": time.time(),
        })
        write(output / "progress.json", {"phase": "complete"})
    elif args.phase == "capture":
        raise RuntimeError("Capture-only mode is intentionally disabled; use --phase all")
    else:
        verify_pins(protocol)
        setup = json.loads((output / "setup.json").read_text())
        if digest(args.protocol) != setup["protocol_sha256"]:
            raise RuntimeError("Replay protocol differs from captured protocol")
        if setup["source_hashes"] != {name: digest(ROOT / name) for name in SOURCE_FILES}:
            raise RuntimeError("Replay source hashes differ from the capture process")
        if os.getpid() == setup["capture_process_pid"]:
            raise RuntimeError("Replay was not started in an independent process")
        receipts = json.loads((output / "capture_receipts.json").read_text())
        results = []
        for index, receipt in enumerate(receipts):
            pair = next(p for p in protocol["pairs"] if p["sequence"] == receipt["sequence"])
            use_fd = f"{pair['sequence']}:{pair['source_frame']}->{pair['target_frame']}" in protocol["attribution"]["finite_difference_pairs"]
            results.append(replay_one(protocol, Path(receipt["capture_path"]), output, use_fd))
            write(output / "progress.json", {"phase": "independent_replay", "completed": index + 1, "total": len(receipts), "sequence": pair["sequence"]})
            write(output / "replay_results.json", results)
        if len(results) != len(protocol["pairs"]):
            raise RuntimeError("Independent replay coverage is incomplete")
        write(output / "utility_alignment.json", utility_alignment(protocol, results))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--output-dir" in sys.argv:
            destination = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            if destination.exists():
                write(destination / "failure.json", {"type": type(error).__name__, "error": str(error), "pid": os.getpid()})
        raise

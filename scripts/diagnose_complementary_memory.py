"""Run the frozen offline, sequence-separated memory-feature probe.

This script reads archived native SAM2 banks, fits scikit-learn ridge probes,
and scores occupancy on held-out sequences. It performs no SAM2 inference.
All generated files must be under the configured Google Drive output tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/complementary_memory_diagnostic_v1.json"
MANIFEST_PATH = ROOT / "data/polypgen_sequence.jsonl"
EXPERIMENT_NAME = "166_complementary_memory_diagnostic_v1"

sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def drive_root(output_directory: Path) -> Path:
    matches = list((Path.home() / "Library/CloudStorage").glob("GoogleDrive-*/My Drive/AdeSEG"))
    matches = [path.resolve() for path in matches if path.is_dir()]
    requested = output_directory.expanduser().resolve()
    matches = [path for path in matches
               if requested.is_relative_to((path / "outputs/experiment").resolve())]
    if len(matches) != 1:
        raise RuntimeError(
            "The requested output directory must resolve beneath exactly one synced "
            f"AdeSEG/outputs/experiment root; found {len(matches)}"
        )
    return matches[0]


def verify_frozen_protocol(protocol: dict) -> tuple[str, str]:
    protocol_hash = digest(PROTOCOL_PATH)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    committed = subprocess.run(
        ["git", "show", f"{head}:protocols/complementary_memory_diagnostic_v1.json"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != protocol_hash:
        raise RuntimeError("Working diagnostic protocol differs from the frozen Git commit")
    return head, protocol_hash


def check_output_path(path: Path, drive: Path) -> Path:
    resolved = path.expanduser().resolve()
    required = (drive / "outputs/experiment").resolve()
    if not resolved.is_relative_to(required) or resolved.name != EXPERIMENT_NAME:
        raise ValueError(f"Output must be Drive/outputs/experiment/{EXPERIMENT_NAME}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to reuse existing experiment directory: {resolved}")
    return resolved


def load_manifest_sequences() -> tuple[dict, dict]:
    dev = load_manifest(MANIFEST_PATH, "dev")
    test = load_manifest(MANIFEST_PATH, "test")
    if set(dev) & set(test):
        raise RuntimeError("A sequence appears in both manifest splits")
    return {**dev, **test}, {"dev": dev, "test": test}


def _bank_outputs(bank: dict) -> dict[int, dict]:
    output = bank["outputs"]
    merged = dict(output["cond_frame_outputs"])
    merged.update(output["non_cond_frame_outputs"])
    return {int(index): value for index, value in merged.items()}


def _area_target(mask_path: Path) -> np.ndarray:
    """Map binary GT to fractional occupancy on the 64x64 square SAM2 grid."""
    mask = load_label_mask(mask_path, [1]).astype(np.float32, copy=False)
    image = Image.fromarray(mask)
    return np.asarray(
        image.resize((64, 64), resample=Image.Resampling.BOX), dtype=np.float32
    ).reshape(4096)


def _stored_inputs(output: dict, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = output["maskmem_features"]
    logits = output["pred_masks"]
    if tuple(features.shape) != (1, 64, 64, 64) or tuple(logits.shape) != (1, 1, 256, 256):
        raise RuntimeError("Unexpected native memory feature or stored-mask shape")
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite native bank tensor")
    feature_grid = features[0].float().permute(1, 2, 0).contiguous().numpy()
    probability = F.avg_pool2d(logits.float().sigmoid(), kernel_size=4, stride=4)
    probability = probability[0, 0].numpy()
    target = _area_target(mask_path).reshape(64, 64)
    return feature_grid, probability, target


def verify_archive(protocol: dict, drive: Path, videos: dict) -> dict:
    base = drive / "outputs/experiment/143_official_native_feature_audit_v1"
    full = base / "full"
    source = full / "completion.json"
    files = {
        "setup.json": full / "setup.json",
        "completion.json": source,
        "independent_audit.json": full / "independent_audit.json",
        "report_verification.json": full / "report_verification.json",
    }
    expected = {
        "setup.json": protocol["inputs"]["archive_setup_sha256"],
        "completion.json": protocol["inputs"]["archive_completion_sha256"],
        "independent_audit.json": protocol["inputs"]["archive_independent_audit_sha256"],
        "report_verification.json": protocol["inputs"]["archive_report_verification_sha256"],
    }
    for name, path in files.items():
        if not path.is_file() or digest(path) != expected[name]:
            raise RuntimeError(f"Archived {name} is absent or differs from the frozen digest")
    if digest(MANIFEST_PATH) != protocol["inputs"]["manifest_sha256"]:
        raise RuntimeError("PolypGen manifest differs from the frozen protocol")
    checkpoint = drive / "outputs/experiment/assets/sam2.1_hiera_tiny.pt"
    if not checkpoint.is_file() or digest(checkpoint) != protocol["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("Pinned native checkpoint is absent or changed")

    completion = json.loads(source.read_text())
    independent = json.loads(files["independent_audit.json"].read_text())
    archive_setup = json.loads(files["setup.json"].read_text())
    if archive_setup.get("manifest_sha256") != protocol["inputs"]["manifest_sha256"]:
        raise RuntimeError("Archived capture manifest hash differs")
    if archive_setup.get("checkpoint_sha256") != protocol["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("Archived capture checkpoint differs")
    if archive_setup.get("sam2_revision") not in (None, protocol["inputs"]["sam2_revision"]):
        raise RuntimeError("Archived SAM2 revision differs")
    allowed_drift = set(protocol["inputs"]["archive_source_audit"][
        "permitted_post_capture_support_only_changes"])
    source_drift = {}
    archived_sources = archive_setup.get("sources", {})
    if len(archived_sources) != protocol["inputs"]["archive_source_audit"]["source_files_in_archive_setup"]:
        raise RuntimeError("Archived source inventory size differs")
    for relative, expected_hash in archived_sources.items():
        current_path = ROOT / relative
        if not current_path.is_file():
            raise RuntimeError(f"Archived source file is missing: {relative}")
        current_hash = digest(current_path)
        if current_hash != expected_hash:
            if relative not in allowed_drift:
                raise RuntimeError(f"Runtime/source provenance changed: {relative}")
            source_drift[relative] = {"archived": expected_hash, "current": current_hash}
    if not completion.get("passed") or completion.get("scope") != "cohort":
        raise RuntimeError("A complete native cohort bank is required")
    if not independent.get("passed") or independent.get("total_targets") != 2012:
        raise RuntimeError("Independent cohort audit is incomplete")
    if completion.get("checkpoint_sha256") != protocol["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("Cohort completion checkpoint differs")
    # The cohort-level completion receipt records the checkpoint but not the
    # manifest. The manifest digest is pinned in setup.json and every
    # per-sequence receipt above, so validate provenance at those sources.
    if external_revision() != protocol["inputs"]["sam2_revision"]:
        raise RuntimeError("Working SAM2 submodule revision differs")
    if set(completion["entries"]) != set(protocol["cohort"]["eligible_sequences"]):
        raise RuntimeError("Native bank sequence coverage differs from protocol")

    bank_receipts = {}
    experiment_root = base.resolve()
    targets = 0
    for split_name in ("fit_split", "evaluation_split"):
        split = protocol["cohort"][split_name]
        split_targets = 0
        for name in split["sequences"]:
            if name not in videos:
                raise RuntimeError(f"Sequence missing from manifest: {name}")
            folder = Path(completion["entries"][name]).resolve()
            if not folder.is_relative_to(experiment_root):
                raise RuntimeError(f"Bank path outside its Drive experiment archive: {name}")
            receipt_path, bank_path = folder / "receipt.json", folder / "bank.pt"
            seq_audit = independent["sequences"].get(name)
            seq_complete = json.loads((folder / "completion.json").read_text())
            if (seq_audit is None or not seq_audit.get("passed")
                    or digest(receipt_path) != seq_audit["receipt_sha256"]
                    or digest(folder / "completion.json") != seq_audit["completion_sha256"]):
                raise RuntimeError(f"Per-sequence receipt mismatch: {name}")
            receipt = json.loads(receipt_path.read_text())
            bank_sha = digest(bank_path)
            if bank_sha != receipt["bank_sha256"]:
                raise RuntimeError(f"Native bank bytes differ from receipt: {name}")
            if receipt.get("checkpoint_sha256") != protocol["inputs"]["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint provenance mismatch: {name}")
            if receipt.get("manifest_sha256") != protocol["inputs"]["manifest_sha256"]:
                raise RuntimeError(f"Manifest provenance mismatch: {name}")
            frames = videos[name]
            prompt = int(receipt["prompt"])
            if receipt["num_frames"] != len(frames) or seq_complete["targets"] != len(frames) - prompt - 1:
                raise RuntimeError(f"Manifest / native frame count mismatch: {name}")

            # Validate every native frame and its actual GT file before any fitting.
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
            outputs = _bank_outputs(bank)
            expected_indices = set(range(prompt, len(frames)))
            if set(outputs) != expected_indices:
                raise RuntimeError(f"Missing or extra native memory frames: {name}")
            if bank.get("object_id") != receipt.get("object_id"):
                raise RuntimeError(f"Object ID differs from native receipt: {name}")
            for index in range(prompt + 1, len(frames)):
                _stored_inputs(outputs[index], frames[index].mask_path)
            n_targets = len(frames) - prompt - 1
            split_targets += n_targets
            targets += n_targets
            bank_receipts[name] = {
                "receipt_sha256": digest(receipt_path),
                "bank_sha256": bank_sha,
                "completion_sha256": digest(folder / "completion.json"),
                "prompt_index": prompt,
                "post_prompt_frames": n_targets,
                "bank_relative_path": str(folder.relative_to(experiment_root)),
            }
            del bank, outputs
            gc.collect()
        if split_targets != split["post_prompt_frames"]:
            raise RuntimeError(f"Frozen frame count mismatch for {split_name}")
    if targets != protocol["cohort"]["total_post_prompt_frames"]:
        raise RuntimeError("Total frame coverage differs from frozen protocol")
    return {
        "archive_completion_sha256": digest(source),
        "archive_independent_audit_sha256": digest(files["independent_audit.json"]),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(MANIFEST_PATH),
        "bank_receipts": bank_receipts,
        "permitted_support_source_drift": source_drift,
        "post_prompt_frames": targets,
    }


def external_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT / "external/sam2"), "rev-parse", "HEAD"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout.strip()


def _read_sequence(videos: dict, completion: dict, name: str,
                   expected_bank_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    folder = Path(completion["entries"][name])
    receipt = json.loads((folder / "receipt.json").read_text())
    bank_path = folder / "bank.pt"
    if digest(bank_path) != expected_bank_sha256:
        raise RuntimeError(f"Native bank changed after preflight: {name}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    outputs = _bank_outputs(bank)
    prompt = int(receipt["prompt"])
    frames = videos[name]
    count = len(frames) - prompt - 1
    features = np.empty((count, 4096, 64), dtype=np.float32)
    probabilities = np.empty((count, 4096), dtype=np.float32)
    targets = np.empty((count, 4096), dtype=np.float32)
    frame_ids = []
    for row, index in enumerate(range(prompt + 1, len(frames))):
        feature, probability, target = _stored_inputs(outputs[index], frames[index].mask_path)
        features[row] = feature.reshape(4096, 64)
        probabilities[row] = probability.reshape(4096)
        targets[row] = target.reshape(4096)
        frame_ids.append(index)
    del bank, outputs
    return features, probabilities, targets, frame_ids


def _fit(condition: str, features: np.ndarray, probabilities: np.ndarray,
         targets: np.ndarray, weights: np.ndarray) -> tuple[StandardScaler, Ridge]:
    if condition == "mask_only":
        matrix = probabilities[:, None].copy()
    elif condition == "feature_only":
        matrix = features.copy()
    elif condition == "combined":
        matrix = np.empty((features.shape[0], 65), dtype=np.float32)
        matrix[:, :64] = features
        matrix[:, 64] = probabilities
    else:
        raise ValueError(condition)
    scaler = StandardScaler(copy=False).fit(matrix, sample_weight=weights)
    scaler.transform(matrix, copy=False)
    probe = Ridge(alpha=1.0, fit_intercept=True, copy_X=False, solver="cholesky")
    probe.fit(matrix, targets, sample_weight=weights)
    return scaler, probe


def _bounded_predict(scaler: StandardScaler, probe: Ridge, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    scaler.transform(matrix, copy=False)
    return np.clip(probe.predict(matrix).astype(np.float32), 0.0, 1.0)


def _frame_scores(prediction: np.ndarray, target: np.ndarray,
                  mask_probability: np.ndarray) -> dict:
    squared = np.square(prediction - target)
    residual_weight = np.abs(mask_probability - target)
    return {
        "brier": float(squared.mean()),
        "mae": float(np.abs(prediction - target).mean()),
        "baseline_error_weighted_brier": (
            float(np.average(squared, weights=residual_weight))
            if residual_weight.sum() > 0 else None
        ),
        "target_present": bool(target.sum() > 0),
    }


def _sequence_scores(frame_metrics: list[dict]) -> dict:
    result = {"frames": len(frame_metrics)}
    for stratum, select in (
        ("all", lambda row: True),
        ("present", lambda row: row["target_present"]),
        ("empty", lambda row: not row["target_present"]),
    ):
        rows = [row for row in frame_metrics if select(row)]
        result[f"{stratum}_frames"] = len(rows)
        for metric in ("brier", "mae", "baseline_error_weighted_brier"):
            vals = [row[metric] for row in rows if row[metric] is not None]
            result[f"{stratum}_{metric}"] = float(np.mean(vals)) if vals else None
    return result


def _bootstrap(values: np.ndarray, seed: int, n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n, len(values)))
    means = values[draws].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def run(args: argparse.Namespace) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    commit, protocol_hash = verify_frozen_protocol(protocol)
    drive = drive_root(args.output_dir)
    output = check_output_path(args.output_dir, drive)
    if sys.version.split()[0] != protocol["probe"]["environment"]["python"]:
        raise RuntimeError("Python version differs from frozen diagnostic environment")
    actual_versions = {
        "python": sys.version.split()[0],
        "scikit-learn": __import__("sklearn").__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__.split("+")[0],
        "Pillow": __import__("PIL").__version__,
        "joblib": __import__("joblib").__version__,
        "threadpoolctl": __import__("threadpoolctl").__version__,
        "narwhals": __import__("narwhals").__version__,
    }
    if actual_versions != protocol["probe"]["environment"]:
        raise RuntimeError(f"Probe package versions differ from frozen protocol: {actual_versions}")

    videos, manifest_splits = load_manifest_sequences()
    eligible = set(protocol["cohort"]["eligible_sequences"])
    excluded = set(protocol["cohort"]["excluded_sequences"])
    if eligible & excluded:
        raise RuntimeError("A sequence is both eligible and excluded")
    for split_name in ("fit_split", "evaluation_split"):
        record = protocol["cohort"][split_name]
        expected_sequences = set(manifest_splits[record["manifest_split"]]) & eligible
        if set(record["sequences"]) != expected_sequences:
            raise RuntimeError(f"Frozen sequence list differs from {record['manifest_split']} manifest")
    input_receipt = verify_archive(protocol, drive, videos)
    if args.validate_only:
        print(json.dumps({"validated": True, "sequences": len(input_receipt["bank_receipts"]),
                          "post_prompt_frames": input_receipt["post_prompt_frames"],
                          "output_directory": str(output), "protocol_commit": commit}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, *sys.argv]
    setup = {
        "created_at": timestamp(), "protocol_commit": commit,
        "protocol_sha256": protocol_hash,
        "script_sha256": digest(Path(__file__)),
        "git_head": commit, "command": command,
        "drive_output_directory": str(output), "versions": actual_versions,
        "input_receipt": input_receipt,
        "fit_sequences": protocol["cohort"]["fit_split"]["sequences"],
        "evaluation_sequences": protocol["cohort"]["evaluation_split"]["sequences"],
        "inference_performed": False,
    }
    atomic_json(output / "setup.json", setup)
    status = {"phase": "loading_fit_split", "started_at": timestamp(),
              "completed_sequences": 0,
              "expected_sequences": len(protocol["cohort"]["fit_split"]["sequences"]),
              "completed_frames": 0,
              "expected_frames": protocol["cohort"]["fit_split"]["post_prompt_frames"],
              "efficacy_run": False}
    atomic_json(output / "status.json", status)
    log_path = output / "fit_log.txt"

    fit_names = protocol["cohort"]["fit_split"]["sequences"]
    fit_frames = protocol["cohort"]["fit_split"]["post_prompt_frames"]
    n = fit_frames * 4096
    x_features = np.empty((n, 64), dtype=np.float32)
    x_probability = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    sample_weight = np.empty(n, dtype=np.float32)
    completion = json.loads((drive / protocol["inputs"]["native_archive"]).read_text())
    offset, frame_offset = 0, 0
    with log_path.open("w", encoding="utf-8") as log, threadpool_limits(limits=2):
        log.write(f"started={timestamp()}\nprotocol_commit={commit}\n")
        for seq_index, name in enumerate(fit_names, start=1):
            features, probability, target, _ = _read_sequence(
                videos, completion, name, input_receipt["bank_receipts"][name]["bank_sha256"]
            )
            count = features.shape[0] * 4096
            x_features[offset:offset + count] = features.reshape(count, 64)
            x_probability[offset:offset + count] = probability.reshape(count)
            y[offset:offset + count] = target.reshape(count)
            sample_weight[offset:offset + count] = 1.0 / count
            offset += count
            frame_offset += features.shape[0]
            status.update(phase="loading_fit_split", completed_sequences=seq_index,
                          completed_frames=frame_offset, updated_at=timestamp())
            atomic_json(output / "status.json", status)
            log.write(f"loaded_sequence={name} frames={features.shape[0]}\n")
            log.flush()
            del features, probability, target
            gc.collect()
        if offset != n or frame_offset != fit_frames:
            raise RuntimeError("Fit split size differs from frozen protocol")

        models: dict[str, tuple[StandardScaler, Ridge]] = {}
        model_records = {}
        for condition in ("mask_only", "feature_only", "combined"):
            start = time.perf_counter()
            status.update(phase=f"fit_{condition}", updated_at=timestamp())
            atomic_json(output / "status.json", status)
            scaler, probe = _fit(condition, x_features, x_probability, y, sample_weight)
            models[condition] = scaler, probe
            model_records[condition] = {
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient": probe.coef_.tolist(),
                "intercept": float(probe.intercept_),
                "alpha": 1.0,
            }
            log.write(f"fit_condition={condition} seconds={time.perf_counter()-start:.3f}\n")
            log.flush()
            status.update(phase=f"fit_{condition}_complete", updated_at=timestamp())
            atomic_json(output / "status.json", status)

        del x_features, x_probability, y, sample_weight
        gc.collect()

        evaluation_names = protocol["cohort"]["evaluation_split"]["sequences"]
        test_frames = protocol["cohort"]["evaluation_split"]["post_prompt_frames"]
        stored_mask = np.empty((test_frames, 4096), dtype=np.float32)
        ground_truth = np.empty((test_frames, 4096), dtype=np.float32)
        predictions = {name: np.empty((test_frames, 4096), dtype=np.float32)
                       for name in models}
        frame_ids: list[dict] = []
        per_sequence = {}
        frame_cursor, completed_eval_frames = 0, 0
        status.update(phase="evaluating_test_split", completed_frames=0,
                      expected_sequences=len(evaluation_names),
                      expected_frames=test_frames, updated_at=timestamp())
        atomic_json(output / "status.json", status)
        for seq_index, name in enumerate(evaluation_names, start=1):
            features, probability, target, local_ids = _read_sequence(
                videos, completion, name, input_receipt["bank_receipts"][name]["bank_sha256"]
            )
            metrics = {condition: [] for condition in models}
            for row, local_id in enumerate(local_ids):
                positions = slice(frame_cursor, frame_cursor + 1)
                f = features[row]
                p = probability[row]
                t = target[row]
                stored_mask[frame_cursor] = p
                ground_truth[frame_cursor] = t
                matrices = {
                    "mask_only": p[:, None].copy(),
                    "feature_only": f.copy(),
                    "combined": np.concatenate((f, p[:, None]), axis=1).astype(np.float32, copy=False),
                }
                for condition, (scaler, probe) in models.items():
                    prediction = _bounded_predict(scaler, probe, matrices[condition])
                    predictions[condition][frame_cursor] = prediction
                    metrics[condition].append(_frame_scores(prediction, t, p))
                frame_ids.append({"sequence": name, "local_frame_index": local_id})
                frame_cursor += 1
                completed_eval_frames += 1
            per_sequence[name] = {
                condition: _sequence_scores(rows) for condition, rows in metrics.items()
            }
            status.update(phase="evaluating_test_split", completed_sequences=seq_index,
                          completed_frames=completed_eval_frames, updated_at=timestamp())
            atomic_json(output / "status.json", status)
            log.write(f"evaluated_sequence={name} frames={len(local_ids)}\n")
            log.flush()
            del features, probability, target, metrics
            gc.collect()
        if frame_cursor != test_frames:
            raise RuntimeError("Evaluation split size differs from frozen protocol")

    sequence_values = {}
    for condition in models:
        sequence_values[condition] = {}
        for stratum in ("all", "present", "empty"):
            sequence_values[condition][stratum] = {}
            for metric in ("brier", "mae", "baseline_error_weighted_brier"):
                values = [per_sequence[name][condition][f"{stratum}_{metric}"]
                          for name in evaluation_names
                          if per_sequence[name][condition][f"{stratum}_{metric}"] is not None]
                sequence_values[condition][stratum][metric] = (
                    float(np.mean(values)) if values else None
                )
                sequence_values[condition][stratum][metric + "_sequences"] = len(values)
    deltas_present = np.array([
        per_sequence[name]["mask_only"]["present_brier"]
        - per_sequence[name]["combined"]["present_brier"]
        for name in evaluation_names
    ], dtype=np.float64)
    deltas_all = np.array([
        per_sequence[name]["mask_only"]["all_brier"]
        - per_sequence[name]["combined"]["all_brier"]
        for name in evaluation_names
    ], dtype=np.float64)
    present_ci = _bootstrap(deltas_present, seed=20261106)
    screen_pass = bool(deltas_present.mean() > 0 and present_ci[0] > 0 and deltas_all.mean() >= 0)
    results = {
        "completed_at": timestamp(), "scope": "exploratory same-frame representation probe",
        "inference_performed": False, "efficacy_run": False,
        "frame_count": frame_cursor, "sequence_count": len(evaluation_names),
        "conditions": sequence_values, "per_sequence": per_sequence,
        "paired_improvement_mask_minus_combined": {
            "present_brier_mean": float(deltas_present.mean()),
            "present_brier_bootstrap_95_percentile_ci": present_ci,
            "all_brier_mean": float(deltas_all.mean()),
        },
        "screen_pass": screen_pass,
        "interpretation": (
            "The frozen probe screen passes; this supports added decodable same-frame information only."
            if screen_pass else
            "The frozen probe screen did not pass; report the result as negative or inconclusive."
        ),
        "limitations": protocol["scope_limits"],
        "model_parameters": model_records,
        "frame_ids": frame_ids,
    }
    atomic_json(output / "diagnostic_results.json", results)
    np.savez_compressed(
        output / "per_sequence_predictions.npz",
        frame_sequences=np.asarray([row["sequence"] for row in frame_ids], dtype="U5"),
        local_frame_indices=np.asarray([row["local_frame_index"] for row in frame_ids], dtype=np.int32),
        mask_probability=stored_mask, target_occupancy=ground_truth,
        **{f"prediction_{name}": values for name, values in predictions.items()},
    )
    status.update(phase="complete", completed_at=timestamp(), completed_frames=frame_cursor,
                  screen_pass=screen_pass, efficacy_started=False)
    atomic_json(output / "status.json", status)
    atomic_json(output / "completion.json", {
        "complete": True, "protocol_commit": commit, "protocol_sha256": protocol_hash,
        "setup_sha256": digest(output / "setup.json"),
        "results_sha256": digest(output / "diagnostic_results.json"),
        "predictions_sha256": digest(output / "per_sequence_predictions.npz"),
        "frames": frame_cursor, "screen_pass": screen_pass,
        "efficacy_started": False,
    })
    print(json.dumps({"complete": True, "output_directory": str(output),
                      "screen_pass": screen_pass,
                      "present_brier_improvement": float(deltas_present.mean()),
                      "present_brier_ci": present_ci}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify pinned receipts and all bank tensors without fitting probes.")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        output = args.output_dir.expanduser().resolve()
        setup_path = output / "setup.json"
        if output.exists() and setup_path.is_file():
            try:
                setup = json.loads(setup_path.read_text())
                if setup.get("protocol_sha256") == digest(PROTOCOL_PATH):
                    previous = {}
                    status_path = output / "status.json"
                    if status_path.is_file():
                        previous = json.loads(status_path.read_text())
                    previous.update({"phase": "failed", "failed_at": timestamp(),
                                     "error_type": type(error).__name__,
                                     "error": str(error), "efficacy_started": False})
                    atomic_json(status_path, previous)
                    with (output / "fit_log.txt").open("a", encoding="utf-8") as log:
                        log.write(f"failure={type(error).__name__}: {error}\n")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()

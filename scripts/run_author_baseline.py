#!/usr/bin/env python3
"""Thin author-predictor benchmark: one prompt, native propagation, shared metrics.

Separate processes prevent repositories' `sam2` packages from colliding.
No replacement segmentation, selection, or attention algorithm is defined here.
"""

# ruff: noqa: E402
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from scripts.memory_token_io import (
    digest,
    prompt_index,
    synchronize_storage,
    validate_output,
)
from training.tracking import finalize_tracking, log_results_artifact, start_wandb

REPOSITORIES = {
    "sam2": ("external/sam2", "2b90b9f5ceec907a1c18123530e92e794ad901a4"),
    "sam2long": ("external/SAM2Long", "d70b50a7936fec55af201244ecde3d4433aff943"),
    "samurai": ("external/SAMURAI", "76ba195984892b0d1e3db5d9c9f90bb62175680a"),
    "lean": ("external/Lean-SAM2", "8d1cf2618519d291905d22fe859b8302476d7f50"),
}
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
for variant in ("sam2_zero", "sam2_region", "sam2_shuffled"):
    REPOSITORIES[variant] = REPOSITORIES["sam2"]


def build_author_predictor(args, repo):
    """Fresh author model prevents cross-sequence Kalman/cache state leakage."""
    torch.manual_seed(20261021)
    if args.method == "lean":
        sys.path.insert(0, str(repo / "tools"))
        from lvos.runner import parse_args, resolve_mode_config
        from lvos_inference import build_predictor_official_aligned

        runtime = parse_args(
            [
                "--config",
                str(repo / "sam2" / MODEL_CONFIG),
                "--checkpoint",
                str(args.checkpoint),
                "--config_file",
                str(repo / "configs/runtime/lvos.yaml"),
                "--config_file",
                str(repo / "configs/lean_sam2/lean_sam2.yaml"),
                "--bypass_ckpt",
                str(repo / "checkpoints/bypass_bottleneck_base.pth"),
                "--device",
                args.device,
                "--force_fp32",
            ]
        )
        model, _, bypass = build_predictor_official_aligned(
            runtime,
            resolve_mode_config(runtime),
            per_obj_png_file=False,
            apply_postprocessing=False,
            require_full_lean=True,
        )
        if not bypass:
            raise RuntimeError("Lean bypass was not loaded")
    else:
        from sam2.build_sam import build_sam2_video_predictor

        config = (
            "configs/samurai/sam2.1_hiera_b+.yaml"
            if args.method == "samurai"
            else MODEL_CONFIG
        )
        model = build_sam2_video_predictor(
            config, str(args.checkpoint), device=args.device, apply_postprocessing=False
        )
        if args.method == "samurai" and not model.samurai_mode:
            raise RuntimeError("SAMURAI mode is disabled")
    model.eval().requires_grad_(False)
    if any(m.training for m in model.modules()):
        raise RuntimeError("Author module left in training mode")
    return model


def source_identity(method):
    relative, revision = REPOSITORIES[method]
    repo = ROOT / relative
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if actual != revision or dirty:
        raise ValueError("Author repository is not the pinned clean checkout")
    return repo, revision


def iter_predictions(predictor, state, method, prompt):
    """Use the author's actual return contract, without reselecting hypotheses."""
    if method == "sam2long":
        object_ids, masks = predictor.propagate_in_video(
            state, start_frame_idx=prompt, reverse=False
        )
        if object_ids != [1] or len(masks) != state["num_frames"] - prompt:
            raise RuntimeError("Unexpected author trajectory coverage")
        for index, logits in enumerate(masks, prompt):
            yield index, object_ids, logits
    else:
        yield from predictor.propagate_in_video(
            state, start_frame_idx=prompt, reverse=False
        )


def tensor_hash(tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("Nonfinite author prediction")
    return hashlib.sha256(array.tobytes()).hexdigest()


def parameter_fingerprints(model):
    """Exact named values, shapes and dtypes; supports BF16 without NumPy casts."""
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest(),
        }
        for name, value in model.named_parameters()
    }


def assert_parameter_integrity(model, before, folder, stage):
    after = parameter_fingerprints(model)
    changed = sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    receipt = {
        "stage": stage,
        "unchanged": not changed,
        "changed_names": changed,
        "before": before,
        "after": after,
    }
    (folder / f"parameter_integrity_{stage}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if changed:
        raise RuntimeError(f"Model parameters changed: {changed}")


@torch.inference_mode()
def rollout(
    predictor,
    method,
    frames,
    prompt,
    folder,
    device,
    run,
    repeat=False,
    capture_logits=False,
):
    rows = []
    from modeling.region_bias import NativeRegionBias

    modes = {
        "sam2_zero": "zero",
        "sam2_region": "boundary",
        "sam2_shuffled": "shuffled",
    }
    controller = (
        NativeRegionBias(predictor, modes[method], frames[0].video_id, prompt)
        if method in modes
        else None
    )
    with tempfile.TemporaryDirectory(prefix="ordered_frames_", dir=folder) as temporary:
        stage_frames(frames, Path(temporary))
        state = predictor.init_state(
            video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False
        )
        if method == "sam2long":
            # Author tools/vos_inference.py CLI values, not helper defaults.
            state.update(num_pathway=3, iou_thre=0.3, uncertainty=1)
        iterator = None
        try:
            initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
            predictor.add_new_mask(
                state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial)
            )
            synchronize_storage(device)
            started = time.monotonic()

            def predictions():
                with controller if controller is not None else nullcontext():
                    yield from iter_predictions(predictor, state, method, prompt)

            iterator = predictions()
            for index, object_ids, logits in iterator:
                synchronize_storage(device)
                if controller is not None:
                    trace = controller.capture(state, index, logits)
                    if not repeat:
                        with (folder / "region_bias.jsonl").open("a") as stream:
                            stream.write(json.dumps(trace) + "\n")
                if object_ids != [1] or tuple(logits.shape) != (1, 1, *initial.shape):
                    raise RuntimeError(
                        "Object IDs or original-image coordinates changed"
                    )
                prediction_hash = tensor_hash(logits)
                if index == prompt:
                    continue
                if index != prompt + len(rows) + 1:
                    raise RuntimeError("Missing, duplicated or nonchronological target")
                mask = logits[0, 0].detach().cpu().numpy() > 0
                truth = load_label_mask(frames[index].mask_path, [1]) == 1
                row = {
                    "frame": index,
                    "source_frame_index": frames[index].frame_index,
                    "prediction_sha256": prediction_hash,
                    "present": bool(truth.any()),
                    "scores": {
                        **calculate_scores(mask, truth),
                        "boundary_f": float(db_eval_boundary(mask, truth)),
                    },
                    "absent_fp": bool(mask.any()) if not truth.any() else None,
                }
                rows.append(row)
                if capture_logits:
                    prefix = "repeat_logits" if repeat else "logits"
                    torch.save(
                        logits.detach().float().cpu(),
                        folder / f"{prefix}_{index:06d}.pt",
                    )
                if not repeat:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(
                        folder / f"frame_{index:06d}.png"
                    )
                    with (folder / "frames.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    run.log(
                        {
                            "progress/frame": index,
                            "progress/sequence": int(frames[0].video_id[3:]),
                            "frame/dice": row["scores"]["dice"],
                        }
                    )
                print(
                    f"[author] {method} {frames[0].video_id} frame={index}/{len(frames) - 1} repeat={repeat}",
                    flush=True,
                )
            synchronize_storage(device)
            seconds = time.monotonic() - started
        finally:
            if iterator is not None:
                iterator.close()
            predictor.reset_state(state)
    if len(rows) != len(frames) - prompt - 1:
        raise RuntimeError("Incomplete target coverage")
    return rows, seconds


def replay_difference(folder, frames):
    differences = []
    for row in frames:
        index = row["frame"]
        first = torch.load(folder / f"logits_{index:06d}.pt", weights_only=True)
        second = torch.load(folder / f"repeat_logits_{index:06d}.pt", weights_only=True)
        differences.append(
            {
                "frame": index,
                "exact": torch.equal(first, second),
                "max_absolute_logit_difference": float((first - second).abs().max()),
                "mean_absolute_logit_difference": float((first - second).abs().mean()),
                "changed_mask_pixels": int(((first > 0) != (second > 0)).sum()),
            }
        )
    return differences


def write_results(output, results, scope):
    """Sequence-first summary; raw frame scores remain the source of truth."""
    from evaluation.statistics import sequence_bootstrap_interval

    lines = [
        "# Author baseline results",
        "",
        f"Scope: {scope}. Smoke scores are not efficacy estimates.",
        "",
        "| Sequence | Targets | Present Dice | All-frame Dice |",
        "|---|---:|---:|---:|",
    ]
    for row in results:
        if row["status"] == "unpromptable":
            lines.append(f"| {row['sequence']} | 0 | N/A: no prompt | N/A |")
        else:
            present = (
                "N/A" if row["present_dice"] is None else f"{row['present_dice']:.5f}"
            )
            lines.append(
                f"| {row['sequence']} | {row['frames']} | {present} | {row['all_frame_dice']:.5f} |"
            )
    values = {
        r["sequence"]: r["present_dice"]
        for r in results
        if r["status"] == "completed" and r["present_dice"] is not None
    }
    if values:
        ci = sequence_bootstrap_interval(values) if len(values) > 1 else None
        lines += [
            "",
            f"Equal-sequence present Dice: {np.mean(list(values.values())):.5f}; 95% sequence-bootstrap CI: {ci}.",
        ]
    lines += [
        "",
        "Raw frame files include IoU, boundary F, absence and original source-frame IDs.",
        "Timing includes transfers, metric evaluation, mask saving and logging; concurrent validation timing is not a throughput benchmark.",
        "Seq1/seq7 are unpromptable. No 23-sequence propagation score is fabricated.",
        "PolypGen outcomes have already been inspected; comparisons are exploratory.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def signature(args, repo, revision):
    return {
        "method": args.method,
        "author_revision": revision,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(args.manifest),
        "runner_sha256": digest(Path(__file__)),
        "device": args.device,
        "torch": torch.__version__,
        "precision": "float32; autocast disabled",
        "config": "configs/samurai/sam2.1_hiera_b+.yaml"
        if args.method == "samurai"
        else MODEL_CONFIG,
        "apply_postprocessing": False,
        "memory_stride": 1,
        "author_repository": str(repo.relative_to(ROOT)),
        "fresh_model_per_rollout": True,
        "integration_sha256": {
            name: digest(ROOT / name)
            for name in [
                "modeling/region_bias.py",
                "modeling/region_tokens.py",
                "scripts/memory_token_io.py",
                "training/tracking.py",
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-validation-dir",
        type=Path,
        help="Native validation required for zero-bias exactness",
    )
    parser.add_argument("--method", choices=REPOSITORIES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two exact-replay complete sequences, seq2 and seq3; not efficacy",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        help="Required successful matching smoke before a full run",
    )
    args = parser.parse_args()
    repo, revision = source_identity(args.method)
    identity = signature(args, repo, revision)
    if args.method == "sam2_zero" and args.smoke:
        if args.reference_validation_dir is None:
            parser.error("Zero-bias validation requires --reference-validation-dir")
        reference = json.loads(
            (args.reference_validation_dir / "validation.json").read_text()
        )
        delivery = json.loads(
            (args.reference_validation_dir / "delivery.json").read_text()
        )
        if (
            not reference["passed"]
            or not delivery["completed"]
            or reference["signature"] != {**identity, "method": "sam2"}
        ):
            raise ValueError("Zero-bias/native reference identity or delivery mismatch")
    protocol = json.loads((ROOT / "protocols/memory_token_regions.json").read_text())
    if identity["checkpoint_sha256"] != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Unregistered B+ checkpoint")
    if not args.smoke:
        if args.validation_dir is None:
            parser.error("Full inference requires --validation-dir")
        gate = json.loads((args.validation_dir / "validation.json").read_text())
        delivery = json.loads((args.validation_dir / "delivery.json").read_text())
        if (
            not gate["passed"]
            or gate["signature"] != identity
            or not delivery["completed"]
        ):
            raise ValueError(
                "Validation/source/device/checkpoint mismatch or incomplete delivery"
            )
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output / key.lower())
    config = {
        **identity,
        "scope": "validation" if args.smoke else "full",
        "initialization": "first_labeled_frame",
        "prompt_count": 1,
        "future_information": "SAM2Long final path uses sequence-level trajectory selection"
        if args.method == "sam2long"
        else "forward online predictions",
        "region_modulation": {
            "enabled": args.method in ("sam2_zero", "sam2_region", "sam2_shuffled"),
            "weights_background_foreground_boundary": [1, 1, 1.1]
            if args.method in ("sam2_region", "sam2_shuffled")
            else [1, 1, 1],
            "shuffled": args.method == "sam2_shuffled",
            "label_source": "stored-frame predicted mask; Boundary IoU width 0.02; prompt excepted",
            "keys": "all retained; pointers receive zero bias",
            "rule_role": "project exploratory intervention, not an author baseline algorithm",
        },
    }
    (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(__file__, output / "runner_source.py")
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"07 | {args.method} | {config['scope']} | single-mask prompt",
        "online",
        "https://forge.coreweave.com/api/wandb",
        config,
        output,
        ["author-baseline", config["scope"]],
    )
    try:
        if "sam2" in sys.modules:
            raise RuntimeError("Run one author repository per clean process")
        sys.path.insert(0, str(repo / "sam2" if args.method == "samurai" else repo))
        torch.set_num_threads(2)
        videos = {
            **load_manifest(args.manifest, "dev"),
            **load_manifest(args.manifest, "test"),
        }
        names = ["seq2", "seq3"] if args.smoke else [f"seq{i}" for i in range(1, 24)]
        results = []
        for name in names:
            frames = videos[name]
            try:
                prompt = prompt_index(frames, protocol)
            except ValueError:
                results.append(
                    {"sequence": name, "status": "unpromptable", "frames": 0}
                )
                continue
            if args.method == "samurai":
                # Author history lookup assumes its initial prompt is index zero.
                # Preserve original source indices in provenance/evaluation rows.
                frames = frames[prompt:]
                prompt = 0
            predictor = build_author_predictor(args, repo)
            import sam2

            if not Path(sam2.__file__).resolve().is_relative_to(repo):
                raise RuntimeError("Wrong sam2 package imported")
            assert (
                predictor.num_maskmem == 7
                and predictor.memory_temporal_stride_for_eval == 1
            )
            assert predictor.fill_hole_area == 0
            before = parameter_fingerprints(predictor)
            folder = output / name
            folder.mkdir()
            provenance = [
                {
                    "frame": f.frame_index,
                    "image": str(f.image_path),
                    "mask": str(f.mask_path),
                    "image_sha256": digest(f.image_path),
                    "mask_sha256": digest(f.mask_path),
                }
                for f in frames
            ]
            (folder / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
            rows, seconds = rollout(
                predictor,
                args.method,
                frames,
                prompt,
                folder,
                args.device,
                run,
                capture_logits=args.smoke,
            )
            if args.smoke:
                assert_parameter_integrity(predictor, before, folder, "first")
                del predictor
                predictor = build_author_predictor(args, repo)
                repeated, _ = rollout(
                    predictor,
                    args.method,
                    frames,
                    prompt,
                    folder,
                    args.device,
                    run,
                    repeat=True,
                    capture_logits=True,
                )
                differences = replay_difference(folder, rows)
                (folder / "replay.json").write_text(
                    json.dumps(differences, indent=2) + "\n"
                )
                if [r["prediction_sha256"] for r in rows] != [
                    r["prediction_sha256"] for r in repeated
                ]:
                    raise RuntimeError("Repeated native logits are not bit-identical")
                if args.method == "sam2_zero":
                    reference_rows = [
                        json.loads(line)
                        for line in (
                            args.reference_validation_dir / name / "frames.jsonl"
                        )
                        .read_text()
                        .splitlines()
                    ]
                    if [
                        (r["source_frame_index"], r["prediction_sha256"]) for r in rows
                    ] != [
                        (r["source_frame_index"], r["prediction_sha256"])
                        for r in reference_rows
                    ]:
                        raise RuntimeError("Zero-bias logits differ from native SAM2")
                # Raw logits are temporary replay caches; retain exact hashes and
                # difference receipts, not gigabytes of redundant successful caches.
                for row in rows:
                    for prefix in ("logits", "repeat_logits"):
                        (folder / f"{prefix}_{row['frame']:06d}.pt").unlink()
            assert_parameter_integrity(
                predictor, before, folder, "replay" if args.smoke else "full"
            )
            del predictor
            present = [r for r in rows if r["present"]]
            result = {
                "sequence": name,
                "status": "completed",
                "frames": len(rows),
                "prompt": prompt,
                "seconds": seconds,
                "fps": len(rows) / seconds,
                "all_frame_dice": float(np.mean([r["scores"]["dice"] for r in rows])),
                "present_dice": float(np.mean([r["scores"]["dice"] for r in present]))
                if present
                else None,
                "replay_exact": True if args.smoke else None,
            }
            results.append(result)
            (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            archive = output / f"{name}.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
                for path in folder.iterdir():
                    packed.write(path, arcname=f"{name}/{path.name}")
            log_results_artifact(run, output, [archive.name])
            (output / "progress.json").write_text(json.dumps(result) + "\n")
            (output / "sequences.json").write_text(json.dumps(results, indent=2) + "\n")
            write_results(output, results, config["scope"])
        valid = [r for r in results if r["status"] == "completed"]
        summary = {
            "completed": True,
            "scope": config["scope"],
            "completed_sequences": len(valid),
            "target_frames": sum(r["frames"] for r in valid),
            "sequences": results,
            "equal_sequence_present_dice": float(
                np.mean(
                    [r["present_dice"] for r in valid if r["present_dice"] is not None]
                )
            ),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_results(output, results, config["scope"])
        names_to_upload = [
            "setup.json",
            "summary.json",
            "sequences.json",
            "runner_source.py",
            "results.md",
        ]
        if args.smoke:
            (output / "validation.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "signature": identity,
                        "checks": [
                            "strict_checkpoint_load",
                            "single_prompt",
                            "original_image_shape",
                            "all_complete_sequence_targets",
                            "finite_logits",
                            "exact_repeated_logits",
                            "frozen_parameters",
                        ],
                        "coverage": "Complete seq2 and seq3, each repeated with a fresh model and state",
                        "limits": "Two-sequence correctness only; not full-cohort efficacy or CUDA parity",
                    },
                    indent=2,
                )
                + "\n"
            )
            names_to_upload.append("validation.json")
        finalize_tracking(run, output, summary, names_to_upload)
        run.finish()
    except BaseException as error:
        (output / "error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}) + "\n"
        )
        try:
            files = [p.name for p in output.iterdir() if p.is_file()]
            for folder in output.iterdir():
                if folder.is_dir() and folder.name.startswith("seq"):
                    files.extend(
                        str(path.relative_to(output))
                        for path in folder.iterdir()
                        if path.suffix in (".json", ".jsonl")
                    )
            log_results_artifact(run, output, files)
        finally:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

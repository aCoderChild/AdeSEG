"""Official SAM2 CPU counterfactuals with prospectively committed rigid nulls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
from unittest.mock import patch

import numpy as np
import cv2
import scipy
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest, stage_frames  # noqa: E402
from modeling.contiguous_exclusion import NativeExclusion, exclusion_plan  # noqa: E402
from modeling.region_bias import feature_key  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_region_exclusions import copy_containers, active_bank_digest, evaluate  # noqa: E402
from scripts.contiguous_exclusion_statistics import summarize_sequence, report  # noqa: E402
from training.tracking import start_wandb, log_results_artifact, finalize_tracking  # noqa: E402

PROTOCOL = "protocols/contiguous_exclusions_v1.json"
SOURCES = [
    PROTOCOL,
    "modeling/contiguous_exclusion.py",
    "scripts/run_contiguous_exclusions.py",
    "scripts/contiguous_exclusion_statistics.py",
    "scripts/queue_contiguous_exclusions.py",
    "scripts/run_region_exclusions.py",
    "modeling/region_bias.py",
    "scripts/run_author_baseline.py",
    "scripts/memory_token_io.py",
    "evaluation/statistics.py",
    "evaluation/segmentation.py",
    "training/tracking.py",
    "modeling/region_tokens.py",
    "modeling/region_attention.py",
    "modeling/region_exclusion.py",
    "modeling/medsam2.py",
    "evaluation/davis2017/metrics.py",
    "datasets/io.py",
    "datasets/video.py",
]


def snapshot(state, frame):
    result = copy_containers(state)
    for outputs in result["output_dict_per_obj"].values():
        for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
            outputs[bucket] = {i: o for i, o in outputs[bucket].items() if i < frame}
    for obj, tracked in result["frames_tracked_per_obj"].items():
        result["frames_tracked_per_obj"][obj] = {
            i: v for i, v in tracked.items() if i < frame
        }
    if frame not in result["cached_features"]:
        raise RuntimeError("Target image features not cached")
    return result


def bank_digest(state, frame, units, horizon):
    # Reuse the retained exact tensor hasher, including shared image cache.
    proxy = {"output_dict": {**state["output_dict_per_obj"][0]}}
    # active_bank_digest hashes two explicit buckets; add cached features inside
    # the conditioning bucket solely in this read-only hashing view.
    proxy["output_dict"]["cond_frame_outputs"] = {
        **proxy["output_dict"]["cond_frame_outputs"],
        -1: state["cached_features"],
    }
    return active_bank_digest(
        proxy, frame, [{"frame_idx": u["frame"]} for u in units], horizon
    )


def infer(model, state, frame):
    current = copy_containers(state)
    with patch.object(
        model,
        "_encode_new_memory",
        side_effect=RuntimeError("Counterfactual encoded memory"),
    ):
        out, low = model._run_single_frame_inference(
            inference_state=current,
            output_dict=current["output_dict_per_obj"][0],
            frame_idx=frame,
            batch_size=1,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=False,
        )
    if out["maskmem_features"] is not None or out["maskmem_pos_enc"] is not None:
        raise RuntimeError("Counterfactual memory output is not empty")
    return model._get_orig_video_res_output(current, low)[1]


def assert_trace(controller, expected_layout, frames):
    if [r["layer"] for r in controller.trace] != list(range(4)):
        raise RuntimeError("Missing counterfactual layers")
    if (
        controller.layout != expected_layout
        or [u["frame"] for u in controller.last_units] != frames
    ):
        raise RuntimeError("QKV contiguity/layout or memory frames changed")


@torch.inference_mode()
def sequence(model, name, frames, source, output, protocol, gate, run):
    reference_folder = source / name
    target = output / name
    target.mkdir()
    reference_rows = {
        r["frame"]: r
        for r in map(
            json.loads, (reference_folder / "frames.jsonl").read_text().splitlines()
        )
    }
    assignments = {
        r["frame"]: r
        for r in map(
            json.loads,
            (reference_folder / "assignment_frames.jsonl").read_text().splitlines(),
        )
    }
    prompt = min(assignments)
    provenance = json.loads((reference_folder / "inputs.json").read_text())
    if len(provenance) != len(frames):
        raise RuntimeError("Source frame coverage changed")
    for f, old in zip(frames, provenance):
        if (
            f.frame_index != old["frame"]
            or digest(f.image_path) != old["image_sha256"]
            or digest(f.mask_path) != old["mask_sha256"]
        ):
            raise RuntimeError("Source images/GT/chronology changed")
    before_parameters = author.parameter_fingerprints(model)
    cases, integrity, schemes = [], [], {}
    controller = NativeExclusion(model, name, prompt)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ordered_", dir=target) as staged:
        stage_frames(frames, Path(staged))
        state = model.init_state(
            video_path=staged, offload_video_to_cpu=True, offload_state_to_cpu=False
        )
        initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
        model.add_new_mask(
            state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial)
        )
        with controller:
            for index, ids, baseline in model.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if ids != [1]:
                    raise RuntimeError("Object ID changed")
                native_out = state["output_dict_per_obj"][0]
                own = native_out["cond_frame_outputs"].get(
                    index, native_out["non_cond_frame_outputs"].get(index)
                )
                if (
                    author.tensor_hash(own["pred_masks"][0, 0])
                    != assignments[index]["native_mask_sha256"]
                ):
                    raise RuntimeError("Stored native low-res source changed")
                saved = torch.load(
                    reference_folder / f"assignments_{index:06d}.pt", weights_only=False
                )
                if (
                    author.tensor_hash(saved["native_low_res_logits"])
                    != assignments[index]["native_mask_sha256"]
                ):
                    raise RuntimeError("Saved assignment source logits changed")
                for definition in protocol["definitions"]:
                    labels = saved["schemes"][definition]["labels"]
                    if (
                        labels.shape != (64, 64)
                        or not np.isin(labels, [0, 1, 2]).all()
                        or np.bincount(labels.ravel(), minlength=3).tolist()
                        != assignments[index]["schemes"][definition]["counts"]
                    ):
                        raise RuntimeError("Saved region assignment/counts changed")
                with (target / "assignment_provenance.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "frame": index,
                                "source_file_sha256": digest(
                                    reference_folder / f"assignments_{index:06d}.pt"
                                ),
                            }
                        )
                        + "\n"
                    )
                schemes[index] = saved["schemes"]
                key = feature_key(own["maskmem_features"].flatten(2).transpose(1, 2))
                if key != assignments[index]["memory_features_sha256"] or (
                    key in controller.bank and controller.bank[key][0] != index
                ):
                    raise RuntimeError("Native memory identity changed/ambiguous")
                controller.bank[key] = (
                    index,
                    torch.from_numpy(schemes[index]["mixed_cells"]["labels"]),
                )
                if index != prompt:
                    if (
                        author.tensor_hash(baseline)
                        != reference_rows[index]["prediction_sha256"]
                    ):
                        raise RuntimeError(
                            f"Native trajectory changed at {name}/{index}"
                        )
                    selected_target = (
                        not gate or index in protocol["gate"]["selected_targets"][name]
                    )
                    if selected_target:
                        units = list(controller.last_units)
                        layout = list(controller.layout)
                        pointers = controller.pointers
                        old_digest = bank_digest(
                            state, index, units, model.max_obj_ptrs_in_encoder
                        )
                        cached = snapshot(state, index)
                        truth = load_label_mask(frames[index].mask_path, [1]) == 1
                        base_scores = evaluate(baseline, truth)
                        predicted = baseline[0, 0].cpu().numpy() > 0
                        outcome = (
                            (
                                "overlap_present"
                                if (predicted & truth).any()
                                else "complete_miss"
                            )
                            if truth.any()
                            else (
                                "false_positive" if predicted.any() else "correct_empty"
                            )
                        )

                        def call(plan):
                            controller.plan, controller.cached = plan, None
                            controller.trace.clear()
                            controller.layout.clear()
                            prediction = infer(model, cached, index)
                            assert_trace(
                                controller, layout, [u["frame"] for u in units]
                            )
                            return prediction

                        restored = call(None)
                        if not torch.equal(restored, baseline):
                            raise RuntimeError("Cached native no-op replay mismatch")
                        zero = {
                            "selected": torch.zeros(
                                sum(4096 for _ in units) + pointers, dtype=torch.bool
                            ),
                            "frames": [u["frame"] for u in units],
                            "pointers": pointers,
                            "undefined": False,
                        }
                        restored = call(zero)
                        if not torch.equal(restored, baseline):
                            raise RuntimeError("Explicit zero-mask replay mismatch")
                        zero_hash = hashlib.sha256(
                            zero["selected"].numpy().tobytes()
                        ).hexdigest()
                        cache = {
                            zero_hash: {
                                "scores": base_scores,
                                "prediction_sha256": author.tensor_hash(baseline),
                            }
                        }
                        pending = []
                        for definition in protocol["definitions"]:
                            for region, region_name in enumerate(protocol["regions"]):
                                plans = [
                                    exclusion_plan(
                                        schemes,
                                        units,
                                        pointers,
                                        definition,
                                        region,
                                        seed,
                                        f"{name}|{index}",
                                    )
                                    for seed in (
                                        None,
                                        *protocol["contiguity_null"]["seeds"],
                                    )
                                ]
                                row = {
                                    "frame": index,
                                    "source_frame_index": frames[index].frame_index,
                                    "definition": definition,
                                    "region": region_name,
                                    "present": bool(truth.any()),
                                    "size_bin": assignments[index]["size_bin"],
                                    "outcome": outcome,
                                    "baseline": base_scores,
                                    "informative": plans[0]["informative"],
                                    "excluded": None,
                                    "controls": [],
                                    "excluded_tokens": int(plans[0]["selected"].sum()),
                                    "undefined": plans[0]["undefined"],
                                }
                                if not any(p["undefined"] for p in plans):
                                    for ordinal, plan in enumerate(plans):
                                        if plan["sha256"] not in cache:
                                            prediction = call(plan)
                                            cache[plan["sha256"]] = {
                                                "scores": evaluate(prediction, truth),
                                                "prediction_sha256": author.tensor_hash(
                                                    prediction
                                                ),
                                            }
                                            if index == prompt + 1:
                                                Image.fromarray(
                                                    (
                                                        prediction[0, 0].cpu().numpy()
                                                        > 0
                                                    ).astype(np.uint8)
                                                    * 255
                                                ).save(
                                                    target
                                                    / f"illustration_{definition}_{region}_{ordinal}.png"
                                                )
                                        item = {
                                            **cache[plan["sha256"]],
                                            "mask_sha256": plan["sha256"],
                                            "per_memory_frame": plan["units"],
                                        }
                                        if ordinal == 0:
                                            row["excluded"], row["regional_control"] = (
                                                item["scores"],
                                                item,
                                            )
                                        else:
                                            row["controls"].append(
                                                {
                                                    "seed": protocol["contiguity_null"][
                                                        "seeds"
                                                    ][ordinal - 1],
                                                    **item,
                                                }
                                            )
                                pending.append(row)
                        restored = call(zero)
                        if (
                            not torch.equal(restored, baseline)
                            or bank_digest(
                                state, index, units, model.max_obj_ptrs_in_encoder
                            )
                            != old_digest
                        ):
                            raise RuntimeError(
                                "Native bank/image-cache/target changed after interventions"
                            )
                        receipt = {
                            "frame": index,
                            "cached_native_exact": True,
                            "zero_mask_exact": True,
                            "bank_image_cache_exact": True,
                            "qkv_layout_exact": True,
                            "no_memory_encoding_or_writeback": True,
                            "unique_counterfactual_masks": len(cache),
                            "cases": len(pending),
                        }
                        integrity.append(receipt)
                        cases.extend(pending)
                        with (target / "cases.jsonl").open("a") as stream:
                            stream.write("".join(json.dumps(r) + "\n" for r in pending))
                        with (target / "integrity.jsonl").open("a") as stream:
                            stream.write(json.dumps(receipt) + "\n")
                        run.log(
                            {
                                "progress/frame": index,
                                "progress/sequence": int(name[3:]),
                                "progress/cases": len(cases),
                            }
                        )
                        print(
                            f"[rigid-exclusion] {name} target={index} cases={len(cases)} unique_masks={len(cache)}",
                            flush=True,
                        )
                    with (target / "native_parity.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "frame": index,
                                    "prediction_sha256": author.tensor_hash(baseline),
                                    "exact": True,
                                }
                            )
                            + "\n"
                        )
                controller.plan, controller.cached = None, None
                controller.trace.clear()
                controller.layout.clear()
                controller.frame = index + 1
        model.reset_state(state)
    if (
        len(cases)
        != (
            len(protocol["gate"]["selected_targets"][name])
            if gate
            else len(reference_rows)
        )
        * 18
    ):
        raise RuntimeError("Incomplete exclusion coverage")
    author.assert_parameter_integrity(model, before_parameters, target, "full_sequence")
    result = {
        "sequence": name,
        "status": "completed",
        "frames": len(reference_rows),
        "intervention_targets": len(integrity),
        "cases": len(cases),
        "seconds": time.monotonic() - started,
        "statistics": summarize_sequence(cases),
        "native_trajectory_exact": True,
    }
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    archive = output / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
        for path in target.iterdir():
            packed.write(path, arcname=f"{name}/{path.name}")
    log_results_artifact(run, output, [archive.name])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *SOURCES],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *SOURCES],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    reference = json.loads((args.reference_dir / "setup.json").read_text())
    delivery = json.loads((args.reference_dir / "delivery.json").read_text())
    if (
        not delivery["wandb_artifact_upload_complete"]
        or not json.loads((args.reference_dir / "summary.json").read_text())[
            "completed"
        ]
    ):
        raise RuntimeError("Attention reference is incomplete")
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    identity = author.signature(args, repo, revision)
    if any(reference[key] != value for key, value in identity.items()):
        raise RuntimeError("Native reference/engine/source/checkpoint mismatch")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", *SOURCES], cwd=ROOT, text=True
    ).strip()
    identity.update(
        source_hashes={name: digest(ROOT / name) for name in SOURCES},
        protocol_commit=commit,
        control_libraries={
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
        },
        reference_setup_sha256=digest(args.reference_dir / "setup.json"),
    )
    if not args.smoke:
        if args.validation_dir is None:
            raise RuntimeError("Full exclusions require a completed matching gate")
        gate = json.loads((args.validation_dir / "validation.json").read_text())
        if (
            not gate["passed"]
            or gate["identity"] != identity
            or not json.loads((args.validation_dir / "delivery.json").read_text())[
                "completed"
            ]
        ):
            raise RuntimeError("Exclusion gate identity/delivery mismatch")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for variable in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[variable] = str(output / variable.lower())
    scope = "gate" if args.smoke else "full"
    (output / "setup.json").write_text(
        json.dumps({**identity, "scope": scope, "protocol": protocol}, indent=2) + "\n"
    )
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as packed:
        for name in SOURCES:
            packed.write(ROOT / name, name)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"17 | Rigid contiguity-matched exclusions | {scope}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {**identity, "scope": scope, "protocol": protocol},
        output,
        ["rigid-exclusion", "cpu-fp32", "prospective-protocol"],
    )
    try:
        torch.set_num_threads(2)
        sys.path.insert(0, str(repo))
        videos = {
            **load_manifest(args.manifest, "dev"),
            **load_manifest(args.manifest, "test"),
        }
        source_results = json.loads((args.reference_dir / "sequences.json").read_text())
        statuses = {r["sequence"]: r["status"] for r in source_results}
        if set(statuses) != {f"seq{i}" for i in range(1, 24)} or any(
            value != ("unpromptable" if name in ("seq1", "seq7") else "completed")
            for name, value in statuses.items()
        ):
            raise RuntimeError("Reference cohort coverage/status changed")
        names = (
            protocol["gate"]["sequences"]
            if args.smoke
            else [f"seq{i}" for i in range(1, 24)]
        )
        results = []
        for name in names:
            if statuses[name] != "completed":
                results.append(
                    {"sequence": name, "status": "unpromptable", "frames": 0}
                )
                continue
            model = author.build_author_predictor(args, repo)
            results.append(
                sequence(
                    model,
                    name,
                    videos[name],
                    args.reference_dir,
                    output,
                    protocol,
                    args.smoke,
                    run,
                )
            )
            del model
            (output / "sequences.json").write_text(json.dumps(results, indent=2) + "\n")
            report_scope = (
                scope
                if args.smoke or sum(r["status"] == "completed" for r in results) == 21
                else "full_in_progress"
            )
            report(output, results, protocol, report_scope)
            (output / "progress.json").write_text(
                json.dumps({"state": "running", "last_sequence": name}) + "\n"
            )
        valid = [r for r in results if r["status"] == "completed"]
        expected_targets, expected_sequences = (73, 2) if args.smoke else (2012, 21)
        if (
            len(valid) != expected_sequences
            or sum(r["frames"] for r in valid) != expected_targets
        ):
            raise RuntimeError("Native cohort target coverage changed")
        summary = {
            "completed": True,
            "scope": scope,
            "completed_sequences": len(valid),
            "native_targets": sum(r["frames"] for r in valid),
            "intervention_targets": sum(r["intervention_targets"] for r in valid),
            "cases": sum(r["cases"] for r in valid),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        names_to_upload = [
            "setup.json",
            "sources.zip",
            "summary.json",
            "sequences.json",
            "exclusion_results.md",
            "exclusion_results.json",
        ]
        if args.smoke:
            (output / "validation.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "identity": identity,
                        "checks": "All73 native hashes, six fixed target counterfactuals, no-op/zero-mask/layout/bank/encoder/parameters/shape-controls",
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
            json.dumps({"type": type(error).__name__, "error": str(error)}) + "\n"
        )
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

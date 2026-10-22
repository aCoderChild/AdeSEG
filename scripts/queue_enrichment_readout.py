#!/usr/bin/env python3
"""Calibrate all eligible sequences, freeze ratios, validate and run readouts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402

STRATEGIES = ("proportional", "inverse", "random")
REGIONS = ("background", "foreground_interior", "boundary")


def estimate_recipe(source, manifest):
    """Dataset-specific CPU calibration; undefined regions are not zero."""
    import math

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    requested = sorted({r["video_id"] for r in rows})
    setup = json.loads((source / "setup.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    delivery = json.loads((source / "delivery.json").read_text())
    if (
        setup["device"] != "cpu"
        or setup["method"] != "sam2_calibration"
        or setup["calibration_label_source"] != "predicted_0.02"
        or setup["manifest_sha256"] != digest(manifest)
        or not summary["completed"]
        or summary["scope"] != "full"
        or not delivery["completed"]
        or not delivery["wandb_artifact_upload_complete"]
    ):
        raise ValueError("Incomplete or incompatible native CPU calibration")
    statuses = {s["sequence"]: s["status"] for s in summary["sequences"]}
    if sorted(statuses) != requested or any(
        v not in ("completed", "unpromptable") for v in statuses.values()
    ):
        raise ValueError("Calibration does not account for every requested sequence")
    estimates, hashes = {}, {}
    for name in requested:
        if statuses[name] == "unpromptable":
            continue
        path = source / name / "attention_summary.json"
        record = json.loads(path.read_text())
        if (
            record["sequence"] != name
            or not record["native_predictions_bit_identical"]
            or record["label_source"] != "predicted_0.02"
        ):
            raise ValueError("Invalid native-attention receipt")
        hashes[name] = digest(path)
        estimates[name] = {}
        for region in REGIONS:
            values = [
                record["layers"][str(layer)]["regions"][region]["enrichment"]
                for layer in range(4)
            ]
            if all(v is None for v in values):
                estimates[name][region] = None
                continue
            if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Undefined enrichment for {name}/{region}")
            estimates[name][region] = sum(values) / 4
    if not estimates:
        raise ValueError("No eligible calibration observations")
    contributors = {
        r: [s for s in estimates if estimates[s][r] is not None] for r in REGIONS
    }
    if any(not s for s in contributors.values()):
        raise ValueError("Region has no defined calibration enrichment")
    return {
        "schema_version": 2,
        "calibration_scope": "all_eligible_sequences",
        "regions": REGIONS,
        "requested_sequences": requested,
        "eligible_sequences": list(estimates),
        "excluded_sequences": {s: v for s, v in statuses.items() if v != "completed"},
        "contributing_sequences": contributors,
        "manifest_sha256": digest(manifest),
        "source_summary_sha256": hashes,
        "source_setup_sha256": digest(source / "setup.json"),
        "source_completion_sha256": digest(source / "summary.json"),
        "runtime_identity": {
            key: setup[key]
            for key in (
                "checkpoint_sha256",
                "author_revision",
                "author_repository",
                "config",
                "device",
                "torch",
                "precision",
                "apply_postprocessing",
                "memory_stride",
                "integration_sha256",
            )
        },
        "source_directory": str(source),
        "label_source": "predicted_0.02; no GT-reference enrichment",
        "aggregation": "Equal four-layer mean of conditional frame-mean enrichment within sequence; equal eligible-sequence mean per region, excluding undefined groups",
        "mean_enrichment": [
            sum(estimates[s][r] for s in contributors[r]) / len(contributors[r])
            for r in REGIONS
        ],
        "sequence_enrichment": estimates,
        "pointer_weight": 1,
        "normalization": "None: raw enrichment and its reciprocal; common spatial rescaling would change competition with pointers",
        "status": "Dataset-specific all-cohort calibration and evaluation; not held-out validation",
        "supersedes": "Development-only calibration and fixed x1.1 boundary boost are retired from the active plan",
    }


def summarize(output, parent):
    import numpy as np
    from evaluation.statistics import (
        holm_adjust,
        paired_wilcoxon,
        sequence_bootstrap_interval,
    )
    from training.tracking import start_wandb, finalize_tracking

    values, coverage = {}, {}
    folders = {
        "native": parent / "sam2_full",
        **{s: output / f"{s}_full" for s in STRATEGIES},
    }
    for method, folder in folders.items():
        summary = json.loads((folder / "summary.json").read_text())
        delivery = json.loads((folder / "delivery.json").read_text())
        if not summary["completed"] or not delivery["wandb_artifact_upload_complete"]:
            raise RuntimeError("Incomplete method or upload")
        values[method], coverage[method] = {}, {}
        for seq in summary["sequences"]:
            if seq["status"] != "completed":
                continue
            name = seq["sequence"]
            records = [
                json.loads(line)
                for line in (folder / name / "frames.jsonl").read_text().splitlines()
            ]
            coverage[method][name] = [
                (r["source_frame_index"], r["present"]) for r in records
            ]
            values[method][name] = seq["present_dice"]
        if coverage[method] != coverage["native"]:
            raise RuntimeError("Unmatched evaluation frames")
    contrasts = [(s, "native") for s in STRATEGIES] + [("proportional", "random")]
    results = []
    for a, b in contrasts:
        changes = {
            s: values[a][s] - values[b][s]
            for s in values[b]
            if values[b][s] is not None
        }
        results.append(
            {
                "contrast": f"{a} - {b}",
                "n": len(changes),
                "delta": float(np.mean(list(changes.values()))),
                "ci": sequence_bootstrap_interval(changes),
                "p": paired_wilcoxon(values[a], values[b])["p"],
            }
        )
    adjusted = holm_adjust({r["contrast"]: r["p"] for r in results})
    lines = [
        "# Frozen enrichment readouts",
        "",
        "Exploratory. Equal-sequence present-frame Dice; seq1/seq7 unpromptable.",
        "",
        "| Method | Dice |",
        "|---|---:|",
    ]
    for method, scores in values.items():
        lines.append(
            f"| {method} | {np.mean([v for v in scores.values() if v is not None]):.5f} |"
        )
    lines += [
        "",
        "| Contrast | N | Mean delta | 95% sequence CI | Holm p |",
        "|---|---:|---:|---|---:|",
    ]
    for r in results:
        r["holm_p"] = adjusted[r["contrast"]]
        lines.append(
            f"| {r['contrast']} | {r['n']} | {r['delta']:.5f} | {r['ci']} | {r['holm_p']} |"
        )
    lines += [
        "",
        "Inverse has no matched shuffled-inverse control. Ratios are calibrated and evaluated on the same full eligible cohort. Dataset-specific adaptation, not held-out validation. Seq1/seq7 are accounted for as unpromptable, never assigned fabricated scores.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "09 | Enrichment readouts | paired summary",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"strategies": STRATEGIES},
        output,
        ["enrichment-readout", "paired-summary"],
    )
    try:
        finalize_tracking(
            run,
            output,
            {"completed": True},
            [
                "comparison.md",
                "comparison.json",
                "enrichment_recipe.json",
                "queue_source.py",
            ],
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "queue_source.py")
    recipe_path = output / "enrichment_recipe.json"
    watched = [
        "scripts/run_author_baseline.py",
        "scripts/run_enrichment_readout.py",
        "modeling/region_bias.py",
        "modeling/region_tokens.py",
        "training/tracking.py",
        "scripts/memory_token_io.py",
        "scripts/calibrate_native_enrichment.py",
        "scripts/measure_region_attention.py",
        "modeling/region_attention.py",
        "evaluation/statistics.py",
        "scripts/queue_enrichment_readout.py",
    ]
    hashes = {name: digest(ROOT / name) for name in watched}
    (output / "source_identity.json").write_text(json.dumps(hashes, indent=2) + "\n")
    receipt = {
        "state": "prepared" if args.prepare_only else "waiting_for_parent",
        "strategies": STRATEGIES,
        "parent": str(args.parent),
        "completed_steps": [],
        "weights": None,
        "device": "cpu",
        "calibration_scope": "all_eligible_sequences",
    }

    def save():
        receipt["updated_at"] = time.time()
        (output / "queue_progress.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )

    save()
    if args.prepare_only:
        return
    try:
        while True:
            parent = json.loads((args.parent / "queue_progress.json").read_text())
            if parent["state"] == "completed_supported_methods":
                break
            if parent["state"] != "running":
                raise RuntimeError(f"Parent queue not successful: {parent['state']}")
            if args.parent_pid is None:
                raise RuntimeError("Running parent requires its PID")
            os.kill(args.parent_pid, 0)
            save()
            time.sleep(30)
        native = json.loads((args.parent / "sam2_full/setup.json").read_text())
        if (
            native["manifest_sha256"] != digest(args.manifest)
            or native["checkpoint_sha256"] != digest(args.checkpoint)
            or native["device"] != "cpu"
        ):
            raise ValueError("Native comparator identity mismatch")
        for phase in ("validation", "full"):
            if hashes != {name: digest(ROOT / name) for name in watched}:
                raise RuntimeError("Calibration implementation changed")
            receipt.update(
                state="running", phase=f"calibration_{phase}", strategy="native"
            )
            save()
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/calibrate_native_enrichment.py"),
                "--reference-baseline",
                str(args.parent / "sam2_full"),
                "--method",
                "sam2_zero",
                "--checkpoint",
                str(args.checkpoint),
                "--manifest",
                str(args.manifest),
                "--device",
                "cpu",
                "--output-dir",
                str(output / f"calibration_{phase}"),
            ]
            command += (
                ["--smoke"]
                if phase == "validation"
                else ["--validation-dir", str(output / "calibration_validation")]
            )
            with (output / f"calibration_{phase}.log").open("w") as stream:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            delivery = json.loads(
                (output / f"calibration_{phase}/delivery.json").read_text()
            )
            if (
                not delivery["completed"]
                or not delivery["wandb_artifact_upload_complete"]
            ):
                raise RuntimeError("Unconfirmed calibration delivery")
            receipt["completed_steps"].append(f"calibration/{phase}")
        recipe = estimate_recipe(output / "calibration_full", args.manifest)
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        receipt["weights"] = recipe["mean_enrichment"]
        receipt["contributing_sequences"] = recipe["contributing_sequences"]
        save()
        for phase in ("validation", "full"):
            for strategy in STRATEGIES:
                if hashes != {name: digest(ROOT / name) for name in watched}:
                    raise RuntimeError(
                        "Queued implementation changed: validate a new release"
                    )
                receipt.update(state="running", phase=phase, strategy=strategy)
                save()
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_enrichment_readout.py"),
                    "--recipe",
                    str(recipe_path),
                    "--strategy",
                    strategy,
                    "--method",
                    "sam2_shuffled" if strategy == "random" else "sam2_region",
                    "--manifest",
                    str(args.manifest),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output / f"{strategy}_{phase}"),
                ]
                command += (
                    ["--smoke"]
                    if phase == "validation"
                    else ["--validation-dir", str(output / f"{strategy}_validation")]
                )
                with (output / f"{strategy}_{phase}.log").open("w") as stream:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                delivery = json.loads(
                    (output / f"{strategy}_{phase}/delivery.json").read_text()
                )
                if (
                    not delivery["completed"]
                    or not delivery["wandb_artifact_upload_complete"]
                ):
                    raise RuntimeError("Unconfirmed W&B delivery")
                receipt["completed_steps"].append(f"{strategy}/{phase}")
        summarize(output, args.parent)
        receipt["state"] = "completed"
    except BaseException as error:
        receipt.update(state="blocked", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

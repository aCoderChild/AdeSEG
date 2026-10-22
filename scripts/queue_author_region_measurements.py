#!/usr/bin/env python3
"""Serial, fail-closed author regional measurements with saved progress."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from modeling.region_tokens import REGION_NAMES  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from training.tracking import start_wandb, finalize_tracking  # noqa: E402


def completed(folder):
    summary = json.loads((folder / "summary.json").read_text())
    delivery = json.loads((folder / "delivery.json").read_text())
    if not summary["completed"] or not delivery["wandb_artifact_upload_complete"]:
        raise RuntimeError(f"Incomplete scientific output or W&B delivery: {folder}")
    return summary


def summarize(output, native):
    sources = {
        "sam2": native,
        **{m: output / f"{m}_full" for m in ("samurai", "sam2long")},
    }
    records, statistics = {}, []
    lines = [
        "# Regional attention across author predictors",
        "",
        "Read-only measurement. Attention allocation is not causal utility.",
        "SAM2/SAMURAI: online reads. SAM2Long: all executed branch reads, not final-path attribution; its final mask uses whole-video path selection.",
        "Sequence means receive equal weight. Empty-region enrichment is undefined. Intervals bootstrap whole sequences (10,000 resamples).",
        "",
        "| Method | Eligible sequences | Postprompt targets | Present-frame Dice |",
        "|---|---:|---:|---:|",
    ]
    for method, source in sources.items():
        summary = completed(source)
        rows = []
        for sequence in summary["sequences"]:
            if sequence["status"] != "completed":
                continue
            attention = json.loads(
                (source / sequence["sequence"] / "attention_summary.json").read_text()
            )
            if not attention["native_predictions_bit_identical"]:
                raise RuntimeError("Missing per-sequence native parity")
            rows.append({**sequence, "layers": attention["layers"]})
        if len(rows) != 21 or summary["target_frames"] != 2012:
            raise RuntimeError("Unexpected full-cohort coverage")
        records[method] = rows
        lines.append(
            f"| {method} | {len(rows)} | {summary['target_frames']} | {summary['equal_sequence_present_dice']:.6f} |"
        )
    lines += [
        "",
        "| Method | Layer | Region | Enrichment N | Mean token share | Native mass | Enrichment [95% CI] | Empty-read fraction | Pointer mass |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for method, rows in records.items():
        for layer in map(str, range(4)):
            for region in REGION_NAMES:
                values = {
                    r["sequence"]: r["layers"][layer]["regions"][region] for r in rows
                }
                enrichment = {
                    name: v["enrichment"]
                    for name, v in values.items()
                    if v["enrichment"] is not None
                }
                mean = float(np.mean(list(enrichment.values()))) if enrichment else None
                ci = (
                    sequence_bootstrap_interval(enrichment)
                    if len(enrichment) > 1
                    else None
                )
                result = {
                    "method": method,
                    "layer": int(layer),
                    "region": region,
                    "enrichment_n": len(enrichment),
                    "enrichment_mean": mean,
                    "enrichment_ci": ci,
                }
                for label, key in (
                    ("token_share", "spatial_token_fraction"),
                    ("mass", "mass"),
                    ("empty_read_fraction", "empty_frame_fraction"),
                ):
                    result[label] = float(np.mean([v[key] for v in values.values()]))
                result["pointer_mass"] = float(
                    np.mean([r["layers"][layer]["pointer_mass"] for r in rows])
                )
                statistics.append(result)
                estimate = f"{mean:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]" if ci else "N/A"
                lines.append(
                    f"| {method} | {layer} | {region} | {len(enrichment)} | {result['token_share']:.6f} | {result['mass']:.6f} | {estimate} | {result['empty_read_fraction']:.6f} | {result['pointer_mass']:.6f} |"
                )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "sources": {k: str(v) for k, v in sources.items()},
                "sequences": records,
                "statistics": statistics,
            },
            indent=2,
        )
        + "\n"
    )
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "12 | Author regional attention | full comparison",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"sources": {k: str(v) for k, v in sources.items()}, "causal_claim": False},
        output,
        ["regional-attention", "full-summary"],
    )
    finalize_tracking(
        run,
        output,
        {"completed": True, "methods": list(sources), "eligible_sequences": 21},
        ["comparison.md", "comparison.json", "source_identity.json"],
    )
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    watched = [
        "scripts/measure_author_regions.py",
        "scripts/queue_author_region_measurements.py",
        "scripts/run_author_baseline.py",
        "scripts/calibrate_native_enrichment.py",
        "scripts/measure_region_attention.py",
        "modeling/region_attention.py",
        "modeling/region_bias.py",
        "modeling/region_tokens.py",
        "scripts/memory_token_io.py",
        "training/tracking.py",
        "evaluation/statistics.py",
    ]
    hashes = {name: digest(ROOT / name) for name in watched}
    (output / "source_identity.json").write_text(json.dumps(hashes, indent=2) + "\n")
    sources = output / "source"
    for name in watched:
        destination = sources / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    progress = {
        "state": "prepared",
        "pid": os.getpid(),
        "completed_steps": [],
        "device": "cpu",
        "methods": ["samurai", "sam2long"],
        "native_sam2": str(args.native_calibration),
    }

    def save():
        progress["updated_unix"] = time.time()
        (output / "queue_progress.json").write_text(
            json.dumps(progress, indent=2) + "\n"
        )

    try:
        completed(args.native_calibration)
        for method in ("samurai", "sam2long"):
            for phase in ("validation", "full"):
                if method == "samurai" and phase == "full":
                    while True:
                        prior = json.loads(
                            (args.wait_for / "queue_progress.json").read_text()
                        )
                        if prior["state"] == "completed":
                            break
                        if prior["state"] == "blocked":
                            raise RuntimeError(
                                "Previous weighting queue blocked; inspect before continuing"
                            )
                        progress.update(
                            state="waiting_for_existing_run", method=method, phase=phase
                        )
                        save()
                        time.sleep(30)
                if hashes != {name: digest(ROOT / name) for name in watched}:
                    raise RuntimeError(
                        "Measurement source changed; new validation required"
                    )
                progress.update(state="running", method=method, phase=phase)
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/measure_author_regions.py"),
                    "--reference-baseline",
                    str(args.parent / f"{method}_full"),
                    "--method",
                    method,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--manifest",
                    str(args.manifest),
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output / f"{method}_{phase}"),
                ]
                command += (
                    ["--smoke"]
                    if phase == "validation"
                    else ["--validation-dir", str(output / f"{method}_validation")]
                )
                with (output / f"{method}_{phase}.log").open("w") as stream:
                    with subprocess.Popen(
                        command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                    ) as process:
                        progress["child_pid"] = process.pid
                        save()
                        if process.wait() != 0:
                            raise RuntimeError(
                                f"{method}/{phase} failed; inspect saved log"
                            )
                completed(output / f"{method}_{phase}")
                progress["completed_steps"].append(f"{method}/{phase}")
                progress.pop("child_pid", None)
                save()
        summarize(output, args.native_calibration)
        progress["state"] = "completed"
    except BaseException as error:
        progress.update(state="blocked", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate every supported author adapter before launching full matched runs.

SMR-only is deliberately excluded until CUDA pruning/encoder parity is verified.
No model algorithm is implemented by this sequential subprocess supervisor.
"""

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
from scripts.memory_token_io import validate_output  # noqa: E402

METHODS = (
    "sam2",
    "sam2_zero",
    "sam2_region",
    "sam2_shuffled",
    "sam2long",
    "samurai",
    "lean",
)


def summarize(output):
    """Paired sequence-level comparison; source-frame coverage must match."""
    import numpy as np
    from evaluation.statistics import (
        holm_adjust,
        paired_wilcoxon,
        sequence_bootstrap_interval,
    )
    from training.tracking import start_wandb, finalize_tracking

    methods = tuple(m for m in METHODS if m != "sam2_zero")
    values, coverage, frame_records = {}, {}, {}
    for method in methods:
        folder = output / f"{method}_full"
        result = json.loads((folder / "summary.json").read_text())
        values[method], coverage[method], frame_records[method] = {}, {}, {}
        for row in result["sequences"]:
            if row["status"] != "completed":
                continue
            name = row["sequence"]
            records = [
                json.loads(line)
                for line in (folder / name / "frames.jsonl").read_text().splitlines()
            ]
            coverage[method][name] = [
                (r["source_frame_index"], r["present"]) for r in records
            ]
            frame_records[method][name] = records
            if row["present_dice"] is not None:
                values[method][name] = row["present_dice"]
        if method != "sam2" and coverage[method] != coverage["sam2"]:
            raise RuntimeError("Method target/source-frame coverage differs")
    baseline = values["sam2"]
    table, tests = [], {}
    for method in methods:
        if values[method].keys() != baseline.keys():
            raise RuntimeError("Method sequence denominators differ")
        delta = {name: values[method][name] - score for name, score in baseline.items()}
        table.append(
            {
                "method": method,
                "sequences": len(delta),
                "present_dice": float(np.mean(list(values[method].values()))),
                "delta": float(np.mean(list(delta.values()))),
                "delta_ci": sequence_bootstrap_interval(delta),
            }
        )
        if method != "sam2":
            tests[method] = {
                key: None
                if isinstance(value, float) and not np.isfinite(value)
                else value
                for key, value in paired_wilcoxon(values[method], baseline).items()
            }
    regional_contrast = None
    if "sam2_region" in values and "sam2_shuffled" in values:
        contrasts = {
            name: values["sam2_region"][name] - values["sam2_shuffled"][name]
            for name in baseline
        }
        tests["region_vs_shuffled"] = {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in paired_wilcoxon(
                values["sam2_region"], values["sam2_shuffled"]
            ).items()
        }
        regional_contrast = {
            "delta": float(np.mean(list(contrasts.values()))),
            "delta_ci": sequence_bootstrap_interval(contrasts),
        }
    corrected = holm_adjust({name: test["p"] for name, test in tests.items()})
    lines = [
        "# Matched author baselines",
        "",
        "Exploratory; equal-sequence present-frame Dice. Seq1/seq7 unpromptable.",
        "",
        "| Method | N | Dice | Delta vs SAM2 [95% CI] | Holm p |",
        "|---|---:|---:|---|---|",
    ]
    for row in table:
        p = corrected.get(row["method"], "reference")
        row["holm_p"] = p
        lines.append(
            f"| {row['method']} | {row['sequences']} | {row['present_dice']:.5f} | {row['delta']:.5f} {row['delta_ci']} | {p} |"
        )
    if regional_contrast is not None:
        regional_contrast["holm_p"] = corrected["region_vs_shuffled"]
        lines += [
            "",
            f"Regional minus shuffled-label Dice: {regional_contrast['delta']:.5f}; 95% CI {regional_contrast['delta_ci']}; Holm p={regional_contrast['holm_p']}.",
        ]
    strata = []
    lines += [
        "",
        "## Baseline-defined target strata",
        "",
        "| Method | Subset | N sequences | Paired Dice delta [95% CI] |",
        "|---|---|---:|---|",
    ]
    for method in methods:
        if method == "sam2":
            continue
        for subset in ("baseline_failure_present", "baseline_success_present"):
            changes = {}
            for name, base_rows in frame_records["sam2"].items():
                paired = frame_records[method][name]
                selected = [
                    i
                    for i, row in enumerate(base_rows)
                    if row["present"]
                    and (
                        (row["scores"]["dice"] < 0.5)
                        == (subset == "baseline_failure_present")
                    )
                ]
                if selected:
                    changes[name] = float(
                        np.mean(
                            [
                                paired[i]["scores"]["dice"]
                                - base_rows[i]["scores"]["dice"]
                                for i in selected
                            ]
                        )
                    )
            if changes:
                result = {
                    "method": method,
                    "subset": subset,
                    "sequences": len(changes),
                    "delta": float(np.mean(list(changes.values()))),
                    "ci": sequence_bootstrap_interval(changes),
                }
                strata.append(result)
                lines.append(
                    f"| {method} | {subset} | {len(changes)} | {result['delta']:.5f} {result['ci']} |"
                )
    lines += [
        "",
        "SAM2Long uses offline final-path selection; these full-method differences do not isolate frame selection.",
        "Matched FP32/device; no stride 2/4. SMR-only remains blocked. Regional weights fixed before these outcomes: [background,foreground,boundary]=[1,1,1.1]. Zero bias is a validation-only exactness control.",
        "Nominal test sequences were previously inspected. This is not a fresh confirmatory test.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "rows": table,
                "tests": tests,
                "regional_control_contrast": regional_contrast,
                "strata": strata,
            },
            indent=2,
        )
        + "\n"
    )
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "07 | Matched author baselines | paired summary",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"methods": methods},
        output,
        ["author-baseline", "paired-summary"],
    )
    try:
        finalize_tracking(
            run,
            output,
            {"completed": True, "methods": len(methods)},
            ["comparison.md", "comparison.json"],
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "batch_source.py")
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output / key.lower())
    receipt = {
        "state": "running",
        "device": args.device,
        "methods": METHODS,
        "excluded": ["stride2", "stride4"],
        "blocked": {
            "efficient_smr": "CUDA SMR-active / dense-encoder parity not verified"
        },
        "completed_steps": [],
    }
    progress = output / "queue_progress.json"
    try:
        for phase in ("validation", "full"):
            for method in (
                METHODS
                if phase == "validation"
                else tuple(m for m in METHODS if m != "sam2_zero")
            ):
                receipt.update(phase=phase, method=method, updated_at=time.time())
                progress.write_text(json.dumps(receipt, indent=2) + "\n")
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_author_baseline.py"),
                    "--method",
                    method,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--manifest",
                    str(args.manifest),
                    "--device",
                    args.device,
                    "--output-dir",
                    str(output / f"{method}_{phase}"),
                ]
                if phase == "validation":
                    command.append("--smoke")
                    if method == "sam2_zero":
                        command += [
                            "--reference-validation-dir",
                            str(output / "sam2_validation"),
                        ]
                else:
                    command += [
                        "--validation-dir",
                        str(output / f"{method}_validation"),
                    ]
                print(f"[batch] {phase}: {method}", flush=True)
                with (output / f"{method}_{phase}.log").open("w") as stream:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                delivery = json.loads(
                    (output / f"{method}_{phase}" / "delivery.json").read_text()
                )
                if (
                    not delivery["completed"]
                    or not delivery["wandb_artifact_upload_complete"]
                ):
                    raise RuntimeError(f"Unconfirmed delivery for {method}/{phase}")
                receipt["completed_steps"].append(f"{method}/{phase}")
        summarize(output)
        receipt["state"] = "completed_supported_methods"
    except BaseException as error:
        receipt.update(state="blocked", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["updated_at"] = time.time()
        progress.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

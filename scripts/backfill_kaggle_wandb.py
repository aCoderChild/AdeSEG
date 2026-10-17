#!/usr/bin/env python3
"""Publish saved Phase-A Kaggle results when its live W&B setup failed."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def collect(output_dir):
    metrics = {}
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name not in {
            "summary.json", "frames.jsonl", "setup.json", "status.json",
            "events.jsonl", "taxonomy.json", "mechanism.json",
            "utility_records.pt", "tracking.json", "delivery_receipt.json",
        }:
            continue
        files.append(path)
        if path.name == "summary.json":
            stage = path.parent.relative_to(output_dir).as_posix()
            for key, value in json.loads(path.read_text()).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"{stage}/{key}"] = value
        elif path.name == "frames.jsonl":
            stage = path.parent.relative_to(output_dir).as_posix()
            metrics[f"{stage}/recorded_frames"] = sum(1 for line in path.read_text().splitlines() if line.strip())
    return metrics, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--version-id", required=True, help="Kaggle scriptVersionId being backfilled")
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required to backfill results")
    output_dir = args.output_dir.resolve()
    metrics, files = collect(output_dir)
    if not files:
        raise RuntimeError(f"No Phase-A result files found in {output_dir}")
    status_path = output_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    complete = status.get("stage") == "complete" and status.get("state") == "complete"

    os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
    import wandb

    run_id = hashlib.sha256(f"adeseg-phase-a:{args.version_id}".encode()).hexdigest()[:8]
    run = wandb.init(
        project="ade-seg", entity="adenoid-hypertrophy", id=run_id,
        resume="allow", name=f"latent-must-phase-a-v{args.version_id}-backfill",
        tags=["utility", "kaggle", "backfill", "not-live"],
        config={"kaggle_version_id": args.version_id, "tracking": "backfilled_from_saved_outputs"},
        dir=str(output_dir),
    )
    try:
        run.summary.update(metrics)
        run.summary["backfill/complete"] = complete
        run.summary["backfill/files"] = len(files)
        artifact = wandb.Artifact(f"latent-must-phase-a-v{args.version_id}", type="experiment-results")
        for path in files:
            artifact.add_file(str(path), name=path.relative_to(output_dir).as_posix())
        run.log_artifact(artifact)
    finally:
        run.finish()
    print(json.dumps({"wandb_run": run_id, "complete": complete, "files": len(files)}))


if __name__ == "__main__":
    main()

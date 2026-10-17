#!/usr/bin/env python3
"""Deliver live Kaggle W&B artifacts directly to Drive, without a local cache.

Works for an interactive notebook before Kaggle publishes a terminal version.
Kaggle's notebook/version export is a separate delivery checked by
collect_region_results.py after the session is saved.
"""

# ruff: noqa: E402 -- direct CLI use
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_region_results import (
    validate,
    validate_study_shard,
    verify_analysis_and_wandb,
)


def download_verified(artifact, destination):
    """Do not trust SDK cached checksums for files overwritten by other exports."""
    destination = Path(destination).resolve()
    for name, entry in artifact.manifest.entries.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or entry.ref is not None:
            raise ValueError(f"Unsupported artifact entry: {name}")

        def matches():
            return (
                path.is_file()
                and base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
                == entry.digest
            )

        if not matches():
            # Only replace this exact downloaded copy; the source remains in W&B.
            path.unlink(missing_ok=True)
            artifact.get_entry(name).download(root=str(destination), skip_cache=True)
        if not matches():
            raise RuntimeError(f"Artifact checksum mismatch after download: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--root-run-id",
        help="Optional known orchestrator ID. Avoids eventual-consistency delays in group search.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folder-name", default="region_tokens_v1")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    folder = args.output_root / args.folder_name
    folder.mkdir(parents=True, exist_ok=True)
    # Both artifact payloads and any SDK staging/cache live in the requested Drive.
    os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
    os.environ["WANDB_CACHE_DIR"] = str(folder / ".wandb-cache")
    os.environ["WANDB_DATA_DIR"] = str(folder / ".wandb-staging")
    import wandb

    downloaded = {}
    while True:
        api = wandb.Api(overrides={"base_url": os.environ["WANDB_BASE_URL"]})
        if args.root_run_id:
            root_run = api.run(f"adenoid-hypertrophy/ade-seg/{args.root_run_id}")
            runs = [root_run]
        else:
            runs = list(
                api.runs("adenoid-hypertrophy/ade-seg", filters={"group": args.group})
            )
            root_run = next(
                (run for run in runs if run.name == "region-tokens-orchestrator"), None
            )
        for run in runs:
            artifacts = list(run.logged_artifacts())
            if not artifacts:
                continue
            latest = max(artifacts, key=lambda artifact: artifact.created_at)
            if downloaded.get(run.id) == latest.digest:
                continue
            stage = run.config.get("stage")
            if run is root_run:
                destination = folder
            elif stage in ("E0", "E1", "E2_calibration", "E2_mean_smoke"):
                destination = folder / stage
            elif stage in (
                "E2",
                "E3",
                "E4_conditions",
                "E6_dev",
                "E6_test",
                "E6_baseline",
            ):
                destination = folder / "study"
            else:
                continue
            download_verified(latest, destination)
            downloaded[run.id] = latest.digest
            print(f"Delivered {run.name}: {latest.name} -> Drive", flush=True)
        snapshot = {
            "group": args.group,
            "updated_unix": time.time(),
            "runs": [
                {"id": r.id, "name": r.name, "state": r.state, "url": r.url}
                for r in runs
            ],
            "artifact_digests": downloaded,
            "kaggle_version_export_verified": False,
            "full_study_complete": False,
        }
        (folder / "live_tracking.json").write_text(
            json.dumps(snapshot, indent=2) + "\n"
        )
        print(json.dumps(snapshot["runs"]), flush=True)
        if root_run and root_run.state in ("finished", "failed", "crashed", "killed"):
            status = (
                json.loads((folder / "status.json").read_text())
                if (folder / "status.json").exists()
                else {}
            )
            if root_run.state == "finished" and status.get("stage") == "pilot_complete":
                verify_analysis_and_wandb(folder)
                (folder / "failure.txt").unlink(missing_ok=True)
                receipt = validate(folder)
                receipt["delivery_source"] = (
                    "W&B artifacts from Kaggle interactive execution"
                )
                receipt["kaggle_version_export_verified"] = False
                (folder / "delivery_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {k: v for k, v in receipt.items() if k != "sha256"}, indent=2
                    ),
                    flush=True,
                )
                if not receipt["delivery_valid"]:
                    raise SystemExit(1)
                return
            if root_run.state == "finished" and status.get("stage") in (
                "E2_calibration_complete",
                "E2_mean_smoke_complete",
                "E2_attention_smoke_complete",
                "E2_attention_primary_complete",
            ):
                receipt = {
                    "delivery_valid": True,
                    "delivery_source": "W&B artifacts from Kaggle interactive execution",
                    "full_study_complete": False,
                    "kaggle_status": status,
                }
                (folder / "delivery_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                return
            if (
                root_run.state == "finished"
                and status.get("stage") == "study_shard_complete"
            ):
                receipt = validate_study_shard(folder)
                receipt["delivery_source"] = (
                    "W&B artifacts downloaded directly to Google Drive"
                )
                (folder / "delivery_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                if not receipt["delivery_valid"]:
                    raise SystemExit(
                        "Study shard delivery failed validation; inspect delivery_receipt.json"
                    )
                return
            raise SystemExit(
                f"Kaggle orchestrator ended with {root_run.state}; inspect exported failure/status"
            )
        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

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
import html
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_region_results import (
    validate,
    validate_e0_delivery,
    validate_study_shard,
    verify_analysis_and_wandb,
)
from training.tracking import latest_results_artifact


def write_study_summary(output_root):
    """Summarize saved evidence, never infer full coverage from one finished shard."""
    output_root = Path(output_root)
    stages = {
        f"E{i}": {"experiment": title, "status": "Not recorded", "evidence": []}
        for i, title in enumerate(
            (
                "Exactness",
                "Pilot",
                "Main interventions",
                "Corrupted memory",
                "Conditions",
                "Attention versus intervention",
                "Method evaluation",
            )
        )
    }
    receipt = output_root / "region_tokens_execution_e0_v2/delivery_receipt.json"
    if receipt.is_file() and json.loads(receipt.read_text()).get("delivery_valid"):
        stages["E0"].update(
            status="Passed; delivery verified",
            evidence=[str(receipt.relative_to(output_root))],
        )
    decision = output_root / "region_tokens_v1/E1/decision_verified.json"
    if decision.is_file():
        data = json.loads(decision.read_text())
        passed = (
            data.get("gate_pass") is True and data.get("all_required_sequences") is True
        )
        stages["E1"].update(
            status="Completed: proceed to E2"
            if passed
            else "STOP: pilot gate not passed",
            evidence=[str(decision.relative_to(output_root))],
        )
    components, analyses, runs = [], [], []
    for folder in sorted(output_root.glob("region_tokens_*")):
        live = folder / "live_tracking.json"
        if live.is_file():
            snapshot = json.loads(live.read_text())
            runs.extend(
                {**run, "observed_unix": snapshot.get("updated_unix")}
                for run in snapshot.get("runs", [])
            )
        study = folder / "study"
        if not study.is_dir():
            continue
        for path in sorted(study.glob("*/status.json")):
            data = json.loads(path.read_text())
            stage = data.get("stage", "")
            key = "E6" if stage.startswith("E6") else stage
            if key not in stages or data.get("state") != "complete":
                continue
            entry = {
                "stage": stage,
                "unit": path.parent.name,
                "rows": data.get("rows"),
                "fresh_e0_pass": data.get("fresh_e0_pass"),
                "source": str(path.relative_to(output_root)),
            }
            components.append(entry)
            stages[key]["status"] = "Partial evidence; full matrix not verified"
            stages[key]["evidence"].append(entry["source"])
        baseline = study / "status.json"
        if baseline.is_file():
            data = json.loads(baseline.read_text())
            if data.get("stage") == "E6_baseline" and data.get("state") == "complete":
                stages["E6"]["status"] = (
                    "Partial baseline evidence; full matrix not verified"
                )
                stages["E6"]["evidence"].append(str(baseline.relative_to(output_root)))
        conditions = study / "conditions.jsonl"
        if conditions.is_file():
            rows = [
                json.loads(line) for line in conditions.read_text().splitlines() if line
            ]
            stages["E4"]["status"] = (
                "Partial condition extraction; interaction analysis pending"
            )
            stages["E4"]["evidence"].append(str(conditions.relative_to(output_root)))
            components.append(
                {
                    "stage": "E4_conditions",
                    "rows": len(rows),
                    "sequences": len({(r["dataset"], r["video"]) for r in rows}),
                    "source": str(conditions.relative_to(output_root)),
                }
            )
        for path in [study / "analysis.json", *folder.glob("interim_*/analysis.json")]:
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            source = str(path.relative_to(output_root))
            if data.get("E5", {}).get("agreement"):
                stages["E5"].update(
                    status="Interim/component analysis; full coverage not verified"
                )
                stages["E5"]["evidence"].append(source)
            for comparison in data.get("E2", {}).values():
                arm = comparison["arm"]
                if (
                    arm[3] == "recent"
                    and arm[-3] == "all"
                    and arm[-1] == "random"
                    and arm[-2] in ("dice", "boundary_f")
                ):
                    analyses.append(
                        {
                            "scope": arm,
                            "n_sequences": comparison["n_sequences"],
                            "mean_excess_harm": comparison["mean_excess_harm"],
                            "source": source,
                            "interpretation": "Descriptive only"
                            if comparison["n_sequences"] < 2
                            else "Component comparison; not a full-study decision",
                        }
                    )
    payload = {
        "updated": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "full_study_complete": False,
        "automatic_submission_enabled": False,
        "monitor_requires": "Mac awake, this watcher running, network and Drive access",
        "stages": stages,
        "components": components,
        "observed_runs": runs,
        "reported_comparisons": analyses,
        "warning": "Not recorded does not establish not run. Completed components do not establish complete registered coverage. No hypothesis decision is made by this status table.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{key}</td><td>{html.escape(value['experiment'])}</td><td>{html.escape(value['status'])}</td><td>{len(value['evidence'])}</td></tr>"
        for key, value in stages.items()
    )
    comparisons = "".join(
        f"<tr><td>{html.escape(json.dumps(value['scope']))}</td><td>{value['n_sequences']}</td><td>{value['mean_excess_harm']:.6g}</td><td>{html.escape(value['interpretation'])}</td></tr>"
        for value in analyses
    )
    live_rows = "".join(
        f"<tr><td>{html.escape(run['name'])}</td><td>{html.escape(run['state'])}</td><td>{html.escape(str(run.get('observed_unix')))}</td></tr>"
        for run in runs
    )
    document = f"<!doctype html><meta charset='utf-8'><title>AdeSEG study status</title><style>body{{font:16px system-ui;max-width:1100px;margin:2em auto}}td,th{{border:1px solid #ccc;padding:.5em}}table{{border-collapse:collapse}}small{{color:#555}}</style><h1>AdeSEG: verified evidence status</h1><p>Updated {html.escape(payload['updated'])}</p><p>{html.escape(payload['warning'])}</p><table><tr><th>Stage</th><th>Experiment</th><th>Status</th><th>Evidence files</th></tr>{rows}</table><h2>Observed W&amp;B runs</h2><p>States are snapshots at the observation timestamp, not guaranteed live.</p><table><tr><th>Run</th><th>Observed state</th><th>Observed Unix time</th></tr>{live_rows}</table><h2>Recorded component comparisons</h2><p>Excess harm: boundary-token intervention minus matched random-control harm; fractional score units. These are not full-study findings.</p><table><tr><th>Scope</th><th>Independent sequences</th><th>Mean excess harm</th><th>Interpretation</th></tr>{comparisons}</table><p>Automatic submission: NOT enabled. Monitoring requires the Mac to remain awake and this watcher to remain running.</p>"
    for name, content in (
        ("region_study_summary.json", json.dumps(payload, indent=2) + "\n"),
        ("region_study_summary.html", document),
    ):
        path = output_root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return payload


def download_verified(artifact, destination, names=None):
    """Do not trust SDK cached checksums for files overwritten by other exports."""
    destination = Path(destination).resolve()
    if names is not None and set(names) - artifact.manifest.entries.keys():
        raise ValueError("Requested file is missing from artifact manifest")
    for name, entry in artifact.manifest.entries.items():
        if names is not None and name not in names:
            continue
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
    parser.add_argument("--group")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--root-run-id",
        help="Optional known orchestrator ID. Avoids eventual-consistency delays in group search.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folder-name", default="region_tokens_v1")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        while True:
            try:
                result = write_study_summary(args.output_root)
                print(f"Study summary updated: {result['updated']}", flush=True)
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                # A collector can be replacing a file while it is read. Preserve
                # the last complete report, with its original update timestamp.
                if not args.watch:
                    raise
                print(
                    f"Summary refresh failed; previous report retained: {exc}",
                    flush=True,
                )
            if not args.watch:
                return
            time.sleep(30)
    if not args.group:
        parser.error("--group is required unless --summary-only is used")
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
            runs = [
                root_run,
                *[
                    child
                    for child in api.runs(
                        "adenoid-hypertrophy/ade-seg", filters={"group": args.group}
                    )
                    if child.id != root_run.id
                ],
            ]
        else:
            runs = list(
                api.runs("adenoid-hypertrophy/ade-seg", filters={"group": args.group})
            )
            root_run = next(
                (
                    run
                    for run in runs
                    if run.config.get("role") == "orchestrator"
                    or run.name == "region-tokens-orchestrator"
                ),
                None,
            )
        for run in runs:
            try:
                latest = latest_results_artifact(run)
            except ValueError:
                continue
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
            if root_run.state == "finished" and status.get("stage") == "E0_complete":
                receipt = validate_e0_delivery(folder)
                receipt["delivery_source"] = (
                    "W&B artifacts downloaded directly to Google Drive"
                )
                (folder / "delivery_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                if not receipt["delivery_valid"]:
                    raise SystemExit("Standalone E0 delivery failed validation")
                return
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

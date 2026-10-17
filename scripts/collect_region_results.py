#!/usr/bin/env python3
"""Wait for Kaggle completion, export directly to Drive and verify delivery."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def validate(folder):
    problems = []
    status_path = folder / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("state") != "complete":
        problems.append("Kaggle pipeline did not complete")
    tracking_path = folder / "tracking.json"
    tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {}
    if tracking.get("mode") != "online" or not tracking.get("run_id"):
        problems.append("Missing online orchestrator provenance")
    for stage in ("E0", "E1"):
        stage_path = folder / stage
        stage_status = (
            json.loads((stage_path / "status.json").read_text())
            if (stage_path / "status.json").exists()
            else {}
        )
        if (
            stage_status.get("state") != "complete"
            or stage_status.get("e0_pass") is not True
        ):
            problems.append(f"{stage} incomplete or sanity failed")
        if not (stage_path / "tracking.json").exists():
            problems.append(f"{stage} missing tracking metadata")
        else:
            stage_tracking = json.loads((stage_path / "tracking.json").read_text())
            if stage_tracking.get("mode") != "online":
                problems.append(f"{stage} was not online tracked")
    frame_path = folder / "E1/frames.jsonl"
    rows = (
        [
            json.loads(line)
            for line in frame_path.read_text().splitlines()
            if line.strip()
        ]
        if frame_path.exists()
        else []
    )
    stage_status_path = folder / "E1/status.json"
    if stage_status_path.exists() and len(rows) != json.loads(
        stage_status_path.read_text()
    ).get("rows"):
        problems.append("Recorded E1 row count does not match output")
    if not rows:
        problems.append("No E1 records")
    decision_path = folder / "E1/decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    if not decision.get("all_required_sequences"):
        problems.append("Incomplete E1 sequence coverage")
    # Hash delivered data, not volatile W&B backend files.
    files = {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.rglob("*")
        if path.is_file()
        and "wandb" not in path.relative_to(folder).parts
        and path.name != "delivery_receipt.json"
    }
    receipt = {
        "delivery_valid": not problems,
        "full_study_complete": False,
        "problems": problems,
        "kaggle_status": status,
        "tracking": tracking,
        "E1_rows": len(rows),
        "E1_decision": decision.get("decision"),
        "sha256": files,
    }
    (folder / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="bunnoob2005/adeseg-region-memory-tokens")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        status = subprocess.check_output(
            ["kaggle", "kernels", "status", args.kernel], text=True
        )
        print(status.strip(), flush=True)
        if not any(word in status.upper() for word in ("RUNNING", "QUEUED", "PENDING")):
            break
        if not args.wait:
            return
        time.sleep(30)
    args.output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "kernels",
            "output",
            args.kernel,
            "--path",
            str(args.output_root),
            "--force",
        ],
        check=True,
    )
    receipt = validate(args.output_root / "region_tokens_v1")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "sha256"}, indent=2
        ),
        flush=True,
    )
    if not receipt["delivery_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

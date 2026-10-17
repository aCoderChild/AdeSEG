#!/usr/bin/env python3
"""Validate downloaded Phase-A files and write a delivery receipt in Drive."""

import argparse
import json
from pathlib import Path

EXPECTED_STAGES = {
    "E1_E2/natural_failures", "E1_E2/c0_matched_failures",
    *(f"E3/{name}" for name in ("natural", "c0", "c1", "c2", "c3", "c4")),
}


def inspect(root, require_online=False):
    root = Path(root)
    problems = []
    root_status = root / "status.json"
    if not root_status.is_file():
        problems.append("Missing root status.json")
        status = {}
    else:
        status = json.loads(root_status.read_text())
        if status.get("stage") != "complete" or status.get("state") != "complete":
            problems.append("Kaggle runner did not reach complete status")
    root_tracking_path = root / "tracking.json"
    root_tracking = json.loads(root_tracking_path.read_text()) if root_tracking_path.is_file() else {}
    if require_online and (root_tracking.get("mode") != "online" or not root_tracking.get("run_id")):
        problems.append("Live W&B orchestrator run not recorded")

    stages = {}
    setups = sorted(root.rglob("setup.json"))
    found = {setup.parent.relative_to(root).as_posix() for setup in setups}
    for name in sorted(EXPECTED_STAGES - found):
        problems.append(f"Missing experiment stage: {name}")
    for name in ("E1_E2/taxonomy.json", "E3/mechanism.json", "events.jsonl"):
        if not (root / name).is_file():
            problems.append(f"Missing result: {name}")
    for setup in setups:
        directory = setup.parent
        name = directory.relative_to(root).as_posix()
        required = ("frames.jsonl", "utility_records.pt", "summary.json", "status.json")
        missing = [item for item in required if not (directory / item).is_file()]
        if missing:
            problems.append(f"{name}: missing {', '.join(missing)}")
            continue
        stage_status = json.loads((directory / "status.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        with (directory / "frames.jsonl").open() as stream:
            frame_count = sum(bool(line.strip()) for line in stream)
        if stage_status.get("state") != "complete":
            problems.append(f"{name}: stage is not complete")
        if frame_count != summary.get("analyzed_frames") or frame_count != stage_status.get("completed_rows"):
            problems.append(f"{name}: recorded frame counts disagree")
        if frame_count == 0:
            problems.append(f"{name}: no analyzed frames")
        if summary.get("utility_units") != stage_status.get("utility_units"):
            problems.append(f"{name}: utility unit counts disagree")
        if (directory / "utility_records.pt").stat().st_size == 0:
            problems.append(f"{name}: empty utility_records.pt")
        tracking_path = directory / "tracking.json"
        tracking = json.loads(tracking_path.read_text()) if tracking_path.is_file() else {}
        if require_online and (tracking.get("mode") != "online" or not tracking.get("run_id")):
            problems.append(f"{name}: live W&B run not recorded")
        stages[name] = {"frames": frame_count, "utility_units": summary.get("utility_units"),
                        "wandb_mode": tracking.get("mode", "unrecorded"),
                        "wandb_run_id": tracking.get("run_id")}

    return {"complete": not problems, "runner_status": status.get("stage"),
            "orchestrator_wandb_run_id": root_tracking.get("run_id"),
            "all_online": bool(stages) and all(
                item["wandb_mode"] == "online" and item["wandb_run_id"] for item in stages.values()
            ) and root_tracking.get("mode") == "online" and bool(root_tracking.get("run_id")),
            "stages": stages, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-online", action="store_true")
    args = parser.parse_args()
    result = inspect(args.output_dir, args.require_online)
    receipt = args.output_dir / "delivery_receipt.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

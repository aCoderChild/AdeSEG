#!/usr/bin/env python3
"""Wait for Kaggle completion, export directly to Drive and verify delivery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_study_shard(folder):
    """Validate component delivery without applying the E1-only report schema."""
    folder = Path(folder)
    problems = []
    for relative in (
        "status.json",
        "tracking.json",
        "study/status.json",
        "study/tracking.json",
    ):
        if not (folder / relative).is_file():
            problems.append(f"Missing {relative}")
    if problems:
        return {
            "delivery_valid": False,
            "problems": problems,
            "full_study_complete": False,
        }
    status = json.loads((folder / "study/status.json").read_text())
    if status.get("state") != "complete":
        problems.append("Study component is incomplete")
    for relative in ("tracking.json", "study/tracking.json"):
        tracking = json.loads((folder / relative).read_text())
        if tracking.get("mode") != "online" or not tracking.get("run_id"):
            problems.append(f"Invalid online tracking: {relative}")
    for name in status.get("completed_units", []):
        unit = folder / "study" / name
        if not unit.resolve().is_relative_to((folder / "study").resolve()):
            problems.append("Unsafe unit path")
            continue
        unit_status = json.loads((unit / "status.json").read_text())
        if (
            unit_status.get("state") != "complete"
            or unit_status.get("fresh_e0_pass") is not True
        ):
            problems.append(f"Missing fresh exactness checks: {name}")
        frames = unit / "frames.jsonl"
        if frames.is_file():
            rows = [
                json.loads(line) for line in frames.read_text().splitlines() if line
            ]
            if len(rows) != unit_status.get("rows"):
                problems.append(f"Recorded row count differs: {name}")
            targets = {}
            for row in rows:
                key = (row["bank"], row["category"], row["seed"])
                target = targets.setdefault(row["frame"], set())
                if key in target:
                    problems.append(f"Repeated target/arm: {name}:{row['frame']}")
                target.add(key)
            if any(len(arms) != 28 for arms in targets.values()):
                problems.append(f"Incomplete matched-control matrix: {name}")
        rollout = unit / "rollout.jsonl"
        if rollout.is_file():
            rows = [
                json.loads(line) for line in rollout.read_text().splitlines() if line
            ]
            if len(rows) != unit_status.get("rows"):
                problems.append(f"Recorded rollout row count differs: {name}")
            arms = {}
            for row in rows:
                key = (row["category"], row.get("bank"), row.get("seed", 0))
                frames = arms.setdefault(key, set())
                if row["frame"] in frames:
                    problems.append(f"Repeated rollout frame: {name}")
                frames.add(row["frame"])
            baseline = arms.get(("baseline", None, 0))
            if baseline is None or any(frames != baseline for frames in arms.values()):
                problems.append(
                    f"Rollout arms do not cover the same baseline frames: {name}"
                )
    files = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*.json*")
        if not any(
            s.startswith(".wandb") or s == "wandb" for s in p.relative_to(folder).parts
        )
    }
    return {
        "delivery_valid": not problems,
        "problems": problems,
        "sha256": files,
        "stage": status.get("stage"),
        "full_study_complete": False,
        "kaggle_version_export_verified": False,
    }


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
    protocol_path = folder / "protocol.json"
    manifest_path = folder / "polypgen_manifest.jsonl"
    if protocol_path.exists() and manifest_path.exists():
        from scripts.audit_region_initialization import target_indices

        protocol = json.loads(protocol_path.read_text())
        counts = {}
        for line in manifest_path.read_text().splitlines():
            row = json.loads(line)
            counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
        expected = set()
        for video in protocol["pilot"]["sequences"]:
            audit_path = folder / "initialization_audit.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            offsets = {
                r["video"]: r["first_visible_frame"] for r in audit.get("sequences", [])
            }
            prompt = (
                offsets.get(video)
                if protocol.get("initialization") == "first_labeled_frame"
                else 0
            )
            if prompt is None:
                problems.append(f"Missing prompt eligibility audit: {video}")
                continue
            try:
                targets = target_indices(
                    counts.get(video, 0),
                    prompt,
                    protocol["pilot"]["targets_per_sequence"],
                )
            except ValueError:
                problems.append(f"No eligible targets: {video}")
                continue
            for frame in targets:
                for bank in ("conditioning", "recent"):
                    for category in (
                        "boundary",
                        "foreground",
                        "near_background",
                        "far_background",
                        "random",
                        "spatial_random",
                    ):
                        for seed in (
                            range(5)
                            if category in ("random", "spatial_random")
                            else (0,)
                        ):
                            expected.add((video, int(frame), bank, category, seed))
        actual = [
            (r["video"], r["frame"], r["bank"], r["category"], r["seed"]) for r in rows
        ]
        if set(actual) != expected or len(actual) != len(set(actual)):
            problems.append(
                "E1 targets/controls do not match preregistered sample grid"
            )
    else:
        problems.append("Missing protocol or input manifest")
    decision_path = folder / "E1/decision_verified.json"
    if not decision_path.exists():
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


def verify_analysis_and_wandb(folder):
    """Recompute the gate with the reviewed analyzer; retain original output."""
    root = Path(__file__).resolve().parents[1]
    verified = folder / "E1/decision_verified.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_region_token_pilot.py"),
            "--input",
            str(folder / "E1/frames.jsonl"),
            "--protocol",
            str(folder / "protocol.json"),
            "--output",
            str(verified),
        ],
        check=True,
    )
    decision = json.loads(verified.read_text())
    decision["analysis_git_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    decision["original_decision_sha256"] = hashlib.sha256(
        (folder / "E1/decision.json").read_bytes()
    ).hexdigest()
    verified.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    # Public Run.upload_file constructs an InternalApi; it also needs the
    # environment base URL, not just the public Api(overrides=...) setting.
    os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
    import wandb

    api = wandb.Api(overrides={"base_url": "https://forge.coreweave.com/api/wandb"})
    for relative in ("tracking.json", "E0/tracking.json", "E1/tracking.json"):
        tracking = json.loads((folder / relative).read_text())
        remote = api.run(f"adenoid-hypertrophy/ade-seg/{tracking['run_id']}")
        if remote.state != "finished":
            raise RuntimeError(
                f"W&B run {tracking['run_id']} is {remote.state}, expected finished"
            )
    orchestrator = api.run(
        f"adenoid-hypertrophy/ade-seg/{json.loads((folder / 'tracking.json').read_text())['run_id']}"
    )
    orchestrator.upload_file(str(verified), root=str(folder))
    orchestrator.summary["E1/verified_gate_pass"] = decision["gate_pass"]
    orchestrator.summary["E1/verified_decision"] = decision["decision"]
    orchestrator.summary["analysis_git_revision"] = decision["analysis_git_revision"]
    orchestrator.summary.update()


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
    folder = args.output_root / "region_tokens_v1"
    status_path = folder / "status.json"
    if (
        status_path.exists()
        and json.loads(status_path.read_text()).get("state") == "complete"
    ):
        verify_analysis_and_wandb(folder)
        # A prior failed bootstrap can have left this file during --force export.
        (folder / "failure.txt").unlink(missing_ok=True)
    receipt = validate(folder)
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

"""Run pinned author DAVIS comparators after local Run 159 completes."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.memory_token_io import digest, validate_output


PROTOCOL = ROOT / "protocols/davis_region_vs_temporal_selection_v1.json"
RUNNER = ROOT / "scripts/run_davis_author_vos_comparator.py"
SUMMARIZER = ROOT / "scripts/summarize_davis_region_vs_selection.py"
METHODS = ("samurai", "sam2long")


def process_command(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def update_status(path, status, **changes):
    status.update(changes)
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, status)


def verify_frozen_protocol(source_path, snapshot_path, expected_sha256):
    if digest(source_path) != expected_sha256 or digest(snapshot_path) != expected_sha256:
        raise RuntimeError("DAVIS comparison protocol changed after the queue was frozen")


def run_stage(command, log_path, status_path, status, state, progress_path=None):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        update_status(status_path, status, state=state, active_pid=process.pid,
                      active_log=str(log_path))
        while process.poll() is None:
            changes = {}
            if progress_path is not None and progress_path.is_file():
                changes["stage_progress"] = json.loads(progress_path.read_text())
            update_status(status_path, status, **changes)
            time.sleep(60)
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resource-pid", type=int, required=True)
    parser.add_argument("--resource-command", required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    root = validate_output(args.output_root)
    if not args.dataset_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("DAVIS root or SAM2.1 Tiny checkpoint is missing")
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite DAVIS comparison output: {root}")
    root.mkdir(parents=True)
    protocol_sha256 = digest(PROTOCOL)
    protocol_snapshot = root / "protocol_frozen.json"
    protocol_snapshot.write_bytes(PROTOCOL.read_bytes())
    verify_frozen_protocol(PROTOCOL, protocol_snapshot, protocol_sha256)
    gate_root, full_root, summary_root = root / "gate", root / "full", root / "summary"
    status_path = root / "queue.json"
    status = {
        "state": "waiting_for_local_davis_run",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256,
        "protocol_snapshot": str(protocol_snapshot),
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "local_run_dir": str(args.local_run_dir.resolve()),
        "output_root": str(root),
        "resource_pid": args.resource_pid,
        "resource_command": args.resource_command,
        "resource_status": str(args.resource_status.resolve()),
    }
    write_json(status_path, status)

    active = process_command(args.resource_pid)
    while active:
        if args.resource_command not in active:
            update_status(status_path, status, state="resource_wait_failed",
                          error=f"PID changed identity: {active}")
            return
        resource_state = None
        if args.resource_status.is_file():
            resource_state = json.loads(args.resource_status.read_text()).get("state")
        update_status(status_path, status, resource_state=resource_state)
        time.sleep(args.poll_seconds)
        active = process_command(args.resource_pid)

    if not args.resource_status.is_file():
        update_status(status_path, status, state="resource_status_missing",
                      error="Run 159 queue status file is missing")
        return
    resource = json.loads(args.resource_status.read_text())
    if resource.get("state") != "complete":
        update_status(status_path, status, state="local_davis_run_not_complete",
                      local_final_state=resource.get("state"),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        return
    local_run = args.local_run_dir.resolve()
    if not (local_run / "results.json").is_file():
        update_status(status_path, status, state="local_results_missing",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        return

    common = [
        sys.executable, str(RUNNER),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
    ]
    for method in METHODS:
        try:
            verify_frozen_protocol(PROTOCOL, protocol_snapshot, protocol_sha256)
        except RuntimeError as error:
            update_status(status_path, status, state="protocol_changed_before_inference",
                          error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
            return
        gate_dir = gate_root / method
        code = run_stage(
            common + ["--method", method, "--stage", "gate", "--output-dir", str(gate_dir)],
            root / f"{method}_gate.log", status_path, status,
            f"{method}_engineering_gate", gate_dir / "progress.json",
        )
        receipt_path = gate_dir / f"{method}_gate.json"
        if code != 0 or not receipt_path.is_file():
            failure_path = gate_dir / "failure.json"
            update_status(status_path, status, state=f"{method}_gate_failed",
                          exit_code=code,
                          failure=json.loads(failure_path.read_text()) if failure_path.is_file() else None,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            return
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("passed") is not True or receipt.get("official_evaluation_accepted") is not True:
            update_status(status_path, status, state=f"{method}_gate_failed",
                          error="Engineering gate or official evaluator did not pass",
                          finished_utc=datetime.now(timezone.utc).isoformat())
            return

    for method in METHODS:
        try:
            verify_frozen_protocol(PROTOCOL, protocol_snapshot, protocol_sha256)
        except RuntimeError as error:
            update_status(status_path, status, state="protocol_changed_before_inference",
                          error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
            return
        command = common + [
            "--method", method,
            "--stage", "full",
            "--local-run-dir", str(local_run),
            "--output-dir", str(full_root),
        ]
        code = run_stage(
            command, root / f"{method}_full.log", status_path, status,
            f"{method}_full_running", full_root / "progress.json",
        )
        summary_path = full_root / f"{method}_full_summary.json"
        if code != 0 or not summary_path.is_file():
            update_status(status_path, status, state=f"{method}_full_failed",
                          exit_code=code,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            return

    try:
        verify_frozen_protocol(PROTOCOL, protocol_snapshot, protocol_sha256)
    except RuntimeError as error:
        update_status(status_path, status, state="protocol_changed_before_summary",
                      error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        return
    summary_code = run_stage(
        [
            sys.executable, str(SUMMARIZER),
            "--dataset-root", str(args.dataset_root.resolve()),
            "--local-run-dir", str(local_run),
            "--author-full-dir", str(full_root),
            "--output-dir", str(summary_root),
        ],
        root / "summary.log", status_path, status, "summarizing_results",
    )
    complete = summary_code == 0 and (summary_root / "comparison.json").is_file()
    update_status(status_path, status,
                  state="complete" if complete else "summary_failed",
                  summary_exit_code=summary_code,
                  results_file=str(summary_root / "comparison.md") if complete else None,
                  finished_utc=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()

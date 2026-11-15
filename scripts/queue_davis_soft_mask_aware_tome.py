"""Queue the DAVIS ToMe gate and val run after the active PolypGen queue."""

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

from scripts.memory_token_io import validate_output


RUNNER = ROOT / "scripts/run_davis_soft_mask_aware_tome.py"


def process_command(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def write_status(path, value):
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
    write_status(path, status)


def run_stage(command, log_path, status_path, status, state_name, progress_path=None):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        update_status(status_path, status, state=state_name, active_pid=process.pid,
                      active_log=str(log_path))
        while process.poll() is None:
            details = {}
            if progress_path is not None and progress_path.is_file():
                details["stage_progress"] = json.loads(progress_path.read_text())
            update_status(status_path, status, **details)
            time.sleep(60)
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resource-pid", type=int, required=True)
    parser.add_argument("--resource-command", required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    output_root = validate_output(args.output_root)
    if not args.dataset_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("DAVIS root or SAM2 checkpoint is missing")
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")
    gate_dir, full_dir = output_root / "gate", output_root / "full"
    if output_root.exists() or gate_dir.exists() or full_dir.exists():
        raise FileExistsError(f"Refusing to overwrite DAVIS output: {output_root}")
    output_root.mkdir(parents=True)
    status_path = output_root / "queue.json"
    gate_log, full_log = output_root / "gate.log", output_root / "full.log"

    active_command = process_command(args.resource_pid)
    if active_command and args.resource_command not in active_command:
        raise RuntimeError("The resource PID does not match the expected active queue")
    status = {
        "state": "waiting_for_polypgen_queue",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resource_pid": args.resource_pid,
        "resource_command": args.resource_command,
        "resource_status": str(args.resource_status.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_root": str(output_root),
    }
    write_status(status_path, status)

    while active_command:
        if args.resource_command not in active_command:
            update_status(status_path, status, state="resource_wait_failed",
                          error=f"PID changed identity: {active_command}")
            return
        time.sleep(args.poll_seconds)
        active_command = process_command(args.resource_pid)
        update_status(status_path, status)

    if not args.resource_status.is_file():
        update_status(status_path, status, state="resource_status_missing",
                      error="PolypGen queue status file is missing")
        return
    resource_status = json.loads(args.resource_status.read_text())
    if resource_status.get("state") != "complete":
        update_status(status_path, status, state="resource_run_not_complete",
                      resource_final_state=resource_status.get("state"))
        return

    common = [
        sys.executable, str(RUNNER),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
    ]
    gate_command = common + ["--stage", "gate", "--output-dir", str(gate_dir)]
    gate_code = run_stage(gate_command, gate_log, status_path, status, "davis_engineering_gate")
    gate_path = gate_dir / "gate.json"
    if gate_code != 0 or not gate_path.is_file():
        failure_path = gate_dir / "failure.json"
        update_status(status_path, status, state="davis_gate_failed", gate_exit_code=gate_code,
                      gate_failure=json.loads(failure_path.read_text()) if failure_path.is_file() else None,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        return
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True:
        update_status(status_path, status, state="davis_gate_failed",
                      error="Gate completed without a passing receipt",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        return

    full_command = common + [
        "--stage", "full", "--gate-receipt", str(gate_path), "--output-dir", str(full_dir),
    ]
    full_code = run_stage(
        full_command, full_log, status_path, status, "davis_full_running",
        progress_path=full_dir / "progress.json",
    )
    results_path = full_dir / "results.json"
    update_status(status_path, status,
                  state="complete" if full_code == 0 and results_path.is_file() else "davis_full_failed",
                  full_exit_code=full_code,
                  results_file=str(results_path) if results_path.is_file() else None,
                  finished_utc=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()

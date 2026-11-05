"""Wait for the active Run 155 process, then launch the soft-ToMe gate on Drive."""

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


def write_status(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_command(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-script", required=True)
    parser.add_argument("--wait-output", type=Path, required=True)
    parser.add_argument(
        "--wait-extra",
        nargs=2,
        action="append",
        default=[],
        metavar=("PID", "COMMAND_SUBSTRING"),
        help="Also wait for a process and verify its command contains the given substring; repeat as needed.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    gate_output = validate_output(args.gate_output)
    queue_root = gate_output.parent
    queue_root.mkdir(parents=True, exist_ok=True)
    status_path = queue_root / "queue.json"
    log_path = queue_root / "gate_launcher.log"
    if gate_output.exists():
        raise FileExistsError(f"Refusing to overwrite queued gate output: {gate_output}")
    if not args.checkpoint.is_file() or not args.native_run.is_dir():
        raise FileNotFoundError("Checkpoint or archived native reference is missing")
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")

    expected_path = str(args.wait_output.resolve())
    wait_targets = [(args.wait_pid, (args.wait_script, expected_path))]
    for pid_text, command_substring in args.wait_extra:
        wait_targets.append((int(pid_text), (command_substring,)))

    def active_wait_targets():
        active = []
        for pid, expected_parts in wait_targets:
            command = process_command(pid)
            if command and not all(part in command for part in expected_parts):
                raise RuntimeError(
                    f"PID {pid} is active but does not match the expected command "
                    f"{expected_parts!r}: {command}"
                )
            if command:
                active.append(pid)
        return active

    active_pids = active_wait_targets()

    status = {
        "state": "waiting" if active_pids else "launching",
        "wait_pid": args.wait_pid,
        "wait_script": args.wait_script,
        "wait_output": expected_path,
        "additional_waits": [
            {"pid": int(pid), "command_substring": substring}
            for pid, substring in args.wait_extra
        ],
        "active_wait_pids": active_pids,
        "checkpoint": str(args.checkpoint.resolve()),
        "native_run": str(args.native_run.resolve()),
        "gate_output": str(gate_output),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_status(status_path, status)

    while active_pids:
        time.sleep(args.poll_seconds)
        try:
            active_pids = active_wait_targets()
        except RuntimeError as exc:
            status.update({
                "state": "failed",
                "error": str(exc),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_status(status_path, status)
            raise RuntimeError(status["error"])
        status.update({
            "state": "waiting" if active_pids else "launching",
            "active_wait_pids": active_pids,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_status(status_path, status)

    command = [
        sys.executable,
        str(ROOT / "scripts/run_soft_mask_aware_tome_gate.py"),
        "--checkpoint", str(args.checkpoint),
        "--native-run", str(args.native_run),
        "--output-dir", str(gate_output),
    ]
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    status.update({
        "state": "complete" if result.returncode == 0 else "failed",
        "gate_exit_code": result.returncode,
        "gate_log": str(log_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_status(status_path, status)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

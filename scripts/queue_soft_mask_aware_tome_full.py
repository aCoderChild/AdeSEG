"""Launch the full soft-ToMe cohort only after its seq2/seq3 gate passes."""

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


RUNNER = "scripts/run_soft_mask_aware_tome_full.py"


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
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-command-substring")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    full_output = validate_output(args.full_output)
    root = full_output.parent
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue.json"
    log_path = root / "full_launcher.log"
    if full_output.exists():
        raise FileExistsError(f"Refusing to overwrite full-run output: {full_output}")
    if not args.checkpoint.is_file() or not args.native_run.is_dir():
        raise FileNotFoundError("Checkpoint or archived native reference is missing")
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")
    if (args.wait_pid is None) != (args.wait_command_substring is None):
        raise ValueError("--wait-pid and --wait-command-substring must be supplied together")

    wait_command = process_command(args.wait_pid) if args.wait_pid is not None else None
    if wait_command and args.wait_command_substring not in wait_command:
        raise RuntimeError(
            f"PID {args.wait_pid} is active but does not match the expected resource process: "
            f"{wait_command}"
        )

    status = {
        "state": "waiting_for_gate",
        "gate_dir": str(args.gate_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "native_run": str(args.native_run.resolve()),
        "full_output": str(full_output),
        "runner": RUNNER,
        "wait_pid": args.wait_pid,
        "wait_command_substring": args.wait_command_substring,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_status(status_path, status)

    while True:
        failure = args.gate_dir / "failure.json"
        completion = args.gate_dir / "completion.json"
        gate = args.gate_dir / "gate.json"
        if failure.is_file() and not completion.is_file():
            status.update({
                "state": "gate_failed",
                "gate_failure": json.loads(failure.read_text()),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_status(status_path, status)
            return
        if completion.is_file() and gate.is_file():
            gate_completion = json.loads(completion.read_text())
            gate_result = json.loads(gate.read_text())
            if not gate_completion.get("passed") or not gate_result.get("passed"):
                status.update({
                    "state": "gate_failed",
                    "gate_completion": gate_completion,
                    "gate_result": gate_result,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                })
                write_status(status_path, status)
                return
            break
        time.sleep(args.poll_seconds)
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_status(status_path, status)

    while wait_command:
        status.update({
            "state": "waiting_for_resource_process",
            "active_wait_pid": args.wait_pid,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_status(status_path, status)
        time.sleep(args.poll_seconds)
        wait_command = process_command(args.wait_pid)
        if wait_command and args.wait_command_substring not in wait_command:
            status.update({
                "state": "resource_wait_failed",
                "error": f"PID {args.wait_pid} changed identity while waiting: {wait_command}",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_status(status_path, status)
            return

    status.update({"state": "full_run_starting", "updated_utc": datetime.now(timezone.utc).isoformat()})
    write_status(status_path, status)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_soft_mask_aware_tome_full.py"),
        "--checkpoint", str(args.checkpoint),
        "--native-run", str(args.native_run),
        "--gate-dir", str(args.gate_dir),
        "--output-dir", str(full_output),
    ]
    status["state"] = "full_run_running"
    status["full_log"] = str(log_path)
    write_status(status_path, status)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    status.update({
        "state": "complete" if result.returncode == 0 else "full_run_failed",
        "full_exit_code": result.returncode,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_status(status_path, status)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

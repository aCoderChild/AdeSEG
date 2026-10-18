"""Test the notebook helper without executing Kaggle bootstrap or secrets."""

import ast
from pathlib import Path
import selectors
import signal
import os
import subprocess
import sys
import time

import pytest


@pytest.fixture
def monitored(tmp_path):
    source = Path("kaggle/region-tokens/runner.py").read_text()
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "command"
    )
    events = []
    namespace = dict(
        OUT=tmp_path,
        REPO=tmp_path,
        selectors=selectors,
        signal=signal,
        os=os,
        subprocess=subprocess,
        sys=sys,
        time=time,
        mark=lambda *a, **k: events.append((a, k)),
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace
    )
    return namespace["command"], events, tmp_path


def test_output_is_saved_and_success_returns(monitored):
    command, _, root = monitored
    command([sys.executable, "-c", "print('completed')"])
    assert next((root / "command_logs").glob("*.log")).read_text() == "completed\n"


def test_nonzero_exit_is_not_success(monitored):
    command, _, _ = monitored
    with pytest.raises(subprocess.CalledProcessError):
        command([sys.executable, "-c", "raise SystemExit(7)"])


def test_child_output_refreshes_idle_clock(monitored):
    command, _, _ = monitored
    command(
        [
            sys.executable,
            "-c",
            "import time\nfor i in range(5):\n print(i, flush=True)\n time.sleep(.1)",
        ],
        timeout=3,
        idle_timeout=0.4,
        heartbeat=0.1,
    )


@pytest.mark.parametrize(
    "wall,idle,reason",
    [
        (2, 0.3, "no_output_timeout"),
        (0.3, 2, "wall_timeout"),
    ],
)
def test_silent_child_times_out_with_honest_heartbeat(monitored, wall, idle, reason):
    command, events, _ = monitored
    with pytest.raises(subprocess.TimeoutExpired):
        command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=wall,
            idle_timeout=idle,
            heartbeat=0.1,
        )
    assert any(a == ("command_heartbeat",) for a, _ in events)
    assert events[-1][1]["reason"] == reason
    with pytest.raises(ProcessLookupError):
        os.kill(events[-1][1]["child_pid"], 0)

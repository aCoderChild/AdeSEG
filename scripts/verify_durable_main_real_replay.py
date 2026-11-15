"""Engineering-only adapter for testing durable cohort main on real seq3."""

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_spatiotemporal_tome_value_exchange_durable as runner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kill-during-local", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.kill_during_local and "--resume" in remaining:
        raise ValueError("Kill injection and resume cannot be combined")
    runner.SEQUENCES = ("seq3",)
    if options.kill_during_local:
        def watchdog():
            while True:
                try:
                    state = json.loads((options.output_dir / "progress.json").read_text())
                    if state.get("condition") == "local" and state.get("targets_done", 0) >= 1:
                        os.kill(os.getpid(), signal.SIGKILL)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(0.02)
        threading.Thread(target=watchdog, daemon=True).start()
    sys.argv = [sys.argv[0], "--output-dir", str(options.output_dir), *remaining]
    runner.main()


if __name__ == "__main__":
    main()

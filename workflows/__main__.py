"""Phase-based CLI for the existing frozen experiment implementations."""

import argparse
from pathlib import Path
import subprocess
import sys

from .catalog import STAGES

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m workflows",
        description="Navigate the study without changing experiment code or settings.",
        usage="%(prog)s [--list] | STAGE [COMMAND [original script arguments ...]]",
    )
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        parser.print_help()
        print("\nCurrent project study:")
        for name in STAGES["current"][1]:
            print(f"  current {name} --help")
        print("\nOther stages:")
        for name, (description, _) in STAGES.items():
            if name == "current":
                continue
            print(f"  {name:13} {description}")
        return 0
    if arguments == ["--list"]:
        for stage, (_, commands) in STAGES.items():
            for name, script in commands.items():
                print(f"{stage:13} {name:20} scripts/{script}")
        return 0
    stage = arguments[0]
    if stage not in STAGES:
        parser.error(f"Unknown stage {stage!r}; use --list")
    description, commands = STAGES[stage]
    if len(arguments) == 1:
        print(description)
        for name, script in commands.items():
            print(f"  {name:20} scripts/{script}")
        return 0
    command = arguments[1]
    if command not in commands:
        parser.error(f"Unknown command {command!r} for {stage!r}; choose {', '.join(commands)}")
    script = ROOT/"scripts"/commands[command]
    # No shell, imports of experiment modules, setting overrides or automatic
    # gate bypass. The original runner remains responsible for all guards.
    return subprocess.run([sys.executable, str(script), *arguments[2:]], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

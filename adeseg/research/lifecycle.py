"""Run reproducible AdeSEG experiment stages with explicit artifact receipts.

Usage:
  python -m adeseg.research.lifecycle verify --protocol experiments/...
  python -m adeseg.research.lifecycle run --protocol ... --stage inference
  python -m adeseg.research.lifecycle run --protocol ... --stage all --dry-run
Commands are argv arrays (never shell strings). A stage is successful only
after its process exits 0 and every declared output exists.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from .protocol import STAGES, verify_protocol, digest

def run_stage(stage, spec, root, workdir, dry_run=False):
    cfg = spec["stages"][stage]
    argv = [arg.replace("{python}", sys.executable)
               .replace("{workdir}", str(workdir))
               .replace("{protocol_dir}", str(root))
            for arg in cfg["command"]]
    outputs = [(workdir / arg).resolve() for arg in cfg["artifacts"]]
    if any(not path.is_relative_to(workdir.resolve()) for path in outputs):
        raise ValueError(f"{stage}: artifacts must be inside workdir")
    if not argv:
        return {"stage": stage, "status": "not_implemented",
                "command": [], "artifacts": []}
    if dry_run:
        return {"stage": stage, "status": "dry_run", "command": argv,
                "artifacts": [str(path) for path in outputs]}
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"{stage}: refusing to overwrite existing artifacts")
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ADESEG_EXPERIMENT_ID"] = spec["experiment_id"]
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(argv, cwd=root, env=env, check=False)
    record = {"stage": stage, "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "command": argv, "exit_code": proc.returncode}
    if proc.returncode:
        record["status"] = "failed"
        return record
    missing = [str(p) for p in outputs if not p.is_file()]
    if missing:
        record.update(status="missing_artifacts", missing_artifacts=missing)
        return record
    record["status"] = "completed"
    record["artifacts"] = [
        {"path": str(p), "sha256": digest(p)} for p in outputs
    ]
    return record

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--workdir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path, spec, checked = verify_protocol(args.protocol)
    if args.action == "verify":
        print(json.dumps({"status": "verified", "experiment_id": spec["experiment_id"],
                          "inputs": checked}, indent=2))
        return 0
    workdir = (args.workdir / spec["experiment_id"]).resolve()
    stages = STAGES if args.stage == "all" else (args.stage,)
    if not args.dry_run and workdir.exists() and (workdir / "receipt.json").exists():
        raise FileExistsError("Receipt already exists; select a fresh output root")
    records = []
    for stage in stages:
        result = run_stage(stage, spec, path.parent, workdir, dry_run=args.dry_run)
        records.append(result)
        if result["status"] not in ("completed", "dry_run"):
            break
    receipt = {"experiment_id": spec["experiment_id"], "track": spec["track"],
               "inputs": checked, "results": records,
               "status": ("dry_run" if args.dry_run else
                          "completed" if all(x["status"] == "completed" for x in records)
                          and len(records) == len(stages) else "incomplete")}
    if not args.dry_run:
        workdir.mkdir(parents=True, exist_ok=True)
        destination = workdir / "receipt.json"
        destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] in ("completed", "dry_run") else 2

if __name__ == "__main__":
    raise SystemExit(main())

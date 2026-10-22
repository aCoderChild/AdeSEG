"""Fail-closed gate -> full read-only attention -> stop for exclusion protocol."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parent = validate_output(args.output_dir)
    parent.mkdir(parents=True, exist_ok=False)
    status = {"state": "prepared", "pid": os.getpid(), "completed": []}

    def save():
        status["updated_unix"] = time.time()
        (parent / "queue_status.json").write_text(json.dumps(status, indent=2) + "\n")

    common = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_token_attention.py"),
        "--method",
        "sam2",
        "--device",
        "cpu",
        "--checkpoint",
        str(args.checkpoint),
        "--manifest",
        str(args.manifest),
    ]
    save()
    try:
        for stage in ("validation", "full"):
            status["state"] = "running_" + stage
            save()
            command = common + ["--output-dir", str(parent / stage)]
            command += (
                ["--smoke"]
                if stage == "validation"
                else ["--validation-dir", str(parent / "validation")]
            )
            with (parent / f"{stage}_console.log").open("wb") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            delivery = json.loads((parent / stage / "delivery.json").read_text())
            if (
                not delivery["completed"]
                or not delivery["wandb_artifact_upload_complete"]
            ):
                raise RuntimeError("Missing scientific output/W&B artifact delivery")
            status["completed"].append(stage)
            if stage == "validation":
                gate = json.loads((parent / stage / "validation.json").read_text())
                if not gate["passed"]:
                    raise RuntimeError("Replay gate failed")
                summaries = [
                    json.loads(p.read_text())
                    for p in (parent / stage).glob("seq*/token_summary.json")
                ]
                compressed = sum(s["compressed_token_bytes"] for s in summaries)
                targets = json.loads((parent / stage / "summary.json").read_text())[
                    "target_frames"
                ]
                projection = compressed / targets * 2012
                status["storage_projection"] = {
                    "gate_token_bytes": compressed,
                    "gate_targets": targets,
                    "projected_full_token_bytes": projection,
                    "free_bytes": shutil.disk_usage(parent).free,
                    "limits": "Attention files only; other masks, annotations, models and archives add storage",
                }
                # Both source token files and per-sequence archives must fit.
                if 2 * projection > shutil.disk_usage(parent).free:
                    raise RuntimeError(
                        "Insufficient free storage for projected tokens plus archives"
                    )
            save()
        status["state"] = "attention_complete_awaiting_committed_exclusion_protocol"
        save()
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()

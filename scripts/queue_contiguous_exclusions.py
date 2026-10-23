"""Fail-closed committed rigid-exclusion gate, then the complete cohort."""

import argparse
import json
import os
from pathlib import Path
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
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--diagnosis-dir", type=Path)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    status = {"state": "prepared", "pid": os.getpid(), "completed": []}

    def save():
        status["updated_unix"] = time.time()
        (output / "queue_status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    common = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_contiguous_exclusions.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--manifest",
        str(args.manifest),
        "--reference-dir",
        str(args.reference_dir),
    ]
    if args.resume_dir is not None:
        if args.diagnosis_dir is None:
            raise RuntimeError("Resume requires completed diagnoses")
        common += ["--repair-validation", "--diagnosis-dir", str(args.diagnosis_dir)]
    try:
        for stage in ("validation", "full"):
            status["state"] = "running_" + stage
            save()
            command = common + ["--output-dir", str(output / stage)]
            command += (
                ["--smoke"]
                if stage == "validation"
                else ["--validation-dir", str(output / "validation")]
            )
            if stage == "full" and args.resume_dir is not None:
                command += ["--resume-dir", str(args.resume_dir)]
            with (output / f"{stage}_console.log").open("wb") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            delivery = json.loads((output / stage / "delivery.json").read_text())
            if (
                not delivery["completed"]
                or not delivery["wandb_artifact_upload_complete"]
            ):
                raise RuntimeError("Missing W&B/scientific delivery")
            status["completed"].append(stage)
            if stage == "validation":
                gate = json.loads((output / stage / "validation.json").read_text())
                if not gate["passed"]:
                    raise RuntimeError("Exclusion integrity gate failed")
                sequences = json.loads((output / stage / "sequences.json").read_text())
                seconds = sum(r["seconds"] for r in sequences)
                targets = sum(r["intervention_targets"] for r in sequences)
                status["runtime_projection"] = {
                    "gate_seconds": seconds,
                    "gate_intervention_targets": targets,
                    "conservative_full_seconds": seconds / targets * 2012,
                    "limits": "Rough upper projection includes gate baseline-only frames; cache reuse varies by mask. No quota/time-limited scientific stopping.",
                }
            save()
        status["state"] = "completed"
        save()
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()

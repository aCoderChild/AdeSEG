"""Online-tracked E0 -> E1 notebook orchestration; later stages remain gated."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from kaggle_secrets import UserSecretsClient

OUT = Path("/kaggle/working/region_tokens_v1")
REPO = Path("/kaggle/temp/AdeSEG-region")
OUT.mkdir(parents=True, exist_ok=False)
started = time.time()
run = None


def mark(stage, state="running", **details):
    payload = {
        "stage": stage,
        "state": state,
        "elapsed_seconds": time.time() - started,
        **details,
    }
    temporary = OUT / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, OUT / "status.json")
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
    print("[region-status]", json.dumps(payload), flush=True)
    if run:
        run.log(
            {
                "stage": stage,
                "state": state,
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )


def command(arguments, timeout=21600):
    arguments = list(map(str, arguments))
    print("+", " ".join(arguments), flush=True)
    subprocess.run(
        arguments, cwd=REPO, check=True, timeout=timeout, env=os.environ.copy()
    )


def upload():
    import wandb

    artifact = wandb.Artifact(f"region-tokens-{run.id}", type="experiment-results")
    for path in OUT.rglob("*"):
        if (
            path.is_file()
            and "wandb" not in path.relative_to(OUT).parts
            and path.suffix != ".tmp"
        ):
            artifact.add_file(str(path), name=path.relative_to(OUT).as_posix())
    run.log_artifact(artifact).wait()


try:
    mark("bootstrap")
    # Authentication fails before GPU inference; no silent disabled/offline fallback.
    secret = UserSecretsClient().get_secret("WANDB_API_KEY")
    if not secret:
        raise RuntimeError("Attach WANDB_API_KEY in Kaggle Add-ons > Secrets")
    os.environ.update(
        WANDB_API_KEY=secret,
        WANDB_BASE_URL="https://forge.coreweave.com/api/wandb",
        WANDB_PROJECT="ade-seg",
        WANDB_ENTITY="adenoid-hypertrophy",
        PYTHONUNBUFFERED="1",
    )
    del secret
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            "--branch",
            "experiment",
            "https://github.com/aCoderChild/AdeSEG.git",
            str(REPO),
        ],
        check=True,
        timeout=300,
    )
    revision = os.environ.get("ADESEG_REVISION")
    command(["git", "checkout", "--detach", revision or "experiment"], 180)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    packages = {
        "hydra": "hydra-core>=1.3.2",
        "iopath": "iopath>=0.1.10",
        "wandb": "wandb",
        "monai": "monai>=1.3",
        "skimage": "scikit-image",
        "scipy": "scipy",
    }
    missing = [
        package
        for module, package in packages.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        command([sys.executable, "-m", "pip", "install", *missing], 600)
    if importlib.util.find_spec("tensordict") is None:
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "tensordict>=0.7,<0.8",
            ],
            300,
        )
    command(
        [sys.executable, REPO / "scripts/verify_wandb_connection.py", "--check-only"],
        120,
    )
    import wandb

    os.environ["WANDB_RUN_GROUP"] = f"region-tokens-v1-{revision[:12]}"
    run = wandb.init(
        project="ade-seg",
        entity="adenoid-hypertrophy",
        mode="online",
        name="region-tokens-orchestrator",
        group=os.environ["WANDB_RUN_GROUP"],
        dir=str(OUT),
        config={
            "git_revision": revision,
            "protocol": "region_tokens_v1_1",
            "scope": "E0/E1 gates",
        },
    )
    (OUT / "tracking.json").write_text(
        json.dumps(
            {
                "mode": "online",
                "run_id": run.id,
                "url": run.url,
                "git_revision": revision,
            },
            indent=2,
        )
        + "\n"
    )
    mark("tracking", wandb_run_id=run.id)
    manifest = OUT / "polypgen_manifest.jsonl"
    command(
        [
            sys.executable,
            REPO / "scripts/build_polypgen_kaggle_manifest.py",
            "--data-root",
            "/kaggle/input",
            "--output",
            manifest,
        ],
        180,
    )
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    protocol = REPO / "protocols/region_tokens_v1_1.json"
    (OUT / "protocol.json").write_bytes(protocol.read_bytes())
    command(
        [
            sys.executable,
            REPO / "scripts/audit_region_initialization.py",
            "--manifest",
            manifest,
            "--output",
            OUT / "initialization_audit.json",
        ]
    )
    audit = json.loads((OUT / "initialization_audit.json").read_text())
    eligibility = {
        row["video"]: row["first_visible_frame"] for row in audit["sequences"]
    }
    for video in json.loads(protocol.read_text())["pilot"]["sequences"]:
        if eligibility.get(video) is None:
            raise RuntimeError(f"Unpromptable preregistered pilot sequence: {video}")
    mark("initialization_audit", prompt_frames=eligibility)
    upload()
    counts = {}
    inventory = []
    for index, row in enumerate(rows, start=1):
        counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
        inventory.append(
            {
                **row,
                "image_sha256": hashlib.sha256(
                    Path(row["image"]).read_bytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(
                    Path(row["mask"]).read_bytes()
                ).hexdigest(),
            }
        )
        if index % 100 == 0 or index == len(rows):
            mark("data_hashes", frames_hashed=index, total_frames=len(rows))
    (OUT / "data_inventory.json").write_text(
        json.dumps(
            {
                "sequence_counts": counts,
                "total_frames": len(rows),
                "source": "debeshjha1/polypgen-video-sequence",
                "synapse_checksum_match": None,
                "files": inventory,
            },
            indent=2,
        )
        + "\n"
    )
    if len(counts) != 23:
        raise RuntimeError(
            f"Expected 23 positive PolypGen sequences, found {len(counts)}"
        )
    mark(
        "data_preflight",
        sequences=len(counts),
        frames=len(rows),
        synapse_equivalence="unverified",
    )
    checkpoint = Path("/kaggle/temp/sam2.1_hiera_base_plus.pt")
    command(
        [
            "wget",
            "--quiet",
            "--timeout=30",
            "--tries=3",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "-O",
            checkpoint,
        ],
        600,
    )
    if checkpoint.stat().st_size < 300_000_000:
        raise RuntimeError("Truncated SAM2.1 checkpoint")
    common = [
        sys.executable,
        REPO / "scripts/run_region_token_pilot.py",
        "--manifest",
        manifest,
        "--checkpoint",
        checkpoint,
        "--protocol",
        protocol,
    ]
    mark("E0")
    command([*common, "--e0-only", "--output-dir", OUT / "E0"])
    if not json.loads((OUT / "E0/status.json").read_text())["e0_pass"]:
        raise RuntimeError("E0 gate failed")
    mark("E0", "complete")
    upload()
    mark("E1")
    command([*common, "--output-dir", OUT / "E1"])
    command(
        [
            sys.executable,
            REPO / "scripts/analyze_region_token_pilot.py",
            "--input",
            OUT / "E1/frames.jsonl",
            "--protocol",
            protocol,
            "--output",
            OUT / "E1/decision.json",
        ]
    )
    decision = json.loads((OUT / "E1/decision.json").read_text())
    run.summary["E1/gate_pass"] = decision["gate_pass"]
    run.summary["full_study_complete"] = False
    mark("E1", "complete", decision=decision["decision"], full_study_complete=False)
    # Analysis/method code must be validated separately after this scientific gate.
    mark(
        "pilot_complete",
        "complete",
        next_stage="E2 pending implementation/GPU validation"
        if decision["gate_pass"]
        else "stop_or_pivot",
        full_study_complete=False,
    )
    upload()
    run.finish()
except BaseException as exc:
    mark("failed", "failed", error=str(exc))
    (OUT / "failure.txt").write_text(traceback.format_exc())
    if run:
        upload()
        run.finish(exit_code=1)
    raise

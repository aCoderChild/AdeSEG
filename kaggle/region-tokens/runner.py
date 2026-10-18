"""Online-tracked E0 -> E1 notebook orchestration; later stages remain gated."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from kaggle_secrets import UserSecretsClient

RUN_STAGE = os.environ.get("REGION_STAGE", "E1")
if RUN_STAGE not in {
    "E0",
    "E1",
    "E2_preflight",
    "E2_calibration",
    "E2_mean_smoke",
    "E2_attention_smoke",
    "E2_attention_primary",
    "study_shard",
}:
    raise ValueError(f"Unsupported REGION_STAGE: {RUN_STAGE}")
OUT = Path("/kaggle/working") / os.environ.get(
    "REGION_OUTPUT_NAME",
    "region_tokens_v1" if RUN_STAGE == "E1" else f"region_tokens_{RUN_STAGE}",
)
REPO = Path("/kaggle/temp/AdeSEG-region")
if OUT.exists():
    previous_status = OUT / "status.json"
    previous = (
        json.loads(previous_status.read_text()) if previous_status.is_file() else {}
    )
    if previous.get("state") != "failed":
        raise FileExistsError(
            f"Refusing to overwrite non-failed Kaggle output directory: {OUT}"
        )
    shutil.rmtree(OUT)
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


def resolve_kaggle_input(*relative_paths: str) -> Path:
    """Return the one verified Kaggle mount layout for an attached dataset."""
    candidates = [
        Path("/kaggle/input") / relative_path for relative_path in relative_paths
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    available = sorted(
        path.relative_to("/kaggle/input").as_posix()
        for path in Path("/kaggle/input").glob("**/*")
        if path.is_dir() and len(path.relative_to("/kaggle/input").parts) <= 3
    )
    raise FileNotFoundError(
        f"None of the expected Kaggle input paths exist: {candidates}. "
        f"Available directories: {available}"
    )


def upload():
    import wandb

    artifact = wandb.Artifact(f"region-tokens-{run.id}", type="experiment-results")
    for path in OUT.rglob("*"):
        if (
            path.is_file()
            and "wandb" not in path.relative_to(OUT).parts
            and "dependencies" not in path.relative_to(OUT).parts
            and path.suffix != ".tmp"
        ):
            artifact.add_file(str(path), name=path.relative_to(OUT).as_posix())
    run.log_artifact(artifact).wait()


def e1_gate_artifact():
    """Fetch and verify the completed E1 decision before an E2 shard starts."""
    # The reviewed decision is produced by the E1 orchestrator, not the
    # child E1 inference run (which contains raw rows only).
    run_id = os.environ.get("REGION_E1_RUN_ID", "7orfgsk2")
    artifact_root = OUT / "e1_gate_artifact"
    source = wandb.Api().run(f"adenoid-hypertrophy/ade-seg/{run_id}")
    from training.tracking import latest_results_artifact, download_forge_run_file

    latest_results_artifact(
        source, ("decision_verified.json", "decision.json")
    ).download(root=str(artifact_root))
    # The independently reviewed decision is a run file uploaded after the
    # original notebook artifact. Prefer it without confusing run history with
    # results artifacts.
    reviewed = [
        item for item in source.files() if item.name == "E1/decision_verified.json"
    ]
    if reviewed:
        download_forge_run_file(reviewed[0], artifact_root)
    decisions = sorted(artifact_root.rglob("decision_verified.json"))
    if not decisions:
        decisions = sorted(artifact_root.rglob("decision.json"))
    if len(decisions) != 1:
        raise RuntimeError("E1 results artifact does not contain one decision file")
    decision = json.loads(decisions[0].read_text())
    if not (
        decision.get("stage") == "E1"
        and decision.get("gate_pass") is True
        and decision.get("decision") == "proceed_to_E2"
        and decision.get("all_required_sequences") is True
    ):
        raise RuntimeError(f"E1 gate did not authorize E2: {decision}")
    return {"run_id": run_id, "decision": decision, "path": str(decisions[0])}


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
    # A notebook draft session can retain a checkout from a completed stage.
    # This is only the runner's fixed Kaggle scratch directory; outputs stay in OUT.
    if REPO.exists():
        shutil.rmtree(REPO)
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
    sys.path.insert(0, str(REPO))
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
        "statsmodels": "statsmodels>=0.14",
        "matplotlib": "matplotlib>=3.7",
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

    os.environ["WANDB_RUN_GROUP"] = (
        f"region-tokens-v1-{revision[:12]}-{OUT.name}-{int(started)}"
    )
    run = wandb.init(
        project="ade-seg",
        entity="adenoid-hypertrophy",
        mode="online",
        name=f"{RUN_STAGE} | pipeline | {OUT.name}",
        group=os.environ["WANDB_RUN_GROUP"],
        dir=str(OUT),
        config={
            "role": "orchestrator",
            "git_revision": revision,
            "protocol": "region_tokens_v1_1",
            "scope": f"{RUN_STAGE} region-token execution",
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
    (OUT / "restore_control_protocol.json").write_bytes(
        (REPO / "protocols/region_tokens_restore_control_v2.json").read_bytes()
    )
    if RUN_STAGE == "study_shard":
        (OUT / "execution_protocol.json").write_bytes(
            (REPO / "protocols/region_tokens_execution_v1.json").read_bytes()
        )
    if RUN_STAGE in {"E2_preflight", "study_shard"}:
        # This is deliberately an input/instrumentation gate, not an E2 result.
        # Kaggle sometimes nests attached public datasets beneath ``datasets``.
        # Accept only the two known layouts, and fail with an inventory otherwise.
        requested_dataset = json.loads(os.environ.get("REGION_JOB", "{}")).get(
            "dataset"
        )
        need_cholec = RUN_STAGE == "E2_preflight" or requested_dataset == "cholecseg8k"
        need_davis = RUN_STAGE == "E2_preflight" or requested_dataset == "davis"
        cholec_root = (
            resolve_kaggle_input("cholecseg8k", "datasets/newslab/cholecseg8k")
            if need_cholec
            else None
        )
        davis_root = (
            resolve_kaggle_input("davis2017", "datasets/lennelenne/davis2017")
            if need_davis
            else None
        )
        cholec_output = OUT / "cholecseg8k"
        davis_manifest = OUT / "davis2017.jsonl"
        if need_cholec:
            command(
                [
                    sys.executable,
                    REPO / "scripts/build_cholecseg8k_manifest.py",
                    "--data-root",
                    cholec_root,
                    "--output-dir",
                    cholec_output,
                ],
                900,
            )
        if need_davis:
            command(
                [
                    sys.executable,
                    REPO / "scripts/build_davis_manifest.py",
                    "--data-root",
                    davis_root,
                    "--protocol",
                    REPO / "protocols/utility_retrieval_v1.json",
                    "--output",
                    davis_manifest,
                ],
                900,
            )
        cholec_rows = (
            [
                json.loads(line)
                for line in (cholec_output / "cholecseg8k.jsonl")
                .read_text()
                .splitlines()
                if line
            ]
            if need_cholec
            else []
        )
        davis_rows = (
            [
                json.loads(line)
                for line in davis_manifest.read_text().splitlines()
                if line
            ]
            if need_davis
            else []
        )
        if (need_cholec and not cholec_rows) or (need_davis and not davis_rows):
            raise RuntimeError("E2 input preflight produced an empty manifest")
        davis_splits = {row["split"] for row in davis_rows}
        if need_davis and davis_splits != {"train", "holdout", "val"}:
            raise RuntimeError(f"Unexpected DAVIS splits: {sorted(davis_splits)}")
        write = {
            "stage": "E2_preflight",
            "claim": "input and E0 gate only; no E2 intervention outcome",
            "inputs": {
                "polypgen": {
                    "dataset": "debeshjha1/polypgen-video-sequence",
                    "frames": len(rows),
                    "videos": len({row["video_id"] for row in rows}),
                },
                "cholecseg8k": {
                    "dataset": "newslab/cholecseg8k",
                    "mounted_path": str(cholec_root),
                    "frames": len(cholec_rows),
                    "clips": len({row["video_id"] for row in cholec_rows}),
                    "source_videos": sorted(
                        {row["source_video"] for row in cholec_rows}
                    ),
                },
                "davis2017": {
                    "dataset": "lennelenne/davis2017",
                    "mounted_path": str(davis_root),
                    "frames": len(davis_rows),
                    "videos": len({row["video_id"] for row in davis_rows}),
                    "splits": sorted(davis_splits),
                },
            },
        }
        if not need_cholec:
            del write["inputs"]["cholecseg8k"]
        if not need_davis:
            del write["inputs"]["davis2017"]
        (OUT / "input_preflight.json").write_text(json.dumps(write, indent=2) + "\n")
        mark("E2_inputs", **write["inputs"])
        upload()
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
    e0_arguments = []
    if os.environ.get("REGION_E0_TARGET_FRAME"):
        if RUN_STAGE != "E0":
            raise ValueError("Targeted E0 diagnostics cannot override study targets")
        e0_arguments = [
            "--sequence",
            os.environ["REGION_E0_SEQUENCE"],
            "--e0-target-frame",
            os.environ["REGION_E0_TARGET_FRAME"],
        ]
    command([*common, "--e0-only", *e0_arguments, "--output-dir", OUT / "E0"])
    if not json.loads((OUT / "E0/status.json").read_text())["e0_pass"]:
        raise RuntimeError("E0 gate failed")
    mark("E0", "complete")
    upload()
    if RUN_STAGE == "study_shard":
        # New experiment components are dispatched explicitly; running this
        # branch never implies that unrequested components were executed.
        gate = e1_gate_artifact()
        job = json.loads(os.environ.get("REGION_JOB", "{}"))
        stage = job.get("stage")
        if stage not in {
            "E2",
            "E3",
            "E2_calibration",
            "E4_conditions",
            "E6_dev",
            "E6_test",
            "E6_baseline",
        }:
            raise ValueError("REGION_JOB requires an explicit implemented study stage")
        dataset = job.get("dataset")
        manifests = {
            "polypgen": manifest,
            "cholecseg8k": cholec_output / "cholecseg8k.jsonl",
            "davis": davis_manifest,
        }
        if dataset not in manifests or not job.get("split") or not job.get("sequences"):
            raise ValueError("REGION_JOB requires dataset, split and sequences")
        sequences = job["sequences"]
        if (
            not isinstance(sequences, list)
            or not all(isinstance(v, str) for v in sequences)
            or len(sequences) != len(set(sequences))
        ):
            raise ValueError("REGION_JOB sequences must be a unique list")
        (OUT / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        stage_output = OUT / "study"
        entry = (
            "measure_region_conditions.py"
            if stage == "E4_conditions"
            else "run_region_baseline.py"
            if stage == "E6_baseline"
            else "run_region_experiments.py"
        )
        arguments = [
            sys.executable,
            REPO / "scripts" / entry,
            "--manifest",
            manifests[dataset],
            "--dataset",
            dataset,
            "--split",
            job["split"],
            "--output-dir",
            stage_output,
            *sum((["--sequence", v] for v in sequences), []),
        ]
        if stage != "E4_conditions":
            arguments += [
                "--checkpoint",
                checkpoint,
                "--gate-dir",
                OUT / "e1_gate_artifact",
            ]
        if stage == "E6_baseline":
            baseline = job.get("baseline")
            if baseline not in {"SAM2Long", "Lean-SAM2"}:
                raise ValueError(
                    "Only complete, author-provided upstream baselines are supported"
                )
            command(
                [
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--depth",
                    "1",
                    f"external/{baseline}",
                ],
                600,
            )
            arguments += ["--baseline", baseline, "--partition", job["partition"]]
        elif stage != "E4_conditions":
            arguments += [
                "--stage",
                stage,
                "--mode",
                job.get("mode", "single_step"),
                "--intervention",
                job.get("intervention", "attention_mask"),
                "--memory-condition",
                job.get("memory_condition", "c0"),
                "--band",
                str(job.get("band", 1)),
                "--thresholds",
                *map(str, job.get("thresholds", [0.1, 0.9])),
                "--boundary-definition",
                job.get("boundary_definition", "mixed_coverage"),
                "--spatial-match",
                job.get("spatial_match", "pooled_distance"),
            ]
            for label in job.get("label_ids", []):
                arguments += ["--label-id", str(label)]
            if job.get("calibration_artifact"):
                dependency = OUT / "dependencies/calibration"
                wandb.Api().artifact(job["calibration_artifact"]).download(
                    root=str(dependency)
                )
                provenance = list(dependency.rglob("calibration_provenance.json"))
                if len(provenance) != 1:
                    raise ValueError(
                        "Calibration artifact must contain exactly one completed calibration"
                    )
                arguments += ["--calibration-dir", provenance[0].parent]
            if job.get("method_artifact"):
                dependency = OUT / "dependencies/method"
                wandb.Api().artifact(job["method_artifact"]).download(
                    root=str(dependency)
                )
                method = dependency / job.get("method_file", "method_frozen.json")
                if (
                    not method.resolve().is_relative_to(dependency.resolve())
                    or not method.is_file()
                ):
                    raise ValueError(
                        "Method config file missing from the selected artifact"
                    )
                arguments += ["--method-config", method]
        # Pure-method E5 reuse needs only the pinned upstream source file.
        command(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
                "external/Lean-SAM2",
            ],
            600,
        )
        mark(stage, e1_gate_run=gate["run_id"], dataset=dataset, sequences=sequences)
        command(arguments)
        result = json.loads((stage_output / "status.json").read_text())
        if result.get("state") != "complete":
            raise RuntimeError("Study shard did not complete")
        frame_files = sorted(stage_output.glob("*/frames.jsonl"))
        rollout_files = sorted(stage_output.glob("*/rollout.jsonl"))
        if (stage_output / "rollout.jsonl").exists():
            rollout_files.append(stage_output / "rollout.jsonl")
        if frame_files or rollout_files:
            command(
                [
                    sys.executable,
                    REPO / "scripts/analyze_region_experiments.py",
                    *sum(
                        (
                            ["--input", str(path)]
                            for path in frame_files + rollout_files
                        ),
                        [],
                    ),
                    "--output",
                    stage_output / "analysis.json",
                    "--figures",
                ]
            )
        run.summary["full_study_complete"] = False
        mark(
            "study_shard_complete",
            "complete",
            experiment_stage=stage,
            full_study_complete=False,
        )
        upload()
    elif RUN_STAGE == "E0":
        mark("E0_complete", "complete", next_stage="E1 or E2 calibration")
        upload()
    elif RUN_STAGE == "E2_preflight":
        run.summary["E2_preflight/cholecseg8k_frames"] = len(cholec_rows)
        run.summary["E2_preflight/davis_frames"] = len(davis_rows)
        run.summary["full_study_complete"] = False
        mark(
            "E2_preflight_complete",
            "complete",
            next_stage="implement and GPU-validate the full E2 intervention matrix",
            full_study_complete=False,
        )
        upload()
    elif RUN_STAGE == "E2_calibration":
        mark("E2_calibration")
        command([*common, "--calibrate-values", "--output-dir", OUT / "E2_calibration"])
        calibration = json.loads(
            (OUT / "E2_calibration/value_calibration.json").read_text()
        )
        run.summary["E2_calibration/categories"] = len(calibration["categories"])
        run.summary["full_study_complete"] = False
        mark(
            "E2_calibration_complete",
            "complete",
            next_stage="E2 interventions pending GPU validation",
            full_study_complete=False,
        )
        upload()
    elif RUN_STAGE == "E2_mean_smoke":
        artifact_name = os.environ.get("REGION_CALIBRATION_ARTIFACT")
        if not artifact_name:
            raise RuntimeError(
                "Set REGION_CALIBRATION_ARTIFACT to the finished calibration artifact"
            )
        mark("E2_mean_smoke", calibration_artifact=artifact_name)
        calibration_root = OUT / "calibration_artifact"
        artifact = wandb.Api().artifact(artifact_name)
        artifact.download(root=str(calibration_root))
        calibration = calibration_root / "E2_calibration" / "value_calibration.json"
        if not calibration.is_file():
            raise RuntimeError(
                "Calibration artifact does not contain E2_calibration/value_calibration.json"
            )
        command(
            [
                *common,
                "--e2-mean-smoke",
                "--value-calibration",
                calibration,
                "--output-dir",
                OUT / "E2_mean_smoke",
            ]
        )
        smoke = json.loads((OUT / "E2_mean_smoke/status.json").read_text())
        if not smoke["e0_pass"]:
            raise RuntimeError("E2 mean-substitution smoke E0 checks failed")
        run.summary["E2_mean_smoke/sequences"] = len(smoke["sequences"])
        run.summary["full_study_complete"] = False
        mark(
            "E2_mean_smoke_complete",
            "complete",
            next_stage="full E2 interventions pending implementation",
            full_study_complete=False,
        )
        upload()
    elif RUN_STAGE in {"E2_attention_smoke", "E2_attention_primary"}:
        gate = e1_gate_artifact()
        sequences = [
            value.strip()
            for value in os.environ.get("REGION_SEQUENCES", "").split(",")
            if value.strip()
        ]
        if not sequences:
            raise RuntimeError("Set REGION_SEQUENCES to the explicit E2 shard")
        if len(sequences) != len(set(sequences)):
            raise RuntimeError("REGION_SEQUENCES contains duplicates")
        if set(sequences) - {row["video_id"] for row in rows}:
            raise RuntimeError(
                "REGION_SEQUENCES contains a video absent from the manifest"
            )
        if RUN_STAGE == "E2_attention_smoke" and len(sequences) != 1:
            raise RuntimeError("E2 attention smoke requires exactly one sequence")
        mark(RUN_STAGE, e1_gate_run=gate["run_id"], sequences=sequences)
        command(
            [
                *common,
                "--stage",
                "E2_attention_primary",
                "--all-eligible-targets",
                *sum((["--sequence", sequence] for sequence in sequences), []),
                "--output-dir",
                OUT / "E2_attention",
            ]
        )
        result = json.loads((OUT / "E2_attention/status.json").read_text())
        if result.get("state") != "complete" or not result.get("e0_pass"):
            raise RuntimeError("E2 attention shard failed exactness checks")
        run.summary["E2_attention/sequences"] = len(sequences)
        run.summary["E2_attention/rows"] = result["rows"]
        run.summary["full_study_complete"] = False
        mark(
            f"{RUN_STAGE}_complete",
            "complete",
            rows=result["rows"],
            full_study_complete=False,
            next_stage="remaining E2 interventions and persistent mode",
        )
        upload()
    else:
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

"""Kaggle runner for frozen-protocol Must-tier sanity and E1--E3 evidence."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = TEMP / "AdeSEG"
OUT = WORK / "latent_utility_must_phase_a"
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.time()


def mark(stage, state="running", **details):
    payload = {"stage": stage, "state": state, "elapsed_seconds": time.time() - STARTED, **details}
    temporary = OUT / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, OUT / "status.json")
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print("[status]", json.dumps(payload, sort_keys=True), flush=True)


def run(command, timeout=21600):
    command = list(map(str, command))
    mark("command", command=" ".join(command))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=os.environ.copy(), check=True, timeout=timeout)


mark("bootstrap")
key = None
for attempt in range(1, 6):
    try:
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        if key:
            break
        print(f"[wandb] Empty Kaggle secret (attempt {attempt}/5)", flush=True)
    except Exception as exc:
        print(f"[wandb] Kaggle secret read failed: {type(exc).__name__} (attempt {attempt}/5)", flush=True)
    if attempt < 5:
        time.sleep(min(2 ** attempt, 10))
os.environ.update({
    "PYTHONUNBUFFERED": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "WANDB_BASE_URL": "https://forge.coreweave.com/api/wandb",
    "WANDB_PROJECT": "ade-seg",
    "WANDB_ENTITY": "adenoid-hypertrophy",
})
if key:
    os.environ["WANDB_API_KEY"] = key
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1", "--branch",
    "experiment/utility-aware-memory-2026-10-16",
    "https://github.com/aCoderChild/AdeSEG.git", str(REPO),
], check=True, timeout=300)
subprocess.run([
    "git", "submodule", "update", "--init", "--depth", "1", "external/MedSAM2",
], cwd=REPO, check=True, timeout=900)
missing = [package for module, package in {
    "hydra": "hydra-core>=1.3.2", "iopath": "iopath>=0.1.10", "wandb": "wandb",
    "statsmodels": "statsmodels>=0.14",
}.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True, timeout=300)
if importlib.util.find_spec("tensordict") is None:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-deps", "tensordict>=0.7,<0.8",
    ], check=True, timeout=300)
tracking_mode = "offline"
if key:
    try:
        run([sys.executable, REPO / "scripts/verify_wandb_connection.py", "--check-only"], 120)
        tracking_mode = "online"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[wandb] Connection check failed: {type(exc).__name__}; recording offline runs", flush=True)
else:
    print("[wandb] Secret unavailable; recording offline runs for later sync", flush=True)
mark("tracking", mode=tracking_mode)

protocol = REPO / "protocols/latent_utility_v2.json"
manifest = TEMP / "davis.jsonl"
checkpoint = TEMP / "sam2.1_hiera_base_plus.pt"
run([sys.executable, REPO / "scripts/build_davis_manifest.py", "--data-root", "/kaggle/input", "--protocol", protocol, "--output", manifest], 180)
subprocess.run([
    "wget", "--quiet", "--timeout=30", "--tries=3",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "-O", str(checkpoint),
], check=True, timeout=600)
if checkpoint.stat().st_size < 300_000_000:
    raise RuntimeError(f"Incomplete B+ checkpoint: {checkpoint.stat().st_size} bytes")

common = [
    sys.executable, REPO / "scripts/generate_utility_labels.py",
    "--sam2-cfg", "configs/sam2.1_hiera_b+1024.yaml",
    "--sam2-checkpoint", checkpoint,
    "--manifest", manifest,
    "--protocol", protocol,
    "--device", "cuda",
    "--minimum-target-frame", "15",
]
tracking = ["--wandb-base-url", os.environ["WANDB_BASE_URL"],
            "--wandb-mode", tracking_mode]


def utility(name, *arguments, timeout=21600):
    target = OUT / name
    run([*common, *tracking, "--output-dir", target,
         "--wandb-name", f"latent-must-a-{name.replace('/', '-')}",
         *arguments], timeout)
    return target


# E1/E2: the natural-memory failure onsets define the immutable matched target set.
natural_failures = utility(
    "E1_E2/natural_failures", "--split", "val", "--frame-selection", "failure_onsets",
    "--max-analyzed-frames", "50", "--max-frames-per-video-object", "2",
    "--memory-source", "predicted",
)
clean_failures = utility(
    "E1_E2/c0_matched_failures", "--split", "val", "--target-frames-jsonl", natural_failures / "frames.jsonl",
    "--memory-source", "ground_truth",
)
run([
    sys.executable, REPO / "scripts/classify_failure_taxonomy.py",
    "--predicted", natural_failures / "frames.jsonl",
    "--ground-truth-memory", clean_failures / "frames.jsonl",
    "--output", OUT / "E1_E2/taxonomy.json",
])

# E3 development-scale preregistered conditions, stratified across all holdout sequences.
conditions = {
    "natural": ("--memory-source", "predicted"),
    "c0": ("--memory-source", "ground_truth"),
    "c1": ("--memory-source", "corrupted", "--corruption-level", "0.25"),
    "c2": ("--memory-source", "corrupted", "--corruption-level", "0.50"),
    "c3": ("--memory-source", "corrupted", "--corruption-level", "0.75"),
    "c4": ("--memory-source", "corrupted", "--corruption-level", "1.00"),
}
paths = {}
for name, arguments in conditions.items():
    paths[name] = utility(
        f"E3/{name}", "--split", "holdout", "--frame-selection", "every_n", "--every-n", "20",
        "--max-frames-per-video-object", "3", "--no-random-null", *arguments,
    )
run([
    sys.executable, REPO / "scripts/analyze_utility_mechanism.py",
    "--condition", "predicted", paths["natural"] / "frames.jsonl",
    "--condition", "ground_truth", paths["c0"] / "frames.jsonl",
    "--condition", "corruption_0.25", paths["c1"] / "frames.jsonl",
    "--condition", "corruption_0.50", paths["c2"] / "frames.jsonl",
    "--condition", "corruption_0.75", paths["c3"] / "frames.jsonl",
    "--condition", "corruption_1.00", paths["c4"] / "frames.jsonl",
    "--output", OUT / "E3/mechanism.json",
])
mark("complete", state="complete")


"""Kaggle-only validation of the recovered frame-box baseline and memory hook.

Attach: debeshjha1/polypgen-video-sequence, bunnoob2005/checkpoints,
bunnoob2005/adeseg-region-boxes-validation-assets, and WANDB_API_KEY secret.
No full evaluation or hyperparameter selection is launched by this file.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

REVISION = "fb086b7"
repo = Path("/kaggle/temp/AdeSEG-boxes")
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
output = Path("/kaggle/working/region_boxes_validation_v1")
os.environ["WANDB_CACHE_DIR"] = str(output / ".wandb-cache")
os.environ["WANDB_DATA_DIR"] = str(output / ".wandb-staging")


def command(args):
    print("[box-validation]", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, env=os.environ.copy())


command([sys.executable, "-m", "pip", "install", "ultralytics", "hydra-core>=1.3.2", "iopath>=0.1.10", "monai>=1.3"])
command([sys.executable, "-m", "pip", "install", "--no-deps", "tensordict>=0.7,<0.8"])
command(["git", "clone", "https://github.com/aCoderChild/AdeSEG.git", repo])
command(["git", "-C", repo, "checkout", "--detach", REVISION])


def asset(name, digest=None):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}; found {len(matches)}")
    path = matches[0]
    if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Checkpoint differs from local historical asset: {name}")
    return path


checkpoint = asset("MedSAM2_latest.pt", "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060")
yolo = asset("polypgen_yolov8n.pt", "858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e")
reference = asset("historical_reference.json")
manifest = Path("/kaggle/temp/box_manifest.jsonl")
command([sys.executable, repo / "scripts/build_polypgen_kaggle_manifest.py",
         "--data-root", "/kaggle/input", "--output", manifest])
base = [sys.executable, "-u", repo / "scripts/run_medsam2_region_boxes.py",
        "--manifest", manifest, "--checkpoint", checkpoint, "--yolo-checkpoint", yolo,
        "--reference", reference, "--validation", "--max-frames", "12",
        "--sequence", "seq1", "--sequence", "seq2", "--sequence", "seq3"]
command(base + ["--output-dir", output, "--scale", "0"])
print("[box-validation] completed zero-scale technical prefix; production still gated", flush=True)

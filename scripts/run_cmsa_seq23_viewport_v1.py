#!/usr/bin/env python3
"""Run the pinned CMSA author loop on staged seq23 viewport images, without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.run_cmsa_polypgen import AUTHOR, REVISION, author_dataset, load_author
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_samurai_automatic_priority import model_state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.stage_cmsa_seq23_viewport_v1 import OUTPUT_ROOT, RESERVE, image_pixels, sha
from modeling.seq23_viewport import VIEWPORT, crop_viewport

LOCK = OUTPUT_ROOT / "contrast_refinement_v1/.source_group_capture_resume_v1.lock"
CHECKPOINT_SHA = "d391d9b6a6eb08eb4bc80651861a38ef52cc30e35d942d529312c226cff96f32"
STAGE_SHA = "1fccca194ccdb9d5ec0542ad3c502cf234294db4c43ba06084a7f62d3539b8f0"
DEPENDENCIES = ("scripts/run_cmsa_polypgen.py", "scripts/resume_source_group_capture_v1.py",
                "scripts/run_samurai_automatic_priority.py", "scripts/source_group_label_guard_v1.py",
                "scripts/stage_cmsa_seq23_viewport_v1.py", "datasets/video.py", "modeling/seq23_viewport.py")


def dependency_hashes():
    return {name: sha(ROOT / name) for name in DEPENDENCIES}


def read(path):
    before = sha(path)
    data = path.read_bytes()
    if before != hashlib.sha256(data).hexdigest() or before != sha(path):
        raise ValueError(f"Input changed during read: {path}")
    return json.loads(data)


def verify_stage(stage: Path) -> tuple[dict, list]:
    stage = stage.resolve(strict=True)
    if not stage.is_relative_to(OUTPUT_ROOT) or sha(stage / "stage.json") != STAGE_SHA:
        raise ValueError("Stage must be in Drive experiment storage")
    receipt = read(stage / "stage.json")
    if (receipt.get("passed") is not True or receipt.get("sequence") != "seq23"
            or receipt.get("viewport_xyxy") != list(VIEWPORT)
            or receipt.get("stager_sha256") != sha(ROOT / "scripts/stage_cmsa_seq23_viewport_v1.py")
            or receipt.get("crop_helper_sha256") != sha(ROOT / "modeling/seq23_viewport.py")
            or receipt.get("manifest_sha256") != sha(Path(receipt["manifest"]))):
        raise ValueError("Staging source or geometry differs")
    dummy = Path(receipt["dummy_mask"]).resolve(strict=True)
    if not dummy.is_relative_to(stage) or sha(dummy) != receipt["dummy_mask_sha256"]:
        raise ValueError("Synthetic shape mask differs")
    with Image.open(dummy) as image:
        dummy_pixels = np.asarray(image.convert("L"))
    if dummy_pixels.shape != (720, 834) or dummy_pixels.any():
        raise ValueError("Shape mask must be an all-zero viewport image")
    source_frames = {**load_manifest(Path(receipt["manifest"]), "dev"),
                     **load_manifest(Path(receipt["manifest"]), "test")}["seq23"]
    if len(source_frames) != 56 or len(receipt["records"]) != len(source_frames):
        raise ValueError("Staged sequence coverage differs from manifest")
    frames = []
    for i, record in enumerate(receipt["records"]):
        source = Path(record["source"]).resolve(strict=True)
        crop = Path(record["crop"]).resolve(strict=True)
        if (record["frame"] != i or record["source_frame_index"] != source_frames[i].frame_index
                or source != source_frames[i].image_path.resolve(strict=True)
                or not source.is_relative_to((ROOT / "data").resolve())
                or crop != stage / "inputs" / "seq23" / f"{i:06d}.png"
                or sha(source) != record["source_sha256"] or sha(crop) != record["crop_sha256"]):
            raise ValueError("Frame order or source binding differs")
        if (not np.array_equal(crop_viewport(image_pixels(source)), image_pixels(crop))
                or sha(source) != record["source_sha256"] or sha(crop) != record["crop_sha256"]):
            raise ValueError("Staged crop differs from the original image viewport")
        frames.append(SimpleNamespace(image_path=crop, mask_path=dummy))
    if sha(dummy) != receipt["dummy_mask_sha256"]:
        raise ValueError("Shape mask changed during verification")
    return receipt, frames


def run(stage: Path, checkpoint: Path, output: Path, smoke: bool, gate: Path | None) -> None:
    stage, checkpoint, output = stage.resolve(strict=True), checkpoint.resolve(strict=True), output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError("Output must be a new Drive experiment directory")
    if not checkpoint.is_relative_to(OUTPUT_ROOT) or sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint must come from the Drive experiment assets")
    if (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0" or os.environ.get("PYTHONHASHSEED") != "0"
            or not torch.backends.mps.is_available()):
        raise RuntimeError("An available MPS device with CPU fallback disabled is required")
    with exclusive_lock(LOCK):
        guard = install_mask_open_guard(ROOT / "data/polypgen_sequence.jsonl")
        disk = os.statvfs(OUTPUT_ROOT)
        if disk.f_bavail * disk.f_frsize < RESERVE:
            raise RuntimeError("Insufficient free space; no inference started")
        receipt, frames = verify_stage(stage)
        identity = {"stage_sha256": sha(stage / "stage.json"), "checkpoint_sha256": sha(checkpoint),
                    "runner_sha256": sha(Path(__file__)), "adapter_sha256": sha(ROOT / "scripts/run_cmsa_polypgen.py"),
                    "author_test_sha256": sha(AUTHOR / "scripts/my_test.py"), "author_revision": REVISION,
                    "statistics_sha256": sha(AUTHOR / "lib/dataloader/statistics.pth"),
                    "guard_sha256": sha(ROOT / "scripts/source_group_label_guard_v1.py"),
                    "verifier_sha256": sha(ROOT / "scripts/verify_cmsa_seq23_viewport_gate_v1.py"),
                    "torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
                    "platform": platform.platform(), "device": "mps", "dtype": "float32", "normalization": "released",
                    "hash_seed": os.environ["PYTHONHASHSEED"], "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                    "fast_math": os.environ["PYTORCH_MPS_FAST_MATH"], "dependencies": dependency_hashes()}
        gate_sha = None
        if not smoke:
            if gate is None:
                raise ValueError("Full run requires a prior completed smoke gate")
            g = read(gate)
            gate_sha = sha(gate)
            if (not gate.resolve(strict=True).is_relative_to(OUTPUT_ROOT) or g.get("passed") is not True
                    or g.get("identity") != identity or g.get("verifier_sha256") != identity["verifier_sha256"]
                    or g.get("smoke_receipt_sha256") != sha(Path(g["smoke_receipt"]))):
                raise ValueError("Smoke gate is not bound to these inputs and sources")
            from scripts.verify_cmsa_seq23_viewport_gate_v1 import audit
            if audit(stage, Path(g["smoke_receipt"]).parent) != g:
                raise ValueError("Smoke gate cannot be reconstructed from its artifacts")
        frames = frames[:4] if smoke else frames
        author = load_author()
        dataset = author_dataset({"seq23": frames}, "released")
        model = author.CMSANet()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model = model.to("mps").eval()
        state_sha = model_state_hash(model)
        if not smoke and g.get("state_sha256") != state_sha:
            raise ValueError("Live model state differs from verified smoke")
        calls = {"finite_calls": 0}
        def check_finite(_module, _inputs, outputs):
            disk = os.statvfs(OUTPUT_ROOT)
            if disk.f_bavail * disk.f_frsize < RESERVE:
                raise RuntimeError("Disk reserve reached during inference")
            if not isinstance(outputs, tuple) or len(outputs) != 3:
                raise RuntimeError("Unexpected CMSA model output")
            for tensor in outputs:
                if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
                    raise RuntimeError("Non-finite or unexpected CMSA feature/logit")
            calls["finite_calls"] += 1
        hook = model.register_forward_hook(check_finite)
        test = author.VPSTest_WithTestingTimeReferUpdate_2Ref.__new__(author.VPSTest_WithTestingTimeReferUpdate_2Ref)
        test.data_root = str(stage / "inputs")
        test.save_dir = str(output / "predictions")
        test.test_dataset = ["PolypGen"]
        test.dataloader = {"PolypGen": DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)}
        test.model = model
        output.mkdir(parents=True)
        try:
            with patch.object(torch.Tensor, "cuda", lambda tensor, *unused, **kwargs: tensor.to("mps")):
                test.test()
        finally:
            hook.remove()
        if calls["finite_calls"] != len(frames) - 3:
            raise RuntimeError("Author clip coverage differs")
        records = []
        for i in range(len(frames)):
            prediction = output / "predictions" / "seq23" / f"{i:06d}.png"
            with Image.open(prediction) as image:
                if image.format != "PNG" or image.size != (834, 720):
                    raise RuntimeError(f"Invalid prediction geometry: {prediction}")
            records.append({"frame": i, "prediction_sha256": sha(prediction)})
        if sha(stage / "stage.json") != identity["stage_sha256"] or sha(checkpoint) != identity["checkpoint_sha256"]:
            raise RuntimeError("Inputs changed during inference")
        verify_stage(stage)
        if not smoke and (sha(gate) != gate_sha or audit(stage, Path(g["smoke_receipt"]).parent) != g):
            raise RuntimeError("Smoke evidence changed during inference")
        disk = os.statvfs(OUTPUT_ROOT)
        revision = subprocess.check_output(['git', '-C', str(AUTHOR.parent), 'rev-parse', 'HEAD'], text=True).strip()
        clean = subprocess.run(['git', '-C', str(AUTHOR.parent), 'diff', '--quiet', 'HEAD', '--', 'cmsa_project'])
        if (guard["open_attempts"] or model_state_hash(model) != state_sha or revision != REVISION
                or clean.returncode != 0 or disk.f_bavail * disk.f_frsize < RESERVE
                or dependency_hashes() != identity["dependencies"]
                or sha(AUTHOR / "lib/dataloader/statistics.pth") != identity["statistics_sha256"]
                or sha(AUTHOR / "scripts/my_test.py") != identity["author_test_sha256"]
                or sha(Path(__file__)) != identity["runner_sha256"]
                or sha(ROOT / "scripts/verify_cmsa_seq23_viewport_gate_v1.py") != identity["verifier_sha256"]):
            raise RuntimeError("Annotation open attempted or model state changed")
        report = {"passed": True, "smoke": smoke, "identity": identity, "state_sha256": state_sha,
                  "label_guard": guard, "records": records,
                  "gate": str(gate.resolve()) if gate is not None else None, "gate_sha256": gate_sha,
                  "finite_clip_calls": calls["finite_calls"], "labels": "Generated black shape image only; no GT masks loaded.",
                  "scope": "Released CMSA loop and preprocessing; unprompted clip model, not streaming-equivalent. Python open guard does not intercept arbitrary native-code I/O."}
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "smoke": smoke, "frames": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    run(args.stage, args.checkpoint, args.output, args.smoke, args.gate)

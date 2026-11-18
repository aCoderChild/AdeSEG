#!/usr/bin/env python3
"""Train a reconstructed MedSAM2 Hiera-T/512 adapter from frozen TRAIN captures.

This is NEW training code, not the original clip trainer used in the reports.
No result from this script is eligible to inherit the old study's Dice score.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from modeling.medical_highres import MedicalHighResolutionAdapter
from modeling.medsam2_latest import (
    MEDSAM2_LATEST_SHA256, build_verified_medsam2_latest,
)
from training.medical_highres_training import (
    validate_decoder_capture, medical_training_step,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_manifest(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Require nonempty JSON list of TRAIN-only captures")
    paths = []
    for row in rows:
        if (not isinstance(row, dict) or row.get("split") != "train"
                or not isinstance(row.get("file"), str)
                or not isinstance(row.get("sequence"), str)
                or not row["sequence"]):
            raise ValueError("Each capture requires split=train, sequence, file")
        filepath = (path.parent / row["file"]).resolve(strict=True)
        if filepath.suffix != ".pt" or not filepath.is_file():
            raise ValueError("Training capture must be a .pt file")
        paths.append(filepath)
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate training capture")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SHA-verified MedSAM2_latest.pt pretrained weights")
    p.add_argument("--model-config", default="sam2.1_hiera_t512.yaml",
                   help="Config from the pinned vendored MedSAM2 package")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--teacher-weight", type=float, default=0.1)
    p.add_argument("--positive-weight", type=float, default=0.0)
    p.add_argument("--maximum-delta", type=float, default=0.1,
                   help="New reconstruction hyperparameter, not historical parity")
    a = p.parse_args(argv)
    if (a.epochs < 1 or a.learning_rate <= 0 or a.learning_rate >= 1
            or not 0 <= a.teacher_weight <= 10
            or not 0 <= a.positive_weight <= 10):
        p.error("Invalid hyperparameter")
    a.manifest = a.manifest.resolve(strict=True)
    paths = load_train_manifest(a.manifest)
    if a.output.exists():
        p.error("Refusing to overwrite checkpoint")
    torch.manual_seed(20261118)
    predictor = build_verified_medsam2_latest(
        a.checkpoint, model_cfg=a.model_config,
        device=a.device, apply_postprocessing=False)
    decoder = predictor.sam_mask_decoder.eval()
    for weight in predictor.parameters():
        weight.requires_grad_(False)
    adapter = MedicalHighResolutionAdapter(
        maximum_delta=a.maximum_delta).to(a.device).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=a.learning_rate)
    losses = []
    for _ in range(a.epochs):
        for filepath in paths:
            capture = torch.load(
                filepath, map_location="cpu", weights_only=True)
            validate_decoder_capture(capture)
            capture = {
                k: v.to(a.device, dtype=torch.float32)
                for k, v in capture.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, components = medical_training_step(
                decoder, adapter, capture,
                teacher_weight=a.teacher_weight,
                positive_weight=a.positive_weight)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite medical training loss")
            loss.backward()
            if any(p.grad is not None and not p.grad.isfinite().all().item()
                   for p in adapter.parameters()):
                raise FloatingPointError("Nonfinite medical adapter gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": "reconstructed_medsam2_highres_teacher_v1",
        "reconstruction_not_historical_forward": True,
        "backbone_sha256": MEDSAM2_LATEST_SHA256,
        "source_sha256": sha256(
            Path(__file__).resolve().parents[1] /
            "modeling/medical_highres/adapter.py"),
        "manifest_sha256": sha256(a.manifest),
        "model_config": a.model_config,
        "teacher_weight": a.teacher_weight,
        "positive_weight": a.positive_weight,
        "maximum_delta": a.maximum_delta,
        "training_steps": len(losses),
        "state_dict": {
            k: v.detach().cpu() for k, v in adapter.state_dict().items()},
    }
    torch.save(record, a.output)
    print(json.dumps({
        "checkpoint": str(a.output), "steps": len(losses),
        "last_training_loss": losses[-1],
        "status": "reconstruction_not_validated"}, indent=2))


if __name__ == "__main__":
    main()

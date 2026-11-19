#!/usr/bin/env python3
"""Offline supervised channel-value adapter training from frozen TRAIN captures.

Not a raw-video trainer or an evaluation pipeline. Labels must be train-only.
"""
import argparse
import hashlib
import json
import random
import numpy as np
from pathlib import Path

from training.split_guard import validate_train_sources
from training.training_provenance import training_source_digest
from training.restart_state import (
    capture_rng, restore_rng, save_epoch_snapshot, load_epoch_snapshot,
)
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adeseg.adapters.multiscale_channel_attention import MultistageFeatureAggregator
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from adeseg.losses.foreground_preservation import positive_preservation_penalty
import modeling.multiscale_channel_value_refiner_v1 as model_source
from modeling.lean_implementation_digest import implementation_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_training_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest must be a nonempty JSON array")
    resolved = []
    for row in data:
        if not isinstance(row, dict) or row.get("split") != "train" or not isinstance(row.get("tensor_file"), str):
            raise ValueError("Every row must refer explicitly to a training capture")
        source = (path.parent / row["tensor_file"]).resolve()
        if not source.is_file() or source.suffix != ".pt":
            raise ValueError(f"Missing training capture: {source}")
        resolved.append(source)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate training captures")
    return resolved


def load_training_capture(path, device):
    x = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(x, dict) or set(x) != {"early", "deep", "logits", "target"}:
        raise ValueError(f"Bad feature capture schema: {path}")
    if any(not isinstance(v, torch.Tensor) or not torch.is_floating_point(v) or v.ndim != 4
           or not torch.isfinite(v).all() for v in x.values()):
        raise ValueError(f"Capture contains invalid tensors: {path}")
    if x["target"].shape != x["logits"].shape or x["target"].shape[1] != 1 or x["target"].shape[0] != 1:
        raise ValueError("Target and logits must be aligned 1x1xHxW tensors")
    if (x["target"] < 0).any() or (x["target"] > 1).any():
        raise ValueError("Training target must be between zero and one")
    return {k: v.to(device=device, dtype=torch.float32) for k, v in x.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True,
                   help="Required independent source-group partition and capture SHA256 guard")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone-sha256", required=True)
    p.add_argument("--feature-pipeline-sha256", required=True)
    p.add_argument("--mode", choices=MultistageFeatureAggregator.MODES,
                   default="foreground_background_asinh")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--teacher-weight", type=float, default=0.0)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--save-epoch-snapshots", action="store_true")
    a = p.parse_args(argv)
    if a.epochs < 1 or not 0 < a.lr < 1 or not 0 <= a.teacher_weight <= 100:
        p.error("Invalid optimizer or teacher-preservation setting")
    for value in (a.backbone_sha256, a.feature_pipeline_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            p.error("Source hashes must be lowercase 64-hex SHA256 digests")
    if bool(a.resume_checkpoint) != bool(a.resume_sha256):
        p.error("--resume-checkpoint and --resume-sha256 must be provided together")
    a.manifest = a.manifest.resolve(strict=True)
    validate_train_sources(a.manifest, a.split_file, file_field="tensor_file")
    captures = read_training_manifest(a.manifest)
    if a.output.exists():
        p.error("Refusing to overwrite an existing checkpoint")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261118)
    random.seed(20261118)
    np.random.seed(20261118)
    model = MultistageFeatureAggregator(mode=a.mode).to(a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    fingerprint = {
        "format": "multiscale_channel_value_refiner_v1", "research_only": True,
        "source_sha256": digest(model_source.__file__),
        "implementation_sha256": implementation_digest(),
        "training_pipeline_sha256": training_source_digest("lean"),
        "mode": a.mode, "config": {"early_channels": 32, "deep_channels": 256,
                                    "hidden": 64, "attention_size": 32},
        "backbone_sha256": a.backbone_sha256,
        "feature_pipeline_sha256": a.feature_pipeline_sha256,
        "train_manifest_sha256": digest(a.manifest),
        "split_file_sha256": digest(a.split_file),
        "teacher_weight": a.teacher_weight, "lr": a.lr,
        "seed": 20261118, "device": a.device,
    }
    steps, losses, start_epoch = 0, [], 0
    if a.resume_checkpoint:
        previous = load_epoch_snapshot(
            a.resume_checkpoint, a.resume_sha256, expected=fingerprint)
        start_epoch = previous["completed_epochs"]
        if (start_epoch >= a.epochs or previous["training_steps"] != start_epoch * len(captures)
                or len(previous["losses"]) != previous["training_steps"]):
            raise ValueError("Resume epoch/steps incompatible with total epochs or captures")
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        restore_rng(previous["rng_state"])
        steps, losses = previous["training_steps"], list(previous["losses"])

    def checkpoint_record(completed_epochs):
        return {
            **fingerprint, "completed_epochs": completed_epochs,
            "epochs": completed_epochs, "training_steps": steps,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_state": capture_rng(), "losses": list(losses),
        }

    for epoch in range(start_epoch, a.epochs):
        for path in captures:
            batch = load_training_capture(path, a.device)
            optimizer.zero_grad(set_to_none=True)
            refined = model(batch["early"], batch["deep"], batch["logits"])
            loss = cmsa_boundary_loss(refined, batch["target"])
            if a.teacher_weight:
                loss = loss + a.teacher_weight * positive_preservation_penalty(
                    refined, batch["logits"], batch["target"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss: {path}")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite gradients: {path}")
            optimizer.step()
            steps += 1
            losses.append(loss.detach().item())
        if a.save_epoch_snapshots:
            snapshot_path = a.output.with_name(a.output.name + f".epoch{epoch+1:04d}.pt")
            save_epoch_snapshot(snapshot_path, checkpoint_record(epoch + 1))
    save_epoch_snapshot(a.output, checkpoint_record(a.epochs))
    print(json.dumps({"checkpoint": str(a.output), "steps": steps,
                      "last_training_loss": losses[-1], "training_only": True}, indent=2))


if __name__ == "__main__":
    main()

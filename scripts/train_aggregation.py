#!/usr/bin/env python3
"""Offline supervised channel-value adapter training from frozen TRAIN captures.

Not a raw-video trainer or an evaluation pipeline. Labels must be train-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from modeling.training_foreground_preservation_v1 import positive_preservation_penalty
import modeling.multiscale_channel_value_refiner_v1 as model_source


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone-sha256", required=True)
    p.add_argument("--feature-pipeline-sha256", required=True)
    p.add_argument("--mode", choices=MultiscaleChannelValueRefinerV1.MODES,
                   default="foreground_background_asinh")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--teacher-weight", type=float, default=0.0)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    a = p.parse_args()
    if a.epochs < 1 or not 0 < a.lr < 1 or not 0 <= a.teacher_weight <= 100:
        p.error("Invalid optimizer or teacher-preservation setting")
    for value in (a.backbone_sha256, a.feature_pipeline_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            p.error("Source hashes must be lowercase 64-hex SHA256 digests")
    a.manifest = a.manifest.resolve(strict=True)
    captures = read_training_manifest(a.manifest)
    if a.output.exists():
        p.error("Refusing to overwrite an existing checkpoint")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261118)
    model = MultiscaleChannelValueRefinerV1(mode=a.mode).to(a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps, losses = 0, []
    for epoch in range(a.epochs):
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
    payload = {
        "format": "multiscale_channel_value_refiner_v1", "research_only": True,
        "training_steps": steps, "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
        "source_sha256": digest(model_source.__file__), "mode": a.mode,
        "config": {"early_channels":32, "deep_channels":256, "hidden":64, "attention_size":32},
        "backbone_sha256": a.backbone_sha256, "feature_pipeline_sha256": a.feature_pipeline_sha256,
        "train_manifest_sha256": digest(a.manifest),
        "teacher_weight": a.teacher_weight, "epochs": a.epochs,
    }
    torch.save(payload, a.output)
    print(json.dumps({"checkpoint": str(a.output), "steps": steps,
                      "last_training_loss": losses[-1], "training_only": True}, indent=2))


if __name__ == "__main__":
    main()

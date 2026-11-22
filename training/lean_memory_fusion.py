"""Train early-FPN memory fusion from sealed TRAIN-only memory capture pairs.

This is supervised *same-frame oracle-memory distillation*, not end-to-end
video BPTT. Oracle masks are used solely to construct frozen TRAIN captures;
the learned module never takes GT during inference. The resulting checkpoint
must still pass a real-video paired Dice evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from modeling.lean_memory_feature_fusion import (
    EarlyFeatureMemoryFusion, MemoryFeatureFusionConfig, implementation_sha256,
)
from training.split_guard import sha256, validate_train_sources


TRAINING_PROTOCOL = "oracle_mask_memory_distillation_v2"
TEACHER_TARGET_POLICIES = ("feasible_clipped", "raw_oracle")


def training_source_sha256():
    hasher = hashlib.sha256()
    for source in (Path(__file__).resolve(),
                   Path(__file__).resolve().parents[1] / "training/split_guard.py"):
        value = source.read_bytes()
        hasher.update(source.name.encode() + b"\x00" + value)
    return hasher.hexdigest()


def read_capture(path: Path, config: MemoryFeatureFusionConfig):
    """Fail closed on untrusted/malformed precomputed captures."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"early", "native_memory", "teacher_memory",
                "predicted_mask_logits", "object_score_logits",
                "video_id", "frame_index"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"Invalid memory capture schema: {path}")
    tensors = [data[k] for k in (
        "native_memory", "teacher_memory",
        "predicted_mask_logits", "object_score_logits")]
    early = data["early"]
    if not isinstance(early, (tuple, list)) or len(early) != len(config.early_channels):
        raise ValueError("Capture is missing the configured early feature levels")
    tensors.extend(early)
    if any(not isinstance(x, torch.Tensor) or not torch.is_floating_point(x)
           or x.dtype != torch.float32 or not bool(torch.isfinite(x).all().item())
           for x in tensors):
        raise ValueError("Captures must contain only finite float32 tensors")
    native, teacher, mask, presence = tensors[:4]
    if not (native.ndim == 4 and native.shape[0:2] == (1, config.memory_channels)
            and teacher.shape == native.shape
            and native.shape[-2] >= 1 and native.shape[-1] >= 1
            and mask.shape[0:2] == (1, 1)
            and mask.ndim == 4 and mask.shape[-2:] == native.shape[-2:]
            and presence.numel() == 1):
        raise ValueError("Capture memory/mask/presence geometry mismatch")
    for index, feature in enumerate(early):
        if (feature.ndim != 4 or feature.shape[0:2] !=
                (1, config.early_channels[index])
                or feature.shape[-2:] != native.shape[-2:]):
            raise ValueError(f"Early capture feature geometry mismatch at {index}")
    if (not isinstance(data["video_id"], str) or not data["video_id"]
            or type(data["frame_index"]) is not int
            or data["frame_index"] < 0):
        raise ValueError("Capture frame provenance missing")
    return data


def audit_teacher_feasibility(captures, config):
    """Quantify when an oracle memory lies outside the adapter's reachable set.

    The residual is bounded elementwise by max_relative_residual * per-entry
    native-memory standard deviation, and is exactly zero for a predicted
    absent object when preserve_native_no_object is enabled. This diagnostic
    is descriptive TRAIN evidence; it is not a robustness or accuracy metric.
    """
    if not captures:
        raise ValueError("No captures for feasibility audit")
    reports = []
    for capture in captures:
        native = capture["native_memory"].float()
        target = capture["teacher_memory"].float()
        if native.shape != target.shape or native.ndim != 4:
            raise ValueError("Teacher/native memory shape mismatch")
        scale = (
            native.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
            .clamp_min(1e-6) * config.max_relative_residual
        )
        diff = (target - native).abs()
        ratio = diff / scale
        hard_absent = (
            config.preserve_native_no_object
            and not bool((capture["object_score_logits"].float() > 0).all().item())
        )
        unattainable = (diff > scale) | (hard_absent & (diff > 0))
        reports.append({
            "video_id": capture["video_id"],
            "frame_index": capture["frame_index"],
            "predicted_absent": bool(hard_absent),
            "max_required_to_allowed_ratio": float(ratio.max().item()),
            "fraction_unattainable_elements": float(unattainable.float().mean().item()),
            "fraction_gt_native_memory_elements_changed_after_bf16": float(
                (target.to(torch.bfloat16) != native.to(torch.bfloat16))
                .float().mean().item()
            ),
        })
    return {
        "capture_count": len(reports),
        "fraction_captures_with_unattainable_target": (
            sum(x["fraction_unattainable_elements"] > 0 for x in reports) / len(reports)
        ),
        "mean_fraction_unattainable_elements": (
            sum(x["fraction_unattainable_elements"] for x in reports) / len(reports)
        ),
        "fraction_predicted_absent": (
            sum(x["predicted_absent"] for x in reports) / len(reports)
        ),
        "max_required_to_allowed_ratio": max(
            x["max_required_to_allowed_ratio"] for x in reports
        ),
        "captures": reports,
    }


def feasible_teacher_memory(native, teacher, presence, config, *,
                            margin=0.95):
    """Project oracle-memory corrections into the representable residual box.

    This projection is the elementwise Euclidean projection onto
    [native - margin*alpha*std(native), native + margin*alpha*std(native)].
    margin<1 avoids asking a tanh residual to reach its unattainable endpoints.
    No-object entries are fixed points when the native-preservation gate is on.
    """
    if not 0.0 < margin < 1.0:
        raise ValueError("Expected strict 0<margin<1 for finite tanh targets")
    if (not isinstance(native, torch.Tensor) or teacher.shape != native.shape
            or native.ndim != 4 or presence.numel() != native.shape[0]):
        raise ValueError("Invalid teacher projection tensors")
    limit = native.detach().float().std(
        dim=(1, 2, 3), unbiased=False, keepdim=True
    ).clamp_min(1e-6) * (config.max_relative_residual * margin)
    target = native + (teacher - native).clamp(-limit, limit)
    if config.preserve_native_no_object:
        gate = (presence.reshape(-1).float() > 0).reshape(-1, 1, 1, 1)
        target = torch.where(gate, target, native)
    if not bool(torch.isfinite(target).all().item()):
        raise FloatingPointError("Nonfinite projected memory teacher")
    return target


def train(args):
    manifest = args.manifest.resolve(strict=True)
    partition = args.split_file.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".receipt.json").exists():
        raise FileExistsError("Refusing to overwrite trained memory fusion checkpoint")
    input_mode = getattr(args, "input_mode", "full")
    target_policy = getattr(args, "teacher_target_policy", "feasible_clipped")
    if target_policy not in TEACHER_TARGET_POLICIES:
        raise ValueError("Unknown oracle-memory teacher target policy")
    if input_mode not in EarlyFeatureMemoryFusion.INPUT_MODES:
        raise ValueError("Invalid memory-fusion input mode")
    if args.epochs < 1 or not 0 < args.lr < 1 or not 0 <= args.preservation_weight <= 10:
        raise ValueError("Invalid training hyperparameters")
    if (not isinstance(args.backbone_sha256, str)
            or len(args.backbone_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.backbone_sha256)):
        raise ValueError("Invalid pinned backbone SHA256")
    before = validate_train_sources(manifest, partition, file_field="tensor_file")
    allow_absent_fusion = getattr(args, "allow_no_object_memory_fusion", False)
    if type(allow_absent_fusion) is not bool:
        raise ValueError("No-object fusion ablation flag must be boolean")
    # The conservative default preserves native no-object memory exactly.
    # An explicitly named independently trained ablation may learn updates
    # on absent-object predictions, including genuine false-negative frames.
    config = MemoryFeatureFusionConfig(
        preserve_native_no_object=not allow_absent_fusion)
    rows = json.loads(manifest.read_text("utf8"))
    captures = []
    producer_sha = sha256(Path(__file__).resolve().parents[1] / "scripts/capture_lean_memory_train.py")
    for row in rows:
        if row.get("capture_source_sha256") != producer_sha:
            raise ValueError("Training capture producer source hash mismatch")
        path = (manifest.parent / row["tensor_file"]).resolve(strict=True)
        obj = read_capture(path, config)
        if (row.get("video_id") != obj["video_id"]
                or row.get("frame_index") != obj["frame_index"]):
            raise ValueError("Manifest frame does not match capture provenance")
        captures.append(obj)
    if not captures:
        raise ValueError("At least one genuine TRAIN capture is required")
    teacher_feasibility = audit_teacher_feasibility(captures, config)
    # A hard no-object gate makes the adapter output exactly native memory
    # for predicted-absent captures. Their teacher loss is independent of
    # adapter weights, so they cannot supply useful training gradients.
    trainable_capture_count = sum(
        bool((c["object_score_logits"].reshape(-1) > 0).all().item())
        for c in captures
    ) if config.preserve_native_no_object else len(captures)
    if not trainable_capture_count:
        raise ValueError(
            "No learnable memory captures: every predicted object is absent "
            "while native no-object preservation is enabled")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = EarlyFeatureMemoryFusion(config).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    steps = 0
    skipped_absent_updates = 0
    for epoch in range(args.epochs):
        order = list(range(len(captures)))
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            c = captures[index]
            if (config.preserve_native_no_object
                    and not bool((c["object_score_logits"].reshape(-1) > 0).all().item())):
                skipped_absent_updates += 1
                continue
            early = [x.to(args.device) for x in c["early"]]
            native = c["native_memory"].to(args.device)
            teacher = c["teacher_memory"].to(args.device)
            mask = c["predicted_mask_logits"].to(args.device)
            presence = c["object_score_logits"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(early, native, mask, presence, input_mode=input_mode)
            target = (feasible_teacher_memory(native, teacher, presence, config)
                      if target_policy == "feasible_clipped" else teacher)
            reconstruction = F.smooth_l1_loss(predicted, target)
            preservation = F.smooth_l1_loss(predicted, native)
            loss = reconstruction + args.preservation_weight * preservation
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite memory-distillation training loss")
            loss.backward()
            if any(param.grad is not None and not bool(torch.isfinite(param.grad).all().item())
                   for param in model.parameters()):
                raise FloatingPointError("Non-finite memory-fusion gradients")
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
    if model.is_identity_initialized():
        raise RuntimeError("No learned memory residual: training did not update output projection")
    after = validate_train_sources(manifest, partition, file_field="tensor_file")
    if after != before:
        raise RuntimeError("Training sources changed during optimization")
    payload = {
        "format": model.CHECKPOINT_FORMAT,
        "config": asdict(config),
        "implementation_sha256": implementation_sha256(),
        "backbone_sha256": args.backbone_sha256,
        "train_manifest_sha256": sha256(manifest),
        "split_file_sha256": sha256(partition),
        "training_protocol": TRAINING_PROTOCOL,
        "teacher_target_policy": target_policy,
        "fusion_input_mode": input_mode,
        "no_object_fusion_ablation": bool(allow_absent_fusion),
        "training_source_sha256": training_source_sha256(),
        "capture_source_sha256": producer_sha,
        "training_steps": steps,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "preservation_weight": args.preservation_weight,
        "capture_count": len(captures),
        "trainable_capture_count": trainable_capture_count,
        "skipped_predicted_absent_updates": skipped_absent_updates,
        "source_groups": before["train_source_groups"],
        "last_train_loss": losses[-1],
        "teacher_feasibility": teacher_feasibility,
        "model_state_dict": {k: v.detach().float().cpu().clone()
                             for k, v in model.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    receipt = {
        "checkpoint": str(output), "checkpoint_sha256": sha256(output),
        "training_protocol": TRAINING_PROTOCOL,
        "teacher_target_policy": target_policy,
        "fusion_input_mode": input_mode,
        "no_object_fusion_ablation": bool(allow_absent_fusion),
        "training_steps": steps,
        "source_groups": before["train_source_groups"],
        "train_manifest_sha256": sha256(manifest),
        "split_file_sha256": sha256(partition),
        "implementation_sha256": payload["implementation_sha256"],
        "training_source_sha256": payload["training_source_sha256"],
        "capture_count": len(captures),
        "trainable_capture_count": trainable_capture_count,
        "skipped_predicted_absent_updates": skipped_absent_updates,
        "last_train_loss": losses[-1],
        "teacher_feasibility": teacher_feasibility,
        "warning": "TRAIN oracle-memory distillation; no future-frame or TEST Dice measured",
    }
    Path(str(output) + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone-sha256", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--input-mode", choices=EarlyFeatureMemoryFusion.INPUT_MODES,
                   default="full", help="Train each control separately with a matched architecture")
    p.add_argument("--allow-no-object-memory-fusion", action="store_true",
                   help="Separate risk ablation: train memory residuals on predicted-absent frames")
    p.add_argument("--teacher-target-policy",
                   choices=TEACHER_TARGET_POLICIES,
                   default="feasible_clipped",
                   help="Feasible bounded teacher or separately named raw-oracle control")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--preservation-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    result = train(p.parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

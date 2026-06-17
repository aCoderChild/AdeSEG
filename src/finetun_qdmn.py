"""
Fine-tune a QDMN-style reliability scorer for SAM2/MedSAM2 predictions.

The reliability model learns:

    input:  image frame + SAM2 predicted mask
    output: reliability score in [0, 1]
    target: IoU(SAM2 predicted mask, ground-truth mask)

Expected directory layout can be either matching relative paths:

    train/images/video_001/00010.png
    train/pred_masks/video_001/00010.png
    train/gt_masks/video_001/00010.png

or matching filename stems somewhere under each root.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or run "
            "without `--wandb`."
        ) from exc
    return wandb


@dataclass
class Sample:
    image_path: str
    pred_mask_path: str
    gt_mask_path: str


class MaskReliabilityDataset(Dataset):
    """Loads image, SAM2 predicted mask, and GT mask triples."""

    def __init__(
        self,
        image_root: str,
        pred_mask_root: str,
        gt_mask_root: str,
        image_size: int = 384,
        binary_target_threshold: float | None = None,
    ):
        self.image_root = Path(image_root)
        self.pred_mask_root = Path(pred_mask_root)
        self.gt_mask_root = Path(gt_mask_root)
        self.image_size = image_size
        self.binary_target_threshold = binary_target_threshold
        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples found. Check that image, pred-mask, and GT-mask paths "
                "share either relative paths or filename stems."
            )

    def _collect_samples(self) -> list[Sample]:
        pred_by_rel = _index_by_relative_path(self.pred_mask_root)
        gt_by_rel = _index_by_relative_path(self.gt_mask_root)
        pred_by_stem = _index_by_stem(self.pred_mask_root)
        gt_by_stem = _index_by_stem(self.gt_mask_root)

        samples = []
        for image_path in _iter_files(self.image_root):
            rel_key = image_path.relative_to(self.image_root).as_posix()
            stem_key = image_path.stem
            pred_path = pred_by_rel.get(rel_key) or pred_by_stem.get(stem_key)
            gt_path = gt_by_rel.get(rel_key) or gt_by_stem.get(stem_key)
            if pred_path is None or gt_path is None:
                continue
            samples.append(
                Sample(
                    image_path=str(image_path),
                    pred_mask_path=str(pred_path),
                    gt_mask_path=str(gt_path),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = _load_image(sample.image_path, self.image_size)
        pred_mask = _load_mask(sample.pred_mask_path, self.image_size)
        gt_mask = _load_mask(sample.gt_mask_path, self.image_size)

        target_iou = _mask_iou(pred_mask, gt_mask)
        target = target_iou
        if self.binary_target_threshold is not None:
            target = (target_iou >= self.binary_target_threshold).float()

        return {
            "image": image,
            "pred_mask": pred_mask,
            "gt_mask": gt_mask,
            "target": target.view(1),
            "target_iou": target_iou.view(1),
            "image_path": sample.image_path,
        }


class TinyReliabilityNet(nn.Module):
    """Small fallback scorer. Useful for smoke tests and quick ablations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        x = self.net(x).flatten(1)
        return torch.sigmoid(self.head(x))


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _index_by_relative_path(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in _iter_files(root)}


def _index_by_stem(root: Path) -> dict[str, Path]:
    index = {}
    for path in _iter_files(root):
        index.setdefault(path.stem, path)
    return index


def _load_image(path: str, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _load_mask(path: str, size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    return (tensor >= 0.5).float()


def _mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6):
    pred = pred_mask >= 0.5
    gt = gt_mask >= 0.5
    intersection = torch.logical_and(pred, gt).float().sum()
    union = torch.logical_or(pred, gt).float().sum()
    return intersection / (union + eps)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "tiny":
        return TinyReliabilityNet()

    from reliability_score.model.reliabilityScoreNet import ReliabilityScoreNet

    return ReliabilityScoreNet(
        single_object=True,
        pretrained=args.pretrained_backbone,
    )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    for key in ("model", "network", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def remap_qdmn_checkpoint_for_reliability_net(state_dict):
    """
    Original QDMN checkpoints are usually saved from PropagationNetwork.
    ReliabilityScoreNet only needs its mask RGB encoder and score head.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("mask_rgb_encoder."):
            remapped["encoder." + key[len("mask_rgb_encoder.") :]] = value
        elif key.startswith("score."):
            remapped[key] = value
    return remapped


def adapt_state_dict_to_model_shapes(model, state_dict):
    model_state = model.state_dict()
    adapted = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_state:
            adapted[key] = value
            continue

        target = model_state[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue

        can_adapt_first_conv = (
            key == "encoder.conv1.weight"
            and value.dim() == 4
            and target.dim() == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
        )
        if can_adapt_first_conv:
            if value.shape[1] > target.shape[1]:
                adapted[key] = value[:, : target.shape[1]]
            else:
                pad_channels = target.shape[1] - value.shape[1]
                pad = torch.zeros(
                    value.shape[0],
                    pad_channels,
                    value.shape[2],
                    value.shape[3],
                    dtype=value.dtype,
                    device=value.device,
                )
                adapted[key] = torch.cat([value, pad], dim=1)
            continue

        skipped.append((key, tuple(value.shape), tuple(target.shape)))

    if skipped:
        print(f"Skipped shape-mismatched keys: {len(skipped)}")
        for key, source_shape, target_shape in skipped[:5]:
            print(f"  {key}: checkpoint {source_shape} -> model {target_shape}")

    return adapted


def load_initial_qdmn_weights(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial QDMN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    remapped = remap_qdmn_checkpoint_for_reliability_net(state_dict)

    if len(remapped) == 0:
        remapped = state_dict

    remapped = adapt_state_dict_to_model_shapes(model, remapped)
    load_result = model.load_state_dict(remapped, strict=False)
    print(f"Loaded initial QDMN weights from {checkpoint_path}")
    if load_result.missing_keys:
        print(f"Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    return load_result


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_one_epoch(model, loader, optimizer, device, loss_name):
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_score = model(batch["image"], batch["pred_mask"]).view(-1, 1)
        target = batch["target"]

        if loss_name == "bce":
            loss = F.binary_cross_entropy(pred_score, target)
        else:
            loss = F.mse_loss(pred_score, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            total_abs_error += (pred_score - batch["target_iou"]).abs().sum().item()

    num_samples = len(loader.dataset)
    return {
        "loss": total_loss / num_samples,
        "mae_to_iou": total_abs_error / num_samples,
    }


@torch.no_grad()
def evaluate(model, loader, device, loss_name):
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_score = model(batch["image"], batch["pred_mask"]).view(-1, 1)
        target = batch["target"]

        if loss_name == "bce":
            loss = F.binary_cross_entropy(pred_score, target)
        else:
            loss = F.mse_loss(pred_score, target)

        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_abs_error += (pred_score - batch["target_iou"]).abs().sum().item()

    num_samples = len(loader.dataset)
    return {
        "loss": total_loss / num_samples,
        "mae_to_iou": total_abs_error / num_samples,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def log_checkpoint_to_wandb(wandb, checkpoint_path: Path, artifact_name: str, epoch: int):
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={"epoch": epoch},
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a QDMN reliability scorer from SAM2 predictions."
    )
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-pred-masks", required=True)
    parser.add_argument("--train-gt-masks", required=True)
    parser.add_argument("--val-images")
    parser.add_argument("--val-pred-masks")
    parser.add_argument("--val-gt-masks")
    parser.add_argument("--test-images")
    parser.add_argument("--test-pred-masks")
    parser.add_argument("--test-gt-masks")
    parser.add_argument("--output-dir", default="outputs/qdmn")
    parser.add_argument("--model", choices=["qdmn", "tiny"], default="qdmn")
    parser.add_argument(
        "--init-checkpoint",
        default="checkpoints/QDMN.pth",
        help=(
            "Initial QDMN checkpoint used before fine-tuning. Set to '' to train "
            "from initialization."
        ),
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Use ImageNet ResNet-50 weights for QDMN. May download weights.",
    )
    parser.add_argument("--resume")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["mse", "bce"], default="mse")
    parser.add_argument(
        "--binary-target-threshold",
        type=float,
        help="Use binary labels: 1 if IoU >= threshold else 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb-project", default="sam2-qdmn")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb-log-checkpoints",
        action="store_true",
        help="Upload best/last checkpoints as wandb artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    wandb = None
    if args.wandb:
        wandb = import_wandb()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(output_dir),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("best/*", step_metric="epoch")

    train_dataset = MaskReliabilityDataset(
        image_root=args.train_images,
        pred_mask_root=args.train_pred_masks,
        gt_mask_root=args.train_gt_masks,
        image_size=args.image_size,
        binary_target_threshold=args.binary_target_threshold,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    has_val = args.val_images and args.val_pred_masks and args.val_gt_masks
    val_loader = None
    if has_val:
        val_dataset = MaskReliabilityDataset(
            image_root=args.val_images,
            pred_mask_root=args.val_pred_masks,
            gt_mask_root=args.val_gt_masks,
            image_size=args.image_size,
            binary_target_threshold=args.binary_target_threshold,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    has_test = args.test_images and args.test_pred_masks and args.test_gt_masks
    test_loader = None
    if has_test:
        test_dataset = MaskReliabilityDataset(
            image_root=args.test_images,
            pred_mask_root=args.test_pred_masks,
            gt_mask_root=args.test_gt_masks,
            image_size=args.image_size,
            binary_target_threshold=args.binary_target_threshold,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
    elif args.model == "qdmn" and args.init_checkpoint:
        load_initial_qdmn_weights(model, args.init_checkpoint, device)

    best_val = math.inf
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test samples: {len(test_loader.dataset)}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.loss
        )
        metrics = {"train": train_metrics}

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args.loss)
            metrics["val"] = val_metrics
            score_for_best = val_metrics["mae_to_iou"]
        elif test_loader is not None:
            test_metrics = evaluate(model, test_loader, device, args.loss)
            metrics["test"] = test_metrics
            score_for_best = test_metrics["mae_to_iou"]
        else:
            score_for_best = train_metrics["mae_to_iou"]

        if val_loader is not None and test_loader is not None:
            test_metrics = evaluate(model, test_loader, device, args.loss)
            metrics["test"] = test_metrics

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_mae_iou={train_metrics['mae_to_iou']:.5f}"
            + (
                f" | val_loss={metrics['val']['loss']:.5f} "
                f"val_mae_iou={metrics['val']['mae_to_iou']:.5f}"
                if "val" in metrics
                else ""
            )
            + (
                f" | test_loss={metrics['test']['loss']:.5f} "
                f"test_mae_iou={metrics['test']['mae_to_iou']:.5f}"
                if "test" in metrics
                else ""
            )
        )

        if wandb is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/mae_to_iou": train_metrics["mae_to_iou"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            if "val" in metrics:
                log_payload.update(
                    {
                        "val/loss": metrics["val"]["loss"],
                        "val/mae_to_iou": metrics["val"]["mae_to_iou"],
                    }
                )
            if "test" in metrics:
                log_payload.update(
                    {
                        "test/loss": metrics["test"]["loss"],
                        "test/mae_to_iou": metrics["test"]["mae_to_iou"],
                    }
                )
            wandb.log(log_payload)

        last_checkpoint = output_dir / "last_qdmn.pth"
        save_checkpoint(last_checkpoint, model, optimizer, epoch, metrics, args)
        if wandb is not None and args.wandb_log_checkpoints:
            log_checkpoint_to_wandb(wandb, last_checkpoint, "last-qdmn", epoch)

        if score_for_best < best_val:
            best_val = score_for_best
            best_checkpoint = output_dir / "best_qdmn.pth"
            save_checkpoint(best_checkpoint, model, optimizer, epoch, metrics, args)
            if wandb is not None:
                wandb.log({"epoch": epoch, "best/mae_to_iou": best_val})
                if args.wandb_log_checkpoints:
                    log_checkpoint_to_wandb(wandb, best_checkpoint, "best-qdmn", epoch)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fine-tune the reliability-gated memory adapters on video sequences."""

from __future__ import annotations

import argparse
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file(PROJECT_ROOT / "src" / ".env")

from scripts.utils.model_defaults import DEFAULT_SAM2_CFG, DEFAULT_SAM2_CHECKPOINT, DEFAULT_SAM2_CODE_ROOT
from src import PromptBatch, QDMNScoreReliabilityEstimator, ReliabilityGatedDynamicMemorySAM
from src.infer_reliability_gated_video import get_frame_names, get_video_frame_dir, load_gt_bboxes, mask_prompt_tensor, resolve_frame_path
from src.training import set_reliability_score_finetune_mode

if DEFAULT_SAM2_CODE_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_CODE_ROOT)

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


def init_wandb(args, trainable_param_count: int):
    if args.disable_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it with `python -m pip install wandb` "
            "or rerun with `--disable_wandb`."
        ) from exc

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            **vars(args),
            "trainable_parameters": trainable_param_count,
            "frozen": [
                "MedSAM2 image_encoder",
                "MedSAM2 prompt_encoder",
                "MedSAM2 mask_decoder",
                "PromptMemoryAttention",
                "QDMN score",
            ],
            "trainable": [
                "QDMN reliability projection",
            ],
        },
    )
    return run


def get_numeric_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def load_binary_mask(path: str) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return (mask > (0 if mask.max() <= 1 else 127)).astype(np.float32)


def resolve_mask_path(mask_dir: str, frame_name: str) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(mask_dir, f"{frame_name}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(os.path.join(mask_dir, f"{frame_name}.png"))


def soft_iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).detach()
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).reshape(-1, 1)


def build_model(args):
    device = None if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available.")
    sam2_model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
        apply_postprocessing=args.apply_postprocessing,
    )
    reliability = QDMNScoreReliabilityEstimator(
        feature_dim=sam2_model.hidden_dim,
        qdmn_root=args.qdmn_root,
        checkpoint_path=args.qdmn_checkpoint,
    )
    model = ReliabilityGatedDynamicMemorySAM.from_sam2_base(
        sam2_model,
        concept_encoder=None,
        reliability_estimator=reliability,
        memory_attention_strength=args.memory_attention_strength,
    ).to(sam2_model.device)
    transforms = SAM2Transforms(resolution=sam2_model.image_size, mask_threshold=0.0)
    return model, transforms


def build_box_prompt(boxes, transforms, orig_hw, device):
    if not boxes:
        return None
    box = torch.as_tensor(boxes[0][0][None, :], dtype=torch.float32, device=device)
    return transforms.transform_boxes(box, normalize=True, orig_hw=orig_hw)


def train_sequence(
    model,
    transforms,
    args,
    seq_name: str,
    optimizer,
    epoch: int,
    wandb_run=None,
) -> tuple[float, int]:
    video_dir = get_video_frame_dir(args.data_root, seq_name)
    mask_dir = os.path.join(args.data_root, seq_name, "masks")
    frame_names = get_frame_names(video_dir)
    gt_boxes = load_gt_bboxes(args.bbox_root, seq_name)
    device = next(model.parameters()).device
    memory = None
    previous_pred_mask = None
    total_loss = 0.0
    num_frames = 0
    grad_frames = 0

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    frame_iter = tqdm(
        enumerate(frame_names),
        total=len(frame_names),
        desc=f"{seq_name} frames",
        leave=False,
        dynamic_ncols=True,
        disable=args.no_tqdm,
    )
    for frame_idx, frame_name in frame_iter:
        image = Image.open(resolve_frame_path(video_dir, frame_name)).convert("RGB")
        orig_hw = (image.height, image.width)
        image_tensor = transforms(image)[None].to(device)
        gt_mask_np = load_binary_mask(resolve_mask_path(mask_dir, frame_name))
        gt_mask = torch.as_tensor(gt_mask_np, device=device)[None, None]

        prompted = frame_idx % args.prompt_stride == 0
        boxes = gt_boxes.get(frame_name, [])
        if not boxes and frame_name.isdigit():
            boxes = gt_boxes.get(str(int(frame_name)), [])
        box_prompt = build_box_prompt(boxes, transforms, orig_hw, device) if prompted else None

        mask_prompt = None
        if not prompted and args.use_previous_mask_prompt and previous_pred_mask is not None:
            mask_prompt = mask_prompt_tensor(previous_pred_mask, model.prompt_encoder, device)

        prompts = PromptBatch(boxes=box_prompt, mask=mask_prompt)
        with autocast_ctx:
            output = model(image_tensor, memory=memory, prompts=prompts)
            logits = F.interpolate(output.mask_logits.float(), size=orig_hw, mode="bilinear", align_corners=False)
            target_reliability = soft_iou_score(logits, gt_mask)
            loss = F.mse_loss(output.reliability.float(), target_reliability.float())

        if loss.requires_grad:
            loss.backward()
            grad_frames += 1
            if grad_frames % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        num_frames += 1
        frame_iter.set_postfix(
            loss=f"{total_loss / max(num_frames, 1):.4f}",
            rel=f"{float(output.reliability.detach().flatten()[0].cpu()):.3f}",
            target=f"{float(target_reliability.detach().flatten()[0].cpu()):.3f}",
        )
        if wandb_run is not None and args.wandb_log_frames:
            wandb_run.log(
                {
                    "train/frame_reliability_loss": float(loss.detach().cpu()),
                    "train/predicted_reliability": float(output.reliability.detach().flatten()[0].cpu()),
                    "train/target_iou": float(target_reliability.detach().flatten()[0].cpu()),
                    "train/epoch": epoch,
                    "train/frame_idx": frame_idx,
                    "train/prompted": int(prompted),
                    "train/sequence": seq_name,
                }
            )
        memory = output.memory
        if memory.tensor is not None:
            memory.tensor = memory.tensor.detach()
        previous_pred_mask = (logits.detach()[0, 0] > args.score_thresh).float().cpu().numpy()

    if grad_frames % args.grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss, num_frames


def save_adapter_checkpoint(model, path: str, args, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "args": vars(args),
        "reliability_project": model.reliability_estimator.project.state_dict(),
    }
    torch.save(state, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/polypgen")
    parser.add_argument("--bbox_root", default="data/bbox")
    parser.add_argument("--seq_nums", type=int, nargs="*", default=[3])
    parser.add_argument("--sam2_cfg", default=DEFAULT_SAM2_CFG)
    parser.add_argument("--sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for fine-tuning. Use cuda on Kaggle GPU.",
    )
    parser.add_argument("--apply_postprocessing", action="store_true")
    parser.add_argument("--qdmn_root", default="external/QDMN")
    parser.add_argument("--qdmn_checkpoint", default="checkpoints/QDMN.pth")
    parser.add_argument("--prompt_stride", type=int, default=5)
    parser.add_argument("--memory_attention_strength", type=float, default=0.0)
    parser.add_argument("--use_previous_mask_prompt", action="store_true", default=True)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--output_checkpoint", default="checkpoints/reliability_score_project.pt")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "AdeSEG"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_log_frames", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, transforms = build_model(args)
    trainable = set_reliability_score_finetune_mode(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    seq_names = [f"seq{num}" for num in args.seq_nums]
    trainable_param_count = sum(p.numel() for p in trainable)
    wandb_run = init_wandb(args, trainable_param_count)
    print(f"Fine-tuning on {seq_names}; trainable parameters: {trainable_param_count:,}")
    print(f"Using device: {next(model.parameters()).device}")

    epoch_iter = tqdm(
        range(1, args.epochs + 1),
        desc="epochs",
        dynamic_ncols=True,
        disable=args.no_tqdm,
    )
    for epoch in epoch_iter:
        total_loss = 0.0
        total_frames = 0
        seq_iter = tqdm(
            seq_names,
            desc=f"epoch {epoch} sequences",
            leave=False,
            dynamic_ncols=True,
            disable=args.no_tqdm,
        )
        for seq_name in seq_iter:
            seq_loss, seq_frames = train_sequence(
                model,
                transforms,
                args,
                seq_name,
                optimizer,
                epoch,
                wandb_run=wandb_run,
            )
            total_loss += seq_loss
            total_frames += seq_frames
            seq_avg = seq_loss / max(seq_frames, 1)
            seq_iter.set_postfix(seq=seq_name, loss=f"{seq_avg:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/sequence_reliability_loss": seq_loss / max(seq_frames, 1),
                        "train/sequence_frames": seq_frames,
                        "train/epoch": epoch,
                        "train/sequence": seq_name,
                    }
                )
        avg_loss = total_loss / max(total_frames, 1)
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")
        print(f"epoch {epoch}/{args.epochs} - avg_reliability_loss={avg_loss:.6f}")
        save_adapter_checkpoint(model, args.output_checkpoint, args, epoch)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_reliability_loss": avg_loss,
                    "train/epoch": epoch,
                    "train/frames": total_frames,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
            wandb_run.save(args.output_checkpoint)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

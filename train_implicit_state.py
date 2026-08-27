#!/usr/bin/env python3
"""Train only the external single-state prompt adapter on frozen MedSAM2.

The trainer uses video-disjoint train/test sequences supplied on the command
line. SAM2's native memory stack and all SAM2 parameters remain frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
MEDSAM2_ROOT = PROJECT_ROOT / "MedSAM2"
for path in (PROJECT_ROOT, MEDSAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam2.build_sam import build_sam2_video_predictor

import infer_gated_reliability as inference
from modeling.implicit_state import ImplicitTemporalState
from utils.eval_metrics import get_bbox_from_mask
from utils.mask_utils import load_binary_mask, resolve_mask_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clips-per-sequence", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    return parser.parse_args()


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + dice.mean()


def freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_target(sequence_name: str, frame_name: str, shape: tuple[int, int], device: str) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
    mask_path = resolve_mask_path(inference.mask_dir_for_sequence(sequence_name), frame_name)
    if mask_path is None:
        raise FileNotFoundError(f"Missing mask for {sequence_name}/{frame_name}")
    mask = load_binary_mask(mask_path)
    box = get_bbox_from_mask(mask)
    target = torch.from_numpy(mask)[None, None].float().to(device)
    target = F.interpolate(target, size=shape, mode="nearest")
    return target, box


def sample_starts(frame_count: int, clip_length: int, prompt_stride: int, count: int, rng: random.Random) -> list[int]:
    max_start = max(0, frame_count - clip_length)
    candidates = list(range(0, max_start + 1, prompt_stride)) or [0]
    if len(candidates) <= count:
        return candidates
    return [0, *rng.sample(candidates[1:], k=count - 1)]


def train_clip(
    predictor,
    adapter: ImplicitTemporalState,
    optimizer: torch.optim.Optimizer,
    sequence_name: str,
    start: int,
    clip_length: int,
    prompt_stride: int,
    device: str,
) -> float:
    frame_dir = Path(inference.get_video_frame_dir(str(inference.DATA_ROOT), sequence_name))
    frame_names = inference.get_frame_names(str(frame_dir))
    state_dict = predictor.init_state(video_path=str(frame_dir))
    state = None
    losses = []
    stop = min(start + clip_length, len(frame_names))
    for frame_idx in range(start, stop):
        frame_name = frame_names[frame_idx]
        with torch.no_grad():
            target, box = load_target(sequence_name, frame_name, (128, 128), device)
            point_inputs = None
            if frame_idx % prompt_stride == 0 and box is not None:
                point_inputs = predictor.prepare_point_inputs(inference_state=state_dict, box=box)
            base_out, _, _ = inference.decode(
                predictor, state_dict, frame_idx, 1, point_inputs=point_inputs
            )
            image_features = inference.current_image_features(predictor, state_dict, frame_idx, 1)

        prompt_logits, state = adapter.forward_step(
            # ``run_single_frame_inference`` returns inference tensors. Clone
            # them here so the trainable adapter can retain its inputs for
            # backward while MedSAM2 itself stays frozen.
            mask_logits=base_out["pred_masks"].detach().clone(),
            image_features=image_features.detach().clone(),
            object_pointer=base_out["obj_ptr"].detach().clone(),
            state=state,
        )
        final_out, _, _ = predictor._run_single_frame_inference(
            inference_state=state_dict,
            output_dict=state_dict["output_dict"],
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=True,
            point_inputs=point_inputs,
            mask_inputs=prompt_logits,
            reverse=False,
            run_mem_encoder=False,
            mask_prompt_weight=1.0,
        )
        losses.append(dice_bce_loss(final_out["pred_masks"], target))

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    predictor = build_sam2_video_predictor(
        config_file="configs/sam2.1_hiera_t512_no_memory.yaml",
        ckpt_path=str(inference.SAM2_CHECKPOINT),
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.use_mask_input_as_output_without_sam=false"],
    )
    if predictor.num_maskmem != 0:
        raise RuntimeError("The experiment requires MedSAM2 native memory to be disabled.")
    freeze(predictor)
    adapter = ImplicitTemporalState(
        hidden_channels=args.hidden_channels,
        image_feature_channels=predictor.hidden_dim,
        object_pointer_dim=predictor.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for sequence_name in args.train_sequences:
            frame_dir = Path(inference.get_video_frame_dir(str(inference.DATA_ROOT), sequence_name))
            starts = sample_starts(
                len(inference.get_frame_names(str(frame_dir))), args.clip_length,
                args.prompt_stride, args.clips_per_sequence, rng,
            )
            for start in starts:
                epoch_losses.append(train_clip(
                    predictor, adapter, optimizer, sequence_name, start,
                    args.clip_length, args.prompt_stride, device,
                ))
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch}/{args.epochs}: loss={mean_loss:.5f}", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "implicit_temporal_state_v1",
            "model": adapter.state_dict(),
            "hidden_channels": adapter.hidden_channels,
            "image_feature_channels": adapter.image_feature_channels,
            "object_pointer_dim": adapter.object_pointer_dim,
            "maximum_prompt_logit": adapter.maximum_prompt_logit,
            "train_sequences": args.train_sequences,
            "prompt_stride": args.prompt_stride,
            "history": history,
        },
        args.checkpoint,
    )
    args.checkpoint.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()

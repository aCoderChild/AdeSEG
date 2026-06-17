"""
Generate frozen SAM2/MedSAM2 mask predictions for PolypGen-style sequences.

Pipeline:

    sequence/images + first-frame GT mask prompt
      -> frozen SAM2 video predictor
      -> predicted masks for all frames
      -> sequence/medsam2/<same filename as image>

Example:

    python src/sam2/infer_sam2.py \
      --data-root data/train/polypgen_train \
      --sam2-cfg configs/sam2.1_hiera_t512.yaml \
      --sam2-checkpoint checkpoints/MedSAM2_latest.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sequence_sort_key(path: Path):
    match = re.search(r"seq(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return path.name


def list_sequences(data_root: Path, sequence: str | None = None) -> list[Path]:
    if sequence is not None:
        seq_path = data_root / sequence
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence not found: {seq_path}")
        return [seq_path]

    sequences = [
        path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "images").is_dir() and (path / "masks").is_dir()
    ]
    return sorted(sequences, key=sequence_sort_key)


def list_frame_names(images_dir: Path) -> list[str]:
    frame_names = [
        path.name for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    ]
    frame_names.sort()
    if not frame_names:
        raise RuntimeError(f"No image frames found in {images_dir}")
    return frame_names


def load_binary_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Prompt mask not found: {mask_path}")
    mask = Image.open(mask_path).convert("L")
    mask_np = np.asarray(mask)
    return mask_np > 0


def save_binary_mask(mask_logits: torch.Tensor, output_path: Path, threshold: float):
    mask = (mask_logits > threshold).detach().cpu().numpy()
    mask = np.squeeze(mask).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def infer_sequence(
    predictor,
    seq_dir: Path,
    prompt_frame_index: int,
    obj_id: int,
    score_threshold: float,
    overwrite: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
):
    images_dir = seq_dir / "images"
    masks_dir = seq_dir / "masks"
    output_dir = seq_dir / "medsam2"

    frame_names = list_frame_names(images_dir)
    if prompt_frame_index < 0 or prompt_frame_index >= len(frame_names):
        raise IndexError(
            f"prompt_frame_index={prompt_frame_index} is outside sequence "
            f"{seq_dir.name} with {len(frame_names)} frames"
        )

    expected_outputs = [output_dir / frame_name for frame_name in frame_names]
    if not overwrite and all(path.exists() for path in expected_outputs):
        return "skipped"

    prompt_frame_name = frame_names[prompt_frame_index]
    prompt_mask = load_binary_mask(masks_dir / prompt_frame_name)

    inference_state = predictor.init_state(
        video_path=str(images_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=prompt_frame_index,
        obj_id=obj_id,
        mask=prompt_mask,
    )

    saved = 0
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        output_path = output_dir / frame_names[out_frame_idx]
        save_binary_mask(out_mask_logits[0], output_path, score_threshold)
        saved += 1

    return f"saved {saved}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen SAM2/MedSAM2 on PolypGen sequences using the first-frame "
            "GT mask as prompt, then save predicted masks under seq*/medsam2."
        )
    )
    parser.add_argument("--data-root", default="data/train/polypgen_train")
    parser.add_argument(
        "--sequence",
        help="Optional single sequence name, e.g. seq1. By default all sequences run.",
    )
    parser.add_argument("--sam2-cfg", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Disable SAM2 postprocessing overrides.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=not args.no_postprocessing,
    )
    predictor.eval()

    sequences = list_sequences(data_root, args.sequence)
    print(f"Found {len(sequences)} sequence(s) under {data_root}")

    with torch.inference_mode():
        for seq_dir in tqdm(sequences, desc="sequences"):
            result = infer_sequence(
                predictor=predictor,
                seq_dir=seq_dir,
                prompt_frame_index=args.prompt_frame_index,
                obj_id=args.obj_id,
                score_threshold=args.score_threshold,
                overwrite=args.overwrite,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
            tqdm.write(f"{seq_dir.name}: {result}")


if __name__ == "__main__":
    main()


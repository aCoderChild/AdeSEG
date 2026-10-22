"""Shared I/O for native memory-token spatial assignment audits.

Existing extraction/visualization helpers moved unchanged from retired runners.
No replacement model or attention algorithm is implemented here.
"""

import hashlib
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from datasets.io import load_label_mask
from datasets.video import stage_frames


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_output(path):
    path = Path(path).resolve()
    allowed = (
        path.is_relative_to(Path("/kaggle/working"))
        or path.is_relative_to(
            Path(
                "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs"
            )
        )
        or path.is_relative_to(Path("/content/drive/MyDrive/AdeSEG/outputs"))
    )
    if not allowed:
        raise ValueError(
            "Experiment outputs must be on Kaggle or the requested Google Drive"
        )
    return path


def parameter_digest(model):
    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        checksum.update(name.encode())
        checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def synchronize_storage(device):
    """Finish async accelerator-to-CPU offloads before Python deep-copies banks."""
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def save_feature_overlay(
    image_path,
    memberships,
    destination,
    names=("Foreground interior", "Background", "Boundary-related"),
    title="Predicted-region overlay",
    caption="Spatial token-region assignment; not exclusive vector semantics.",
):
    """Overlay spatial region memberships using nearest-neighbor token footprints."""
    palette = np.array([[0, 200, 100], [30, 140, 255], [255, 180, 0]], np.uint8)
    weights = memberships.detach().cpu()
    if (
        weights.ndim != 4
        or weights.shape[:2] != (1, 3)
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("Expected finite [1,3,H,W] region memberships")
    labels = weights[0].argmax(0).numpy()
    with Image.open(image_path) as source:
        original = source.convert("RGB")
        original.thumbnail((640, 480))
    colors = Image.fromarray(palette[labels]).resize(
        original.size, Image.Resampling.NEAREST
    )
    overlay = Image.blend(original, colors, 0.45)
    canvas = Image.new("RGB", (2 * original.width, original.height + 80), "white")
    canvas.paste(original, (0, 30))
    canvas.paste(overlay, (original.width, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((8, 5), "Original", fill="black", font=font)
    draw.text((original.width + 8, 5), title, fill="black", font=font)
    for index, name in enumerate(names):
        x, y = 8 + index * (canvas.width // 3), original.height + 38
        draw.rectangle((x, y, x + 12, y + 12), fill=tuple(palette[index]))
        draw.text((x + 18, y - 2), name, fill="black", font=font)
    draw.text((8, original.height + 60), caption, fill="black", font=font)
    canvas.save(destination)


def prompt_index(frames, protocol, label_id=1, label_ids=None):
    """Honor the frozen frame-zero protocol or its explicitly approved amendment."""
    candidates = (
        range(len(frames))
        if protocol.get("initialization") == "first_labeled_frame"
        else (0,)
    )
    for index in candidates:
        if (
            load_label_mask(frames[index].mask_path, label_ids or [label_id])
            == label_id
        ).any():
            return index
    raise ValueError("No eligible labeled prompt under the selected protocol")


@torch.no_grad()
def extract_memory_features(
    predictor,
    frames,
    videos,
    device,
    output,
    mask_output=None,
    include_prompt=False,
    progress=None,
):
    """Read native stored spatial memory; never hook, replace, or re-encode it.

    The first visible GT mask initializes normal SAM2 video propagation.
    Calls exclude the prompt by default. Reference audits may opt into copying
    conditioning-frame memory too; pre-prompt frames are never synthesized.
    Object pointers and positional encodings are not spatial grouping inputs.
    """
    captured, records = {}, []
    for name in dict.fromkeys(f.video_id for f in frames):
        sequence = videos[name]
        prompt = prompt_index(sequence, {"initialization": "first_labeled_frame"})
        wanted = {f.frame_index for f in frames if f.video_id == name}
        if min(wanted) < prompt or (min(wanted) == prompt and not include_prompt):
            raise ValueError("Memory probe samples must follow the prompt frame")
        with tempfile.TemporaryDirectory(
            prefix="memory_probe_frames_", dir=output
        ) as directory:
            stage_frames(sequence[: max(wanted) + 1], Path(directory))
            state = predictor.init_state(
                video_path=directory,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )
            try:
                initial = load_label_mask(sequence[prompt].mask_path, [1]) == 1
                predictor.add_new_mask(
                    state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial)
                )
                for index, _, logits in predictor.propagate_in_video(
                    state, start_frame_idx=prompt
                ):
                    synchronize_storage(device)
                    if index in wanted:
                        bucket = (
                            "cond_frame_outputs"
                            if index == prompt
                            else "non_cond_frame_outputs"
                        )
                        item = state["output_dict"][bucket][index]
                        native = item["maskmem_features"]
                        if (
                            native is None
                            or native.ndim != 4
                            or native.shape[0] != 1
                            or not torch.isfinite(native).all()
                        ):
                            raise ValueError(
                                "Expected finite single-object native spatial memory"
                            )
                        captured[(name, index)] = (
                            native.detach()
                            .to(device="cpu", dtype=torch.float32)
                            .clone()
                        )
                        if mask_output is not None:
                            mask_output[(name, index)] = (
                                logits[0, 0].detach().cpu() > 0
                            ).clone()
                        records.append(
                            {
                                "sequence": name,
                                "frame": index,
                                "prompt_frame": prompt,
                                "shape": list(native.shape),
                                "stored_dtype": str(native.dtype),
                                "prediction_sha256": hashlib.sha256(
                                    logits.detach()
                                    .cpu()
                                    .float()
                                    .contiguous()
                                    .numpy()
                                    .tobytes()
                                ).hexdigest(),
                            }
                        )
                        if progress is not None:
                            progress(records[-1])
                        print(
                            f"[memory-probe] captured {name} frame={index} shape={list(native.shape)}",
                            flush=True,
                        )
                    if index >= max(wanted):
                        break
            finally:
                predictor.reset_state(state)
    if len(captured) != len(frames):
        raise ValueError("Missing or duplicate memory feature samples")
    return torch.cat([captured[(f.video_id, f.frame_index)] for f in frames]), records

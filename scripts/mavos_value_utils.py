"""Shared, non-model I/O helpers for the MAVOS-derived memory-value study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from datasets.io import load_label_mask
from datasets.video import stage_frames
from scripts.memory_token_io import prompt_index
from scripts.run_author_baseline import source_identity


def activate_author_repo():
    repo, revision = source_identity("sam2")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo, revision


def prepare(model, frames, directory):
    stage_frames(frames, Path(directory))
    state = model.init_state(
        video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=False
    )
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    mask = torch.from_numpy(
        (load_label_mask(frames[prompt].mask_path, [1]) == 1).copy()
    )
    model.add_new_mask(state, frame_idx=prompt, obj_id=1, mask=mask)
    return state, prompt


def target_for(frame):
    gt = load_label_mask(frame.mask_path, [1]) == 1
    target = torch.from_numpy(gt.astype("float32"))[None, None]
    return F.interpolate(target, size=(256, 256), mode="nearest"), gt


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

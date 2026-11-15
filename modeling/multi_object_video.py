"""Use SAM2's independent object banks with a shared regional adapter."""

import numpy as np
import torch

from datasets.instance_masks import load_instance_mask
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from modeling.cfbi_training import capture_training_read, clone_tensors
from modeling.regional_low_rank import NativeRegionalLowRank


def propagate_object_prompts(model, state, prompts):
    """Insert first masks online, then delegate each interval to native SAM2."""
    if not prompts:
        raise ValueError("At least one instance prompt is required")
    seen = set()
    times = sorted(prompts)
    for frame in times:
        if not isinstance(frame, int) or not 0 <= frame < state["num_frames"]:
            raise ValueError("Prompt frame is outside the video")
        if not prompts[frame]:
            raise ValueError("Prompt frame has no objects")
        for object_id, mask in prompts[frame].items():
            if (not isinstance(object_id, int) or isinstance(object_id, bool)
                    or object_id <= 0 or object_id in seen):
                raise ValueError("Each positive instance ID must receive one first prompt")
            if torch.as_tensor(mask).ndim != 2 or not torch.as_tensor(mask).bool().any():
                raise ValueError("Instance prompt must be a nonempty 2-D mask")
            seen.add(object_id)
    for index, frame in enumerate(times):
        for object_id, mask in prompts[frame].items():
            model.add_new_mask(state, frame_idx=frame, obj_id=object_id, mask=mask)
        end = times[index + 1] - 1 if index + 1 < len(times) else state["num_frames"] - 1
        yield from model.propagate_in_video(
            state, start_frame_idx=frame, max_frame_num_to_track=end - frame)


def capture_object_clips(model, state, frames, prompts, label_ids):
    """Capture separate autoregressive training episodes from one native video."""
    if len(frames) != state["num_frames"]:
        raise ValueError("Manifest and native video lengths differ")
    clips = {object_id: [] for objects in prompts.values() for object_id in objects}
    with torch.inference_mode():
        for frame, ids, _ in propagate_object_prompts(model, state, prompts):
            labels = load_instance_mask(frames[frame].mask_path)
            if set(np.unique(labels).tolist()) - {0, *label_ids}:
                raise ValueError("Training mask contains undeclared instance IDs")
            for object_id in ids:
                obj_idx = state["obj_id_to_idx"][object_id]
                bank = state["output_dict_per_obj"][obj_idx]
                if frame in bank["cond_frame_outputs"]:
                    continue
                stored = bank["non_cond_frame_outputs"][frame]
                sample = capture_training_read(model, state, frame, object_id=object_id)
                with torch.inference_mode(False):
                    sample.update(clone_tensors({
                        "object_id": object_id, "native_logits": stored["pred_masks"].float(),
                        "native_object_score": stored["object_score_logits"],
                        "target": torch.from_numpy((labels == object_id).copy())[None, None].float(),
                    }))
                clips[object_id].append(sample)
    return clips


def object_clip_loss(model, heads, clips, mode, *, check_native=False):
    """Average per-instance native losses; each instance retains its own bank."""
    if not clips or any(len(samples) < 2 for samples in clips.values()):
        raise ValueError("Each training instance needs at least two post-prompt reads")
    losses, reads = [], []
    for object_id, samples in clips.items():
        if any(sample["object_id"] != object_id for sample in samples):
            raise ValueError("Training samples have inconsistent instance IDs")
        with NativeRegionalLowRank(model, heads, mode) as observer:
            losses.append(native_qk_clip_loss(model, samples, check_native=check_native))
        reads.extend({**call, "object_id": object_id} for call in observer.calls)
    return torch.stack(losses).mean(), reads

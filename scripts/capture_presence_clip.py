"""Capture native training reads, targets, and per-frame SAM2 presence scores."""
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from modeling.cfbi_training import capture_training_read
from scripts.run_cfbi_region_pilot import prepare


def capture_presence_clip(model, frames, out):
    samples = []
    with tempfile.TemporaryDirectory(prefix='presence_frames_', dir=out) as temp:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError('Sampled clip must begin at its prompt frame')
            for frame, ids, _ in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError('Unexpected object IDs')
                if frame == prompt:
                    continue
                sample = capture_training_read(model, state, frame)
                stored = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                with torch.inference_mode(False):
                    sample['native_logits'] = stored['pred_masks'].detach().float().clone()
                    sample['native_object_score'] = stored['object_score_logits'].detach().float().clone()
                    target = load_label_mask(frames[frame].mask_path, [1]) == 1
                    sample['target'] = torch.from_numpy(target.copy())[None, None].float()
                samples.append(sample)
            model.reset_state(state)
    if len(samples) != len(frames) - 1:
        raise RuntimeError('Missing sampled reads')
    return samples

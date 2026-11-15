#!/usr/bin/env python3
"""Two-frame synthetic, real-predictor routing smoke; no experiment outputs.

Example:
  .venv/bin/python scripts/verify_lean_contrast_routing_smoke.py --checkpoint "/path/to/sam2.1_hiera_base_plus.pt"

Uses explicitly test-only identity and forced-logit refiners, not trained or
authorized models for evaluation. This cannot establish learned-refiner
correctness, long-video parity, dataset performance, or checkpoint provenance.
"""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from scripts.run_author_baseline import build_author_predictor


class _SmokeIdentity(torch.nn.Module):
    # The routing adapter recognizes this only within the synthetic smoke test.
    # This flag does not authorize or manufacture a research checkpoint.
    _trained_checkpoint_loaded = True

    def forward(self, early, deep, low):
        if early.shape != (1, 32, 256, 256) or deep.shape != (1, 256, 64, 64) or low.shape != (1, 1, 256, 256):
            raise RuntimeError('Unexpected feature geometry')
        return low.clone()


class _SmokeForcedLogits(_SmokeIdentity):
    def forward(self, early, deep, low):
        super().forward(early, deep, low)
        # Large enough to cross the native sigmoid even if the original mask is
        # suppressed by SAM2's no-object output. This is only a routing stimulus.
        return torch.full_like(low, 2.0)


def _capture(model, frames, arm, identity):
    state = model.init_state(video_path=str(frames), offload_video_to_cpu=True,
                             offload_state_to_cpu=False)

    def propagate():
        _, ids, mask = model.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            box=np.array([30, 30, 95, 95], dtype=np.float32))
        if ids != [1]:
            raise RuntimeError('Unexpected object identity')
        outputs = [(0, mask.cpu().clone())]
        for frame, ids, masks in model.propagate_in_video(state):
            if ids != [1]:
                raise RuntimeError('Unexpected propagated object identity')
            outputs.append((frame, masks.cpu().clone()))
        entry = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][1]
        keys = ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits')
        return outputs, {key: entry[key].cpu().clone() for key in keys}

    with torch.inference_mode():
        if arm is None:
            return propagate()
        with route_lean_contrast_refinement(model, identity, arm, test_only=True):
            return propagate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error('checkpoint file does not exist')
    torch.set_num_threads(2)
    model = build_author_predictor(
        SimpleNamespace(method='lean', checkpoint=args.checkpoint.resolve(), device='cpu'),
        ROOT / 'external/Lean-SAM2')
    identity = _SmokeIdentity().eval()
    with TemporaryDirectory(prefix='lean_refiner_smoke_') as temporary:
        frames = Path(temporary)
        y, x = np.indices((128, 128))
        for i in range(2):
            image = np.stack(((x * 2 + i) % 256, (y * 2) % 256, (x + y) % 256), axis=-1).astype(np.uint8)
            Image.fromarray(image).save(frames / f'{i:05d}.jpg')
        reference, reference_state = _capture(model, frames, None, identity)
        for arm in ('A', 'B', 'C', 'D'):
            output, state = _capture(model, frames, arm, identity)
            if len(output) != len(reference):
                raise RuntimeError(f'{arm}: output count changed')
            for (index, expected), (actual_index, actual) in zip(reference, output):
                if index != actual_index or not torch.equal(expected, actual):
                    raise RuntimeError(f'{arm}: video output changed at frame {index}')
            for key, expected in reference_state.items():
                if not torch.equal(expected, state[key]):
                    raise RuntimeError(f'{arm}: stored {key} changed')
            print(f'PASS {arm}: output and stored tensors bit-identical')
        forced_logits = _SmokeForcedLogits().eval()
        for arm in ('B', 'C', 'D'):
            output, state = _capture(model, frames, arm, forced_logits)
            if [index for index, _ in output] != [index for index, _ in reference]:
                raise RuntimeError(f'{arm}: output frames changed')
            for (index, expected), (_, actual) in zip(reference, output):
                should_change = index == 1 and arm in ('B', 'D')
                if torch.equal(expected, actual) == should_change:
                    raise RuntimeError(f'{arm}: wrong output route on frame {index}')
            for key in ('pred_masks', 'obj_ptr', 'object_score_logits'):
                if not torch.equal(reference_state[key], state[key]):
                    raise RuntimeError(f'{arm}: native stored {key} changed')
            memory_changed = not torch.equal(reference_state['maskmem_features'], state['maskmem_features'])
            if memory_changed != (arm in ('C', 'D')):
                raise RuntimeError(f'{arm}: wrong memory route')
            print(f'PASS {arm}: forced logits follow output and memory routes')


if __name__ == '__main__':
    main()

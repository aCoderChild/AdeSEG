"""Exploratory C2-v2: preserve interior/background, prune older boundary cells.

This project policy is motivated by one-step exclusions, not supplied by an
author repository. Gathering, RoPE, budgets and matched randomness are reused.
"""

import numpy as np
import torch

from modeling.matched_region_memory import MatchedRegionMemory
from modeling.region_definitions import mixed_cell_regions


def contribution_cells(labels):
    labels = torch.as_tensor(labels)
    if labels.shape != (64, 64) or not torch.isin(labels, labels.new_tensor([0, 1, 2])).all():
        raise ValueError("Expected 64x64 background/interior/boundary labels")
    return (labels != 2).flatten()


class ContributionRegionMemory(MatchedRegionMemory):
    """Change only older-frame selection after the actual native memory write."""

    def __init__(self, model, prompt, sequence, *, reference=None, seed=None):
        super().__init__(model, prompt, sequence, reference=reference, seed=seed)
        # Reuse native temporal assembly: prompt plus at most six prior frames.
        # The current frame is the query, never a memory source for its own read.
        self.mode = "region_native"
        self.max_frames = 7

    def _before(self, layer, module, args, kwargs):
        super()._before(layer, module, args, kwargs)
        frames = self.layers[-1]["frames"]
        if frames != [self.prompt] + list(range(max(self.prompt+1, self.frame-6), self.frame)):
            raise RuntimeError("C2-v2 read exceeded native recent-six memory window")

    def capture(self, state, index):
        record = super().capture(state, index)
        if self.reference is None:
            outputs = state["output_dict_per_obj"][0]
            stored = outputs["cond_frame_outputs"].get(index, outputs["non_cond_frame_outputs"].get(index))
            labels = mixed_cell_regions(stored["pred_masks"][0, 0])["labels"]
            self.bank[index]["keep"] = contribution_cells(labels)
        keep = self.bank[index]["keep"].cpu().numpy()
        record.update(policy="contribution_v2", memory_window="prompt_plus_previous_six",
                      retained_tokens=int(keep.sum()),
                      keep_bits_hex=np.packbits(keep, bitorder="big").tobytes().hex())
        return record

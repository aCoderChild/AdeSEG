"""Cutie regional memory summaries with a zero-initialized output adapter.

This variant keeps the native SAM2 memory-attention result unchanged at
initialization, but avoids placing a single scalar gate in front of the whole
regional branch.  The pretrained Cutie summarizer and CrossAttention modules
are reused from the pinned author repository via ``CutieRegionMessageHeads``.
"""

import torch
from torch import nn

from modeling.cutie_region_prototypes import CutieRegionMessageHeads


class CutieRegionProjectionHead(CutieRegionMessageHeads):
    """Project a Cutie regional message into the native SAM2 residual stream."""

    def __init__(self, checkpoint=None, layer_index=0):
        if checkpoint is None:
            super().__init__(layer_index=layer_index)
        else:
            super().__init__(checkpoint=checkpoint, layer_index=layer_index)
        # The prior scalar residual learned values near zero, making the
        # complete branch almost a no-op. Freeze that unused parameter and
        # place the zero initialization at the branch's output projection,
        # following the zero-connection pattern used by pretrained adapters.
        self.scale_logit.requires_grad_(False)
        self.output_projection = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def read(self, query, messages, positions, native_output):
        query = query.to(dtype=self.region_position.dtype)
        query_position = torch.zeros_like(query)
        branch, _ = self.reader(query, messages, query_position, positions)
        update = self.output_projection(branch.to(self.output_projection.weight.dtype))
        return native_output + update.to(native_output.dtype)


class CutieRegionProjectionHeads(nn.Module):
    """Four layer-specific region-message adapters for SAM2 memory attention."""

    def __init__(self, layers=4, checkpoint=None):
        super().__init__()
        self.layers = nn.ModuleList(
            CutieRegionProjectionHead(checkpoint, layer_index=index)
            for index in range(layers)
        )


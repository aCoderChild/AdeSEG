"""Recover and validate spatial memory positions from Lean's position inputs.

This is a diagnostic for the existing fixed-size, single-object FP32 run. It
does not change attention or make guesses for vectors outside the known grids.
"""
import numpy as np
import torch


class MemoryPositionLookup:
    def __init__(self, spatial_grid, temporal_offsets):
        if spatial_grid.ndim != 2 or temporal_offsets.ndim != 2:
            raise ValueError('Expected [locations, channels] and [offsets, channels]')
        if spatial_grid.shape[1] != temporal_offsets.shape[1]:
            raise ValueError('Position channel count differs')
        if spatial_grid.device.type != 'cpu' or spatial_grid.dtype != torch.float32:
            raise ValueError('Diagnostic requires CPU FP32 positional encodings')
        self.channels = spatial_grid.shape[1]
        self.lookup = {}
        self.ambiguous = set()
        offsets = torch.cat((temporal_offsets.cpu().float(), torch.zeros(1, self.channels)), dim=0)
        for offset in offsets:
            combined = (spatial_grid + offset).contiguous().numpy()
            for index, row in enumerate(combined):
                key = row.tobytes()
                previous = self.lookup.setdefault(key, index)
                if previous != index:
                    self.ambiguous.add(key)

    def decode(self, memory_pos, pointer_count):
        if memory_pos.ndim != 3 or memory_pos.shape[1] != 1:
            raise ValueError('Diagnostic requires sequence-first, single-object positions')
        if memory_pos.device.type != 'cpu' or memory_pos.dtype != torch.float32:
            raise ValueError('Diagnostic requires CPU FP32 positional encodings')
        if not 0 <= pointer_count <= memory_pos.shape[0] or memory_pos.shape[2] != self.channels:
            raise ValueError('Invalid pointer count or position channels')
        count = memory_pos.shape[0] - pointer_count
        rows = memory_pos[:count, 0, :].contiguous().numpy()
        result = np.empty(count, dtype=np.int64)
        for i, row in enumerate(rows):
            key = row.tobytes()
            if key in self.ambiguous or key not in self.lookup:
                raise ValueError(f'Position {i} cannot be mapped unambiguously to a spatial token')
            result[i] = self.lookup[key]
        return torch.from_numpy(result)

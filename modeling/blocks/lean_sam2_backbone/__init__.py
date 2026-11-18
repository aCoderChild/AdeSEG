"""Lean-SAM2 backbone corrections and runtime controls.

The actual pretrained Hiera-B+ model lives in the pinned
external/Lean-SAM2 submodule; importing this block does not load weights.
"""
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps

__all__ = ["indexed_spatial_rope", "indexed_spatial_rope_mps"]

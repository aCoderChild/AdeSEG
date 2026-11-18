"""Untrained pooled memory-anchor aggregation (separate Lean-SAM2 ablation)."""
from modeling.lean_multiscale_anchor import install_multiscale_anchor, multiscale_anchor_score

__all__ = ["install_multiscale_anchor", "multiscale_anchor_score"]

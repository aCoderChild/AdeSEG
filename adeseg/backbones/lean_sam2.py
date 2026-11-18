"""Pinned author Lean-SAM2 Hiera-B+ implementation.

The author source lives in external/Lean-SAM2. This module is an explicit
entrypoint, not a second implementation or a MedSAM2 checkpoint alias.
"""
from pathlib import Path

from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position

LEAN_AUTHOR_REVISION = "8d1cf2618519d291905d22fe859b8302476d7f50"


def build_lean_sam2_predictor(args, repository=None):
    """Call the retained author-runner builder with its original ABI.

    This intentionally does not bypass the runner's revision/checkpoint checks.
    For a source-identical historical experiment, use the archived runner.
    """
    if getattr(args, "method", None) != "lean":
        raise ValueError("Expected --method lean; do not load MedSAM2 weights here")
    from scripts.run_author_baseline import build_author_predictor
    root = Path(__file__).resolve().parents[2]
    repo = Path(repository) if repository is not None else root / "external/Lean-SAM2"
    return build_author_predictor(args, repo)


__all__ = [
    "LEAN_AUTHOR_REVISION",
    "build_lean_sam2_predictor",
    "indexed_spatial_rope",
    "indexed_spatial_rope_mps",
    "blocking_mps_pointer_position",
]

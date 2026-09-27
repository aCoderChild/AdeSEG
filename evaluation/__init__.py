"""Dataset-independent quantitative evaluation helpers."""

from .ratio import evaluate_ratios
from .segmentation import evaluate_regions

__all__ = ["evaluate_ratios", "evaluate_regions"]

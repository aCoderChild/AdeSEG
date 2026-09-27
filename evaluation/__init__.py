"""Dataset-independent quantitative evaluation helpers."""

from .measurement import evaluate_two_region_measurement
from .ratio import evaluate_ratios
from .segmentation import evaluate_regions

__all__ = ["evaluate_ratios", "evaluate_regions", "evaluate_two_region_measurement"]

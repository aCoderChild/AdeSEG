"""SAM2/MedSAM2 Dice, IoU and regional segmentation scoring."""
from evaluation.segmentation import calculate_scores, evaluate_regions

__all__ = ["calculate_scores", "evaluate_regions"]

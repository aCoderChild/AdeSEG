"""FGPI: a fixed-budget YOLO-box plus early-feature prompt constructor."""
from __future__ import annotations

from dataclasses import dataclass

from adeseg.prompts.feature_guided import (
    apply_fgpi_prompt, feature_guided_prompt,
)
from adeseg.prompts.target_anchored import target_anchored_points


@dataclass(frozen=True)
class FGPIConfig:
    positive_points: int = 1
    negative_points: int = 1
    max_pseudo_anchors: int = 4

    def __post_init__(self):
        if (type(self.positive_points) is not int
                or type(self.negative_points) is not int
                or type(self.max_pseudo_anchors) is not int
                or not 1 <= self.positive_points <= 8
                or not 1 <= self.negative_points <= 8
                or not 1 <= self.max_pseudo_anchors <= 16):
            raise ValueError("FGPI fixed point and anchor budgets are invalid")


class FGPI:
    """FGPI does not run YOLO: it consumes a YOLO box and frozen FPN features."""

    def __init__(self, config: FGPIConfig | None = None):
        self.config = config if config is not None else FGPIConfig()
        if not isinstance(self.config, FGPIConfig):
            raise TypeError("FGPI requires FGPIConfig")

    def construct(self, features, box_xyxy, image_hw, *, detector_confidence=1.0):
        return feature_guided_prompt(
            features, box_xyxy, image_hw,
            mode="fgpi", detector_confidence=detector_confidence,
            positives=self.config.positive_points,
            negatives=self.config.negative_points,
            max_anchors=self.config.max_pseudo_anchors,
        )

    @staticmethod
    def initialize(predictor, state, frame_index, object_id, prompt):
        return apply_fgpi_prompt(predictor, state, frame_index, object_id, prompt)


__all__ = ["FGPI", "FGPIConfig", "target_anchored_points", "feature_guided_prompt"]

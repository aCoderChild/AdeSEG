"""Selection of usable per-frame measurements."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def select_valid_frames(frame_predictions: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Keep records marked valid with a finite ratio."""
    return [
        prediction
        for prediction in frame_predictions
        if prediction.get("valid_frame") and np.isfinite(prediction.get("ratio", float("nan")))
    ]

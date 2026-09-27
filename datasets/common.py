"""Small dataset-independent frame sample definition."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FrameSample:
    """One frame and its named binary masks.

    ``masks`` can contain one region (for PolypGen) or two regions (for REFUGE2
    and the future adenoid dataset).
    """

    frame_idx: int
    image_path: Path
    masks: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def sequence_id(self) -> str | None:
        value = self.metadata.get("sequence_id", self.metadata.get("sequence"))
        return None if value is None else str(value)

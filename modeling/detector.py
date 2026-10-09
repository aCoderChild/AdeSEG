"""Polyp detector observations for the memory update: the top YOLOv8n box per frame.

The detector (``checkpoints/polypgen_yolov8n.pt``) was trained on PolypGen single frames
(data_C1..C6 minus the frames that duplicate test frames) and negativeOnly backgrounds, never
on ``sequenceData/positive`` frames, so it is out-of-sample on every sequence used here.
"""

from __future__ import annotations

from pathlib import Path

import torch


class PolypDetector:
    """Top box (normalised x1, y1, x2, y2) and confidence per image, cached by path."""

    def __init__(self, weights: str | Path, device: str = "cpu", min_confidence: float = 0.05):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.min_confidence = min_confidence
        self.cache: dict[str, tuple[list[float], float]] = {}

    def __call__(self, paths) -> list[tuple[list[float], float]]:
        paths = [str(path) for path in paths]
        missing = [path for path in dict.fromkeys(paths) if path not in self.cache]
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            results = self.model.predict(batch, conf=self.min_confidence, device=self.device, verbose=False)
            for path, result in zip(batch, results):
                if len(result.boxes):
                    top = int(result.boxes.conf.argmax())
                    self.cache[path] = (result.boxes.xyxyn[top].tolist(), float(result.boxes.conf[top]))
                else:
                    self.cache[path] = ([0.0, 0.0, 0.0, 0.0], 0.0)
        return [self.cache[path] for path in paths]

    def frame_detections(self, paths, device) -> dict[int, dict[str, torch.Tensor]]:
        """{frame index: {"box": [1, 4], "confidence": [1, 1]}} for the memory update."""
        return {
            index: {"box": torch.tensor([box], device=device), "confidence": torch.tensor([[confidence]], device=device)}
            for index, (box, confidence) in enumerate(self(paths))
        }

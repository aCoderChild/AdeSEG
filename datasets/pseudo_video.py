"""Pseudo-videos from PolypGen single frames with spliced polyp-free stretches.

1. labeled image from data_C1 - data_C6
2. Simulate a camera - random drift (Ornstein-Uhlenbeck) + zoom + rotate
3. add realism: brightness and colour jitter + occasional motion blus
4. insert absence: 2 - 8 frames where the polyp is gone - simulate reappearance
5. Returns: frames + masks + flags which frames have polyp
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from adenoid.io import load_binary_mask, resolve_mask_path

IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _numeric_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


@dataclass(frozen=True)
class LabeledFrame:
    image_path: Path
    mask_path: Path
    center: str


def list_single_frames(
    polypgen_root: Path, centers=(1, 2, 3, 4, 5, 6), excluded: set[str] | None = None
) -> list[LabeledFrame]:
    """Labeled single frames with a non-empty mask, minus excluded relative paths."""
    excluded = excluded or set()
    frames = []
    for center in centers:
        image_dir = polypgen_root / f"data_C{center}" / f"images_C{center}"
        mask_dir = polypgen_root / f"data_C{center}" / f"masks_C{center}"
        for image_path in sorted(image_dir.glob("*.jpg"), key=_numeric_key):
            if str(image_path.relative_to(polypgen_root)) in excluded:
                continue
            mask_path = resolve_mask_path(mask_dir, image_path.stem)
            if mask_path is None or not load_binary_mask(mask_path).any():
                continue
            frames.append(LabeledFrame(image_path, mask_path, f"C{center}"))
    if not frames:
        raise FileNotFoundError(f"No labeled single frames under {polypgen_root}")
    return frames


def list_negative_sequences(polypgen_root: Path, excluded: set[str] | None = None) -> list[list[Path]]:
    excluded = excluded or set()
    root = polypgen_root / "sequenceData" / "negativeOnly"
    sequences = []
    for directory in sorted(root.iterdir(), key=_numeric_key):
        if not directory.is_dir() or directory.name in excluded:
            continue
        frames = sorted(directory.rglob("*.jpg"), key=_numeric_key)
        if len(frames) >= 2:
            sequences.append(frames)
    if not sequences:
        raise FileNotFoundError(f"No negative sequences under {root}")
    return sequences


def load_excluded_paths(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


@dataclass(frozen=True)
class ClipDifficulty:
    """Augmentation strength. Probabilities of 0 draw no random numbers, so the
    ``easy`` preset reproduces the original clip stream exactly."""

    max_frame_gap: int = 40
    motion_spread: tuple[float, float, float, float] = (0.12, 0.12, 0.2, 8.0)
    gain_range: tuple[float, float] = (0.85, 1.15)
    offset_range: float = 0.06
    tint_range: tuple[float, float] = (0.92, 1.08)
    motion_blur_probability: float = 0.2
    motion_blur_max: int = 12
    defocus_probability: float = 0.0
    noise_probability: float = 0.0
    jpeg_probability: float = 0.0
    specular_probability: float = 0.0
    vignette_probability: float = 0.0
    occlusion_probability: float = 0.0
    out_of_view_probability: float = 0.3
    near_out_of_view: bool = False  # look away within the same scene instead of sliding off the image
    inpaint_probability: float = 0.0  # absence = same scene with the polyp inpainted away
    min_log_zoom: float | None = None  # limit zoom-out so the view never shrinks into black


EASY = ClipDifficulty()
HARD = replace(
    EASY,
    max_frame_gap=80,
    motion_spread=(0.18, 0.18, 0.35, 15.0),
    gain_range=(0.7, 1.3),
    offset_range=0.1,
    tint_range=(0.85, 1.15),
    motion_blur_probability=0.35,
    motion_blur_max=25,
    defocus_probability=0.2,
    noise_probability=0.5,
    jpeg_probability=0.3,
    specular_probability=0.5,
    vignette_probability=0.4,
    occlusion_probability=0.4,
    out_of_view_probability=0.5,
    near_out_of_view=True,
    inpaint_probability=0.4,
    min_log_zoom=-0.05,
)
DIFFICULTIES = {"easy": EASY, "hard": HARD}


class PseudoVideoGenerator:
    def __init__(
        self,
        frames: list[LabeledFrame],
        negative_sequences: list[list[Path]],
        image_size: int = 512,
        clip_length: int = 16,
        max_frame_gap: int | None = None,
        absence_probability: float = 0.6,
        min_absence: int = 2,
        max_absence: int = 12,
        out_of_view_probability: float | None = None,
        motion_time_constant: float = 60.0,
        base_zoom: float = 1.1,
        difficulty: ClipDifficulty = EASY,
    ):
        if clip_length < 3:
            raise ValueError("clip_length must be at least 3.")
        self.frames = frames
        self.negative_sequences = negative_sequences
        self.image_size = image_size
        self.clip_length = clip_length
        self.max_frame_gap = difficulty.max_frame_gap if max_frame_gap is None else max_frame_gap
        self.absence_probability = absence_probability
        self.min_absence = min_absence
        self.max_absence = max_absence
        self.out_of_view_probability = (
            difficulty.out_of_view_probability if out_of_view_probability is None else out_of_view_probability
        )
        self.motion_time_constant = motion_time_constant
        self.base_zoom = base_zoom
        self.difficulty = difficulty

    def _load(self, path: Path, interpolation=cv2.INTER_AREA) -> np.ndarray:
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=interpolation)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = load_binary_mask(path)
        return cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

    def _camera_path(self, times: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        """Mean-reverting (Ornstein-Uhlenbeck) camera wander sampled at raw-frame times.

        Stationary spread and time constant are set so a gap of 3 raw frames moves
        the view by about 3-4% of the image and a gap of 80 by about 12%, matching
        the annotated-neighbour centroid shifts measured on PolypGen sequences.
        """
        spread = np.array(self.difficulty.motion_spread)  # x, y (fraction of size), log-scale, degrees
        state = np.zeros(4)
        path = [state.copy()]
        for gap in np.diff(times):
            decay = np.exp(-gap / self.motion_time_constant)
            state = state * decay + rng.normal(0.0, 1.0, 4) * spread * np.sqrt(1.0 - decay ** 2)
            path.append(state.copy())
        return path

    def _affine(self, pose: np.ndarray, offset=(0.0, 0.0)):
        size = self.image_size
        shift = (pose[:2] + np.asarray(offset)) * size
        log_zoom = pose[2] if self.difficulty.min_log_zoom is None else max(pose[2], self.difficulty.min_log_zoom)
        scale = self.base_zoom * float(np.exp(log_zoom))
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), float(pose[3]), scale)
        matrix[:, 2] += shift
        return matrix

    def _look_away(self, pose: np.ndarray, polyp_direction: np.ndarray, zoom: float):
        """Zoom onto tissue on the far side from the polyp, so the view stays filled
        with the same scene while the polyp falls outside it."""
        size = self.image_size
        center = np.array([size / 2, size / 2])
        target = center - polyp_direction * 0.25 * size
        matrix = cv2.getRotationMatrix2D(tuple(center), float(pose[3]), self.base_zoom * zoom)
        mapped = matrix[:, :2] @ target + matrix[:, 2]
        matrix[:, 2] += center - mapped + pose[:2] * size * 0.5
        return matrix

    def _photometric(self, image: np.ndarray, rng: np.random.Generator, clip_effects: dict):
        level = self.difficulty
        image = image.astype(np.float32) / 255.0
        image = image * rng.uniform(*level.gain_range) + rng.uniform(-level.offset_range, level.offset_range)
        image = image * clip_effects["tint"]
        if rng.random() < level.motion_blur_probability:
            length = int(rng.integers(3, level.motion_blur_max))
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1.0 / length
            rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
            kernel = cv2.warpAffine(kernel, rotation, (length, length))
            image = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))
        if level.defocus_probability and rng.random() < level.defocus_probability:
            image = cv2.GaussianBlur(image, (0, 0), rng.uniform(1.0, 4.0))
        if clip_effects.get("vignette") is not None:
            image = image * clip_effects["vignette"][..., None] ** rng.uniform(0.8, 1.25)
        if clip_effects.get("specular"):
            image = self._add_specular(image, rng)
        if level.noise_probability and rng.random() < level.noise_probability:
            image = image + rng.normal(0.0, rng.uniform(0.005, 0.03), image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        if level.jpeg_probability and rng.random() < level.jpeg_probability:
            quality = int(rng.integers(25, 71))
            encoded = cv2.imencode(".jpg", (image * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        return image

    def _add_specular(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Small saturated highlights that move every frame, as on wet mucosa."""
        size = self.image_size
        highlight = np.zeros((size, size), np.float32)
        for _ in range(int(rng.integers(3, 13))):
            center = tuple(int(v) for v in rng.integers(0, size, 2))
            axes = tuple(int(v) for v in rng.integers(2, max(3, size // 60), 2))
            cv2.ellipse(highlight, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        highlight = cv2.GaussianBlur(highlight, (0, 0), 1.5) * (image.mean(axis=-1) > 0.08)
        highlight = highlight[..., None]
        return image * (1.0 - highlight) + highlight

    def _vignette(self, rng: np.random.Generator) -> np.ndarray:
        size = self.image_size
        grid = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        center = rng.uniform(-0.3, 0.3, 2)
        radius = (grid[None, :] - center[0]) ** 2 + (grid[:, None] - center[1]) ** 2
        return np.clip(1.0 - rng.uniform(0.2, 0.6) * radius, 0.15, 1.0)

    def _occluder(self, mask: np.ndarray, image: np.ndarray, rng: np.random.Generator):
        """Soft ellipse of displaced, blurred tissue over part of the polyp (fold or fluid).

        Returns the occluded image and the visible-polyp mask; hidden pixels are
        removed from the label, as an annotator would not mark them.
        """
        points = np.argwhere(mask)
        if points.size == 0:
            return image, mask
        (y0, x0), (y1, x1) = points.min(axis=0), points.max(axis=0)
        center = (
            int(rng.uniform(x0, x1 + 1)),
            int(rng.uniform(y0, y1 + 1)),
        )
        axes = (
            max(4, int((x1 - x0 + 1) * rng.uniform(0.3, 0.7))),
            max(4, int((y1 - y0 + 1) * rng.uniform(0.3, 0.7))),
        )
        cover = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(cover, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        shift = rng.integers(self.image_size // 6, self.image_size // 3, 2) * rng.choice([-1, 1], 2)
        texture = cv2.GaussianBlur(np.roll(image, tuple(int(v) for v in shift), axis=(0, 1)), (0, 0), 3.0)
        alpha = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), 2.0)[..., None]
        occluded = (image * (1.0 - alpha) + texture * alpha).astype(image.dtype)
        return occluded, (mask * (1 - cover)).astype(mask.dtype)

    def _absence_segment(self, rng: np.random.Generator):
        if rng.random() >= self.absence_probability:
            return None
        length = int(rng.integers(self.min_absence, self.max_absence + 1))
        length = min(length, self.clip_length - 2)
        start = int(rng.integers(1, self.clip_length - length))
        if self.difficulty.inpaint_probability and rng.random() < self.difficulty.inpaint_probability:
            return start, start + length, "inpaint"
        mode = "out_of_view" if rng.random() < self.out_of_view_probability else "negative"
        return start, start + length, mode

    def sample(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        frame = self.frames[int(rng.integers(len(self.frames)))]
        image = self._load(frame.image_path)
        mask = self._load_mask(frame.mask_path)
        level = self.difficulty
        clip_effects = {"tint": rng.uniform(*level.tint_range, size=3).astype(np.float32)}
        gaps = rng.integers(1, self.max_frame_gap + 1, size=self.clip_length)
        gaps[0] = 0
        if rng.random() < 0.5:
            gaps[1:] = gaps[1]
        times = np.cumsum(gaps).astype(np.float32)
        path = self._camera_path(times, rng)
        absence = self._absence_segment(rng)
        negative_frames = None
        if absence is not None and absence[2] == "negative":
            sequence = self.negative_sequences[int(rng.integers(len(self.negative_sequences)))]
            step = max(1, int(round(float(gaps[1:].mean()) / 4)))
            needed = (absence[1] - absence[0] - 1) * step + 1
            start = int(rng.integers(0, max(1, len(sequence) - needed)))
            negative_frames = [sequence[min(start + i * step, len(sequence) - 1)]
                               for i in range(absence[1] - absence[0])]
            negative_path = self._camera_path(times[absence[0] : absence[1]], rng)
        if level.vignette_probability and rng.random() < level.vignette_probability:
            clip_effects["vignette"] = self._vignette(rng)
        if level.specular_probability:
            clip_effects["specular"] = rng.random() < level.specular_probability
        occlusion = None
        if level.occlusion_probability and rng.random() < level.occlusion_probability:
            start = int(rng.integers(1, self.clip_length))
            occlusion = (start, start + int(rng.integers(1, 5)))

        images, masks = [], []
        size = (self.image_size, self.image_size)
        centroid = np.argwhere(mask).mean(axis=0)[::-1] / self.image_size - 0.5
        out_direction = centroid / max(np.linalg.norm(centroid), 1e-3)
        look_away_zoom = None
        polyp_removed = None
        if absence is not None and absence[2] == "inpaint":
            region = cv2.dilate(mask, np.ones((15, 15), np.uint8))
            polyp_removed = cv2.inpaint(image, region, 15, cv2.INPAINT_TELEA)
        if absence is not None and absence[2] == "out_of_view" and level.near_out_of_view:
            look_away_zoom = float(rng.uniform(1.8, 2.4))
        for index in range(self.clip_length):
            in_absence = absence is not None and absence[0] <= index < absence[1]
            if in_absence and absence[2] == "negative":
                source = self._load(negative_frames[index - absence[0]])
                matrix = self._affine(negative_path[index - absence[0]])
                warped = cv2.warpAffine(source, matrix, size, borderValue=0)
                warped_mask = np.zeros(size, np.uint8)
            elif in_absence and absence[2] == "inpaint":
                matrix = self._affine(path[index])
                warped = cv2.warpAffine(polyp_removed, matrix, size, borderValue=0)
                warped_mask = np.zeros(size, np.uint8)
            else:
                if in_absence and absence[2] == "out_of_view" and look_away_zoom is not None:
                    matrix = self._look_away(path[index], out_direction, look_away_zoom)
                else:
                    offset = out_direction * 1.1 if in_absence and absence[2] == "out_of_view" else (0.0, 0.0)
                    matrix = self._affine(path[index], offset)
                warped = cv2.warpAffine(image, matrix, size, borderValue=0)
                warped_mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST, borderValue=0)
                if occlusion is not None and occlusion[0] <= index < occlusion[1]:
                    warped, warped_mask = self._occluder(warped_mask, warped, rng)
            images.append(self._photometric(warped, rng, clip_effects))
            masks.append(warped_mask)

        if not masks[0].any():
            return self.sample(rng)

        pixels = (np.stack(images) - IMAGE_MEAN) / IMAGE_STD
        masks = np.stack(masks).astype(np.float32)
        return {
            "images": torch.from_numpy(pixels).permute(0, 3, 1, 2).float(),
            "masks": torch.from_numpy(masks),
            "frame_gaps": torch.from_numpy(gaps.astype(np.float32)),
            "present": torch.from_numpy(masks.reshape(len(masks), -1).any(axis=1)),
            "spliced_negative": torch.tensor([
                absence is not None and absence[2] == "negative" and absence[0] <= index < absence[1]
                for index in range(self.clip_length)
            ]),
            "source": str(frame.image_path),
            "absence_mode": "none" if absence is None else absence[2],
        }

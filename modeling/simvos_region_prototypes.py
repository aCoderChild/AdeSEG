"""Load the authors' mask-guided TokenLearner for three-region memory pools."""

import ast
from functools import lru_cache
from pathlib import Path
import subprocess

import torch
from torch import nn
import torch.nn.functional as F


SIMVOS_REVISION = "ef94709b9e8a4bda43276751be67f98ae1fc12e8"
SIMVOS_SOURCE = "model/models_vit.py"
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def author_token_learner():
    repo = ROOT / "external/SimVOS"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != SIMVOS_REVISION:
        raise RuntimeError("SimVOS checkout differs from the recorded revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", SIMVOS_SOURCE],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    path = repo / SIMVOS_SOURCE
    tree = ast.parse(path.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "TokenLearnerModuleV11_w_Mask"
    )
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


class RegionPrototypeHeads(nn.Module):
    """Three author TokenLearners and one zero-initialized gain per SAM2 layer."""

    def __init__(self, channels=64, prototypes=16, layers=4):
        super().__init__()
        if channels <= 0 or prototypes <= 0 or layers != 4:
            raise ValueError("Expected positive dimensions and four SAM2 layers")
        learner = author_token_learner()
        self.prototypes = int(prototypes)
        self.learners = nn.ModuleList(
            learner(in_channels=channels, num_tokens=prototypes)
            for _ in range(3)
        )
        self.gains = nn.Parameter(torch.zeros(layers))

    def assign(self, features, labels, mode):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global prototype control")
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError("Expected one NHWC memory-frame feature grid")
        height, width = features.shape[1:3]
        if labels.shape != (height, width):
            raise ValueError("Region labels do not match the memory grid")
        if not ((labels >= 0) & (labels <= 2)).all():
            raise ValueError("Expected background, foreground and boundary labels")

        outputs = []
        for region, learner in enumerate(self.learners):
            region_present = labels.eq(region)
            if mode == "regional":
                mask = region_present
                if not region_present.any():
                    continue
            else:
                if not region_present.any():
                    continue
                mask = torch.ones_like(labels, dtype=torch.bool)
            weights = mask.to(features.dtype)[None, None]
            prototypes, assignment = learner(features, weights)
            if not torch.isfinite(prototypes).all() or not torch.isfinite(assignment).all():
                raise RuntimeError("SimVOS produced nonfinite region prototypes")
            if mode == "regional":
                invalid = assignment.masked_select(~mask.flatten()[None, None, :].expand_as(assignment))
                if invalid.numel() and torch.count_nonzero(invalid).item():
                    raise RuntimeError("SimVOS assigned a prototype outside its region")
            outputs.append((region, prototypes, assignment))
        return outputs

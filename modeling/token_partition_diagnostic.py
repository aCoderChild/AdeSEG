"""Offline TokenCut diagnostic; never installed into SAM2 inference.

Reuse the author's normalized-cut implementation on a spatial feature grid.
Partitions have no foreground identity until independently anchored/evaluated.
"""
import importlib.util
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REVISION = "fed52cd5b60891baefd8ec7110dafa73be816ee1"


def partition_tokens(features, height, width):
    """Return upstream eigenvector and component mask for [H*W, C] features.

    Uses upstream defaults, including its graph threshold. A discarded dummy
    CLS entry satisfies TokenCut's API; no spatial token is dropped. This is
    an adaptation to SAM2 features, not a reproduction of DINO TokenCut scores.
    Inputs must be detached: this diagnostic does not provide gradients.
    """
    if height <= 0 or width <= 0 or features.ndim != 2 or features.shape[0] != height * width:
        raise ValueError("Expected [height * width, channels] spatial features")
    if height * width < 3 or features.requires_grad:
        raise ValueError("Use at least three detached spatial tokens")
    if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
        raise ValueError("Features must be finite and nonzero")
    repo = ROOT / "external/TokenCut"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError("TokenCut revision differs from diagnostic pin")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--",
                    "object_discovery.py"], cwd=repo, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location(
        "_upstream_tokencut_diagnostic", repo / "object_discovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spatial = features.cpu().float()
    # A constant thresholded graph admits arbitrary eigenvector partitions.
    # Reject it rather than claiming evidence of foreground/background structure.
    normalized = F.normalize(spatial, dim=1)
    adjacency = normalized @ normalized.T > 0  # upstream default tau=0
    off_diagonal = adjacency[~torch.eye(len(spatial), dtype=torch.bool)]
    if bool(off_diagonal.all()) or not bool(off_diagonal.any()):
        raise ValueError("Degenerate TokenCut graph: no discriminative off-diagonal edges")
    inputs = torch.cat([torch.zeros_like(spatial[:1]), spatial], dim=0)[None]
    result = module.ncut(inputs, (height, width), (1, 1), (3, height, width))
    return {"component": result[2], "eigenvector": result[5],
            "bipartition": result[5] > result[5].mean(),
            "seed": int(result[3]), "source_revision": revision}

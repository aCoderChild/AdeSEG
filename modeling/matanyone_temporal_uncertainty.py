"""Load MatAnyone's temporal uncertainty predictor from its pinned source."""

import ast
from functools import lru_cache
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MATANYONE_REVISION = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"
SOURCE = "matanyone/model/big_modules.py"


@lru_cache(maxsize=1)
def author_uncertainty_class():
    repo = ROOT / "external/MatAnyone"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if head != MATANYONE_REVISION:
        raise RuntimeError("MatAnyone source revision differs from the reviewed pin")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", SOURCE],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    path = repo / SOURCE
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "UncertPred"
    )
    namespace = {"torch": torch, "nn": nn, "F": F, "DictConfig": object}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["UncertPred"]


def build_temporal_uncertainty(pixel_dim=256, value_dim=256):
    module = author_uncertainty_class()
    config = SimpleNamespace(pixel_dim=pixel_dim, value_dim=value_dim)
    return module(config)

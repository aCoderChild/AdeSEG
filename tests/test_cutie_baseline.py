"""Prompt-budget rules only; real author-model validation is saved in Drive."""
import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys

from scripts.run_cutie_region_baseline import correction_mask


@pytest.mark.parametrize("arm", ["initial_only", "detector_refresh"])
def test_initialization_needs_a_detector_prediction_not_ground_truth(arm):
    positive, empty = np.ones((2, 2), bool), np.zeros((2, 2), bool)
    assert correction_mask(arm, False, True, positive)
    assert not correction_mask(arm, False, False, positive)
    assert not correction_mask(arm, False, True, empty)


@pytest.mark.parametrize("detected", [False, True])
def test_initial_only_never_inserts_later_prompts(detected):
    assert not correction_mask("initial_only", True, detected, np.ones((2, 2)))


@pytest.mark.parametrize("foreground", [False, True])
def test_refresh_accepts_every_detection_even_an_empty_predicted_mask(foreground):
    mask = np.full((2, 2), foreground)
    assert correction_mask("detector_refresh", True, True, mask)
    assert not correction_mask("detector_refresh", True, False, mask)


def test_unknown_policy_is_not_silently_substituted():
    with pytest.raises(ValueError):
        correction_mask("confidence_rule", True, True, np.ones((2, 2)))


def test_direct_cli_does_not_import_author_scripts_package():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_cutie_region_baseline.py"),
                             "--help"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_author_builder_uses_strict_weights_and_requested_cache(monkeypatch, tmp_path):
    from hydra.core.global_hydra import GlobalHydra
    import cutie.model.cutie as author
    import scripts.run_cutie_region_baseline as adapter

    calls = {}

    class Model:
        def __init__(self, cfg):
            calls["cache"] = cfg.model.resnet_model_path

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, weights, strict):
            calls["strict"] = strict

    monkeypatch.setattr(author, "CUTIE", Model)
    monkeypatch.setattr(adapter.torch, "load", lambda *args, **kwargs: {})
    # Other tests import SAM2, which initializes Hydra globally. Production
    # baselines run in separate processes; preserve the test suite's context.
    global_hydra = GlobalHydra.instance()
    previous = global_hydra.hydra
    global_hydra.clear()
    try:
        _, cfg = adapter.build_author_model(tmp_path / "weights.pth", "cpu", tmp_path)
    finally:
        global_hydra.hydra = previous
    assert calls == {"cache": str(tmp_path), "strict": True}
    assert cfg.max_internal_size == 480

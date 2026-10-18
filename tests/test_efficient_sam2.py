"""Adapter tests; actual model equivalence requires a separate runtime check."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from scripts.efficient_sam2_adapter import UPSTREAM_REVISION, selection_rows, save_overlay

ROOT = Path(__file__).resolve().parents[1]


def test_native_selection_measurement_counts_without_mutation():
    categories = np.tile(np.arange(4, dtype=np.uint8), (64, 16))
    native = np.zeros(4096, bool)
    native[:204] = True
    original = native.copy()
    rows = selection_rows([native] * 4, categories)
    assert np.array_equal(native, original)
    assert len(rows) == 4
    assert all(r["retained"] == 204 for r in rows)
    assert all(sum(v["retained"] for v in r["regions"].values()) == 204 for r in rows)
    assert rows[0]["regions"]["boundary"]["retained"] == 51


def test_native_selection_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="4096"):
        selection_rows([np.zeros(10)], np.zeros((64, 64)))
    with pytest.raises(ValueError, match="taxonomy"):
        selection_rows([], np.zeros((32, 32)))


def test_overlay_is_a_display_not_an_intervention(tmp_path):
    source = tmp_path / "frame.png"
    Image.new("RGB", (80, 60), (100, 150, 200)).save(source)
    destination = tmp_path / "overlay.png"
    save_overlay(source, np.zeros((64, 64), np.uint8), [0, 1], destination)
    with Image.open(destination) as result:
        assert result.size == (240, 60)


def test_author_tree_has_only_the_recorded_device_patch():
    upstream = ROOT / "external/Efficient-SAM2"
    if not (upstream / "sam2").exists():
        pytest.skip("Initialize Efficient-SAM2 submodule and apply recorded patch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    assert revision == UPSTREAM_REVISION
    patch = subprocess.check_output(["git", "diff", "--no-ext-diff", "--unified=0", "HEAD"], cwd=upstream, text=True)
    assert patch == (ROOT / "patches/efficient-sam2-mps.patch").read_text()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_author_rotary_functions_match_cpu_on_mps():
    # Separate process prevents the other SAM2 packages/configs from colliding.
    code = r'''
import sys, torch
sys.path.insert(0, "external/Efficient-SAM2")
from sam2.modeling.position_encoding import compute_axial_cis, apply_rotary_enc
from build_model.window_bypass import apply_rotary_enc_prune
torch.manual_seed(1)
q, k = torch.randn(1, 1, 16, 8), torch.randn(1, 1, 96, 8)
freq = compute_axial_cis(8, 4, 4)
cpu = apply_rotary_enc(q, k, freq, True)
mps = apply_rotary_enc(q.to('mps'), k.to('mps'), freq.to('mps'), True)
for reference, value in zip(cpu, mps):
    torch.testing.assert_close(value.cpu(), reference, rtol=1e-5, atol=1e-5)
mask = torch.arange(64) % 3 == 0
select = torch.cat([torch.ones(16, dtype=torch.bool), mask, torch.ones(16, dtype=torch.bool)])
pruned_k = k[:, :, select]
cpu = apply_rotary_enc_prune(q, pruned_k, freq, 96, True, mask, 16, 80)
mps = apply_rotary_enc_prune(q.to('mps'), pruned_k.to('mps'), freq.to('mps'), 96, True, mask.to('mps'), 16, 80)
for reference, value in zip(cpu, mps):
    torch.testing.assert_close(value.cpu(), reference, rtol=1e-5, atol=1e-5)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

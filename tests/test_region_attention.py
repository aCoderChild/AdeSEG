"""Author attention reuse, pointer separation and read-only observation."""

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from modeling.region_attention import (
    EFFICIENT_SAM2_REVISION,
    RegionAttentionObserver,
    attention_summary,
    author_attention_forward,
)


def test_author_class_ast_matches_pinned_repository():
    repo = Path(__file__).resolve().parents[1] / "external/Efficient-SAM2"
    relative = "sam2/modeling/sam/transformer.py"
    committed = subprocess.check_output(
        ["git", "show", f"{EFFICIENT_SAM2_REVISION}:{relative}"], cwd=repo, text=True
    )

    def attention(text):
        return ast.dump(
            next(
                n
                for n in ast.parse(text).body
                if isinstance(n, ast.ClassDef) and n.name == "Attention"
            )
        )

    assert attention(committed) == attention((repo / relative).read_text())
    assert author_attention_forward().__name__ == "forward"


def test_uniform_attention_has_unit_spatial_enrichment_and_separate_pointers():
    result = attention_summary(torch.full((10,), 0.1), [0, 0, 0, 0, 1, 1, 2, 2, -1, -1])
    assert result["pointer_mass"] == pytest.approx(0.2)
    for region in result["regions"].values():
        assert region["enrichment"] == pytest.approx(1)
    assert (
        attention_summary([0.5, 0.5], [0, -1])["regions"]["boundary"]["enrichment"]
        is None
    )
    with pytest.raises(ValueError, match="sum to one"):
        attention_summary([0.2, 0.3], [0, 1])


@pytest.mark.parametrize(
    "device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)
def test_chunked_author_measurement_preserves_output_and_denominator(device):
    observer = RegionAttentionObserver(
        SimpleNamespace(training=False), None, chunk_size=2
    )
    observer.labels = torch.tensor([0, 0, 1, 1, 2, 2, -1])
    observer.units = []
    q = torch.randn(1, 2, 5, 8, device=device)
    k = torch.randn(1, 2, 7, 8, device=device)
    v = torch.randn(1, 2, 7, 4, device=device)
    original = F.scaled_dot_product_attention
    out = observer._observe(original, 0, q, k, v)
    assert torch.equal(out, original(q, k, v))
    weights = (q @ k.transpose(-2, -1) / (8**0.5)).softmax(-1)
    expected = attention_summary(weights.mean((0, 1, 2)).cpu(), observer.labels)
    for region in expected["regions"]:
        assert observer.rows[0]["regions"][region]["mass"] == pytest.approx(
            expected["regions"][region]["mass"], abs=1e-6
        )


def test_native_rope_projection_and_pointer_exclusion_are_preserved(monkeypatch):
    native = Path(__file__).resolve().parents[1] / "external/MedSAM2"
    monkeypatch.syspath_prepend(str(native))
    from sam2.modeling.sam.transformer import RoPEAttention

    attention = RoPEAttention(
        embedding_dim=8, num_heads=1, kv_in_dim=4, feat_sizes=(2, 2), rope_k_repeat=True
    ).eval()
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 10, 4), torch.randn(1, 10, 4)
    with torch.no_grad():
        baseline = attention(q, k, v, num_k_exclude_rope=2)
        observer = RegionAttentionObserver(
            SimpleNamespace(training=False), None, chunk_size=2
        )
        observer.labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0, -1, -1])
        observer.units = []
        original = attention._attention
        monkeypatch.setattr(
            attention,
            "_attention",
            lambda q, k, v, attn_mask=None: observer._observe(
                original, 0, q, k, v, attn_mask
            ),
        )
        measured = attention(q, k, v, num_k_exclude_rope=2)
    assert torch.equal(baseline, measured)
    assert observer.rows[0]["pointer_tokens"] == 2


def test_observer_restores_native_methods_on_failure():
    class Memory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.utility_controller = None
            self.capture_attention = False
            self.layers = [
                SimpleNamespace(
                    cross_attn_image=SimpleNamespace(
                        _attention=F.scaled_dot_product_attention
                    )
                )
            ]

    module = Memory().eval()
    original = module.layers[0].cross_attn_image._attention
    with pytest.raises(RuntimeError, match="intentional"):
        with RegionAttentionObserver(module, None):
            raise RuntimeError("intentional")
    assert module.layers[0].cross_attn_image._attention is original
    assert not module._forward_pre_hooks


def test_six_label_schemes_reuse_one_attention_calculation():
    mappings = {}
    for source in ("predicted", "ground_truth"):
        for j, ratio in enumerate((0.01, 0.02, 0.04)):
            mappings[f"{source}_{ratio:.2f}"] = (
                torch.arange(4096).reshape(64, 64) + j + (source == "ground_truth")
            ) % 3
    observer = RegionAttentionObserver(
        SimpleNamespace(training=False), lambda index: mappings, chunk_size=2
    )
    observer._context(
        None,
        (),
        {
            "memory_units": [
                {"start": 0, "end": 4096, "frame_idx": 4, "conditioning": False}
            ],
            "num_obj_ptr_tokens": 2,
        },
    )
    calls = []
    author = observer.author

    def record(*args):
        calls.append(1)
        return author(*args)

    observer.author = record
    q, k, v = (
        torch.randn(1, 1, 3, 4),
        torch.randn(1, 1, 4098, 4),
        torch.randn(1, 1, 4098, 4),
    )
    output = observer._observe(F.scaled_dot_product_attention, 0, q, k, v)
    assert torch.equal(output, F.scaled_dot_product_attention(q, k, v))
    assert len(calls) == 2  # Query chunks, NOT chunks times six label definitions.
    row = observer.rows[0]
    assert len(row["label_schemes"]) == 6
    assert row["label_schemes"]["predicted_0.02"]["regions"] == row["regions"]
    for scheme in row["label_schemes"].values():
        assert scheme["pointer_mass"] == row["pointer_mass"]
        assert scheme["spatial_mass"] == pytest.approx(row["spatial_mass"])
        assert sum(r["tokens"] for r in scheme["regions"].values()) == 4096


def test_sensitivity_tables_retain_empty_regions_and_token_support(tmp_path):
    from scripts.measure_region_attention import (
        sequence_report,
        write_sensitivity_results,
        check_primary_agreement,
    )

    rows = []
    for layer in range(4):
        rows.extend(
            [
                {"layer": layer, **attention_summary([0.2, 0.3, 0.5], [0, 1, 2])},
                {"layer": layer, **attention_summary([0.4, 0.6], [0, 2])},
            ]
        )
    report = sequence_report(rows)
    fg = report["0"]["regions"]["foreground_interior"]
    assert fg["tokens"] == 0.5
    assert fg["mass"] == pytest.approx(0.15)
    assert fg["enrichment"] == pytest.approx(0.9)
    assert fg["empty_frames"] == 1 and fg["empty_frame_fraction"] == 0.5
    schemes = {
        f"{s}_{r:.2f}": report
        for s in ("predicted", "ground_truth")
        for r in (0.01, 0.02, 0.04)
    }
    result = {
        "sequence": "seq3",
        "status": "completed",
        "attention_frames": 2,
        "sensitivity": schemes,
    }
    output = write_sensitivity_results(tmp_path, [result])
    assert len(output) == 72
    assert "Empty frames" in (tmp_path / "sequence_results.md").read_text()
    check_primary_agreement(rows[0], rows[0])
    with pytest.raises(RuntimeError, match="did not reproduce"):
        check_primary_agreement(rows[0], rows[1])


def test_all_sequence_overlay_verification_detects_wrong_pixels(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from scripts.audit_memory_token_regions import audit_full_frame
    from scripts.measure_region_attention import verify_overlays
    from scripts.memory_token_io import digest

    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((8, 8, 3), 80, np.uint8)).save(image)
    Image.fromarray(np.ones((8, 8), np.uint8)).save(mask)
    audit, output = tmp_path / "audit", tmp_path / "output"
    audit.mkdir()
    output.mkdir()
    (audit / "summary.json").write_text(
        json.dumps({"completed": True, "dataset_sequences": 23})
    )
    (audit / "delivery.json").write_text(
        json.dumps({"wandb_artifact_upload_complete": True})
    )
    for i in range(1, 24):
        folder = audit / f"seq{i}"
        folder.mkdir()
        frame = SimpleNamespace(
            video_id=f"seq{i}", frame_index=1, image_path=image, mask_path=mask
        )
        prompt = None if i in (1, 7) else 0
        (folder / "summary.json").write_text(json.dumps({"prompt_frame": prompt}))
        (folder / "frame_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "frame": 1,
                        "image": str(image),
                        "mask": str(mask),
                        "image_sha256": digest(image),
                        "mask_sha256": digest(mask),
                    }
                ]
            )
        )
        audit_full_frame(
            frame,
            None,
            None,
            np.ones((8, 8), bool) if prompt is not None else None,
            folder,
            [0.01, 0.02, 0.04],
        )
    report = verify_overlays(audit, output)
    assert report["pixel_exact"] and report["frames"] == 23
    assert report["overlays_verified"] == (21 * 2 + 2) * 3
    assert (output / "all_23_sequences_regions.png").exists()
    wrong = audit / "seq1/frame_000001_ground_truth_ratio_0.01.png"
    with Image.open(wrong) as source:
        corrupted = source.convert("RGB")
    corrupted.putpixel((8, 30), (255, 0, 255))
    corrupted.save(wrong)
    with pytest.raises(ValueError, match="Overlay pixels disagree"):
        verify_overlays(audit, output)

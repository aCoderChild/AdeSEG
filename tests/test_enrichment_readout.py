import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.queue_enrichment_readout import estimate_recipe
from scripts.run_enrichment_readout import read_recipe, weighted_bias
from test_region_bias import setup
from modeling import region_bias


def test_weight_ratios_reciprocals_shuffle_counts_and_pointer_bias():
    labels = torch.arange(4096) % 3
    weights = [0.6, 4.0, 6.0]
    bias = weighted_bias(labels, 4, "boundary", 2, weights)
    inverse = weighted_bias(labels, 4, "boundary", 2, [1 / w for w in weights])
    assert torch.allclose(bias, -inverse)
    shuffled = weighted_bias(labels, 4, "shuffled", 2, weights)
    assert torch.equal(shuffled.sort().values, bias.sort().values)
    assert torch.equal(shuffled, weighted_bias(labels, 4, "shuffled", 2, weights))
    assert not torch.equal(shuffled, bias)
    assert not bias[-4:].any()
    assert not weighted_bias(labels, 4, "zero", 2, weights).any()


@pytest.mark.parametrize(
    "weights", [[0, 1, 2], [1, float("nan"), 2], [1, -1, 2], [1, 2]]
)
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        weighted_bias([0, 1, 2], 0, "boundary", 1, weights)


@pytest.mark.parametrize("mode", ["boundary", "shuffled", "zero"])
def test_all_four_layers_use_original_sdpa_and_restore(monkeypatch, mode):
    weights = [0.6, 4, 6]
    monkeypatch.setattr(
        region_bias,
        "region_log_bias",
        lambda labels, pointers, mode, seed: weighted_bias(
            labels, pointers, mode, seed, weights
        ),
    )
    model, state, logits, q, v = setup()
    original = F.scaled_dot_product_attention
    with region_bias.NativeRegionBias(model, mode, "seq3", 0) as controller:
        controller.capture(state, 0, logits)
        for block in model.memory_attention.layers:
            actual = block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
            expected = original(
                q[:, None], v[:, None], v[:, None], attn_mask=controller.mask
            )[:, 0]
            assert torch.equal(actual, expected)
        assert len(controller.trace) == 4
        assert not controller.bias[-4:].any()
    assert F.scaled_dot_product_attention is original


def test_recipe_uses_all_eligible_sequences_and_omits_undefined_groups(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"split": split, "video_id": name})
            for split, name in [("dev", "seq2"), ("dev", "seq3"), ("test", "seq16")]
        )
    )
    from scripts.memory_token_io import digest

    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "method": "sam2_calibration",
                "calibration_label_source": "predicted_0.02",
                "manifest_sha256": digest(manifest),
                "checkpoint_sha256": "checkpoint",
                "author_revision": "author",
                "author_repository": "external/sam2",
                "config": "cfg",
                "torch": "torch",
                "precision": "fp32",
                "apply_postprocessing": False,
                "memory_stride": 1,
                "integration_sha256": {},
            }
        )
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "scope": "full",
                "sequences": [
                    {"sequence": name, "status": "completed"}
                    for name in ("seq2", "seq3", "seq16")
                ],
            }
        )
    )
    (tmp_path / "delivery.json").write_text(
        json.dumps({"completed": True, "wandb_artifact_upload_complete": True})
    )
    for name, value in [("seq2", 2), ("seq3", 4), ("seq16", 6)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "attention_summary.json").write_text(
            json.dumps(
                {
                    "sequence": name,
                    "native_predictions_bit_identical": True,
                    "label_source": "predicted_0.02",
                    "layers": {
                        str(i): {
                            "regions": {
                                r: {
                                    "enrichment": None
                                    if name == "seq16" and r == "foreground_interior"
                                    else value + i
                                }
                                for r in [
                                    "background",
                                    "foreground_interior",
                                    "boundary",
                                ]
                            }
                        }
                        for i in range(4)
                    },
                }
            )
        )
    recipe = estimate_recipe(tmp_path, manifest)
    assert recipe["mean_enrichment"] == [5.5, 4.5, 5.5]
    assert set(recipe["eligible_sequences"]) == {"seq2", "seq3", "seq16"}
    assert len(recipe["contributing_sequences"]["foreground_interior"]) == 2
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    assert read_recipe(path, "inverse", manifest)[1] == [1 / 5.5, 1 / 4.5, 1 / 5.5]
    recipe["mean_enrichment"][0] = 100
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="changed"):
        read_recipe(path, "proportional", manifest)

    recipe = estimate_recipe(tmp_path, manifest)
    path.write_text(json.dumps(recipe))
    source = tmp_path / "seq2/attention_summary.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        read_recipe(path, "proportional", manifest)


def test_old_running_source_not_modified():
    # Dedicated adapter: active queue still points at its unchanged entry point.
    source = Path("scripts/run_enrichment_readout.py").read_text()
    assert "author.main()" in source
    assert "enrichment_recipe_sha256" in source

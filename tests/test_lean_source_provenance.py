"""Verify strict source pinning for NEW Lean refiner checkpoints.

Legacy v1 class is not modified; only the public structured implementation
requires an additional full source fingerprint.
"""
import copy
from pathlib import Path
import pytest
import torch

from modeling.structured_feature_aggregation import StructuredMultistageFeatureAggregator
from modeling.lean_implementation_digest import implementation_digest
from modeling.foreground_conditioned_refiner import _source_sha256
import modeling.multiscale_channel_value_refiner_v1 as legacy

CONFIG = {"early_channels": 2, "deep_channels": 4, "hidden": 8, "attention_size": 4}
HASHES = {"backbone_sha256": "a"*64, "feature_pipeline_sha256": "b"*64,
          "train_manifest_sha256": "c"*64}

def payload_from(model):
    return {
        "format": "multiscale_channel_value_refiner_v1",
        "research_only": True, "training_steps": 2,
        "state_dict": copy.deepcopy(model.state_dict()),
        "source_sha256": _source_sha256(Path(legacy.__file__)),
        "implementation_sha256": implementation_digest(),
        "mode": model.mode,
        "config": dict(CONFIG), **HASHES,
    }

def test_structured_checkpoint_accepts_complete_source_digest():
    model = StructuredMultistageFeatureAggregator(**CONFIG)
    model.load_trained_state(payload_from(model), **HASHES)
    assert model._trained_checkpoint_loaded is True

def test_structured_checkpoint_rejects_stale_implementation():
    model = StructuredMultistageFeatureAggregator(**CONFIG)
    payload = payload_from(model)
    payload["implementation_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="implementation SHA256"):
        model.load_trained_state(payload, **HASHES)
    assert model._trained_checkpoint_loaded is False

def test_legacy_source_only_checkpoint_requires_explicit_opt_in():
    model = StructuredMultistageFeatureAggregator(**CONFIG)
    payload = payload_from(model)
    payload.pop("implementation_sha256")
    with pytest.raises(ValueError, match="implementation SHA256"):
        model.load_trained_state(payload, **HASHES)
    model.load_trained_state(payload, **HASHES, allow_legacy_source_only=True)
    assert model._trained_checkpoint_loaded is True

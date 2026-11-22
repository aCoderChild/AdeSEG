"""Smoke protocol rejects false-positive 'passed' results."""
import pytest

from scripts.smoke_lean_memory_fusion import verify_smoke_outputs


def case(frames=(1, 2, 3), *, effective=1, anchor=1):
    return {
        "predictions": [
            {"frame_index": i, "prediction_sha256": f"{i}" * 64,
             "logits_sha256": f"{i+1}" * 64}
            for i in frames
        ],
        "per_frame_resident_memory": [{"frame": i} for i in range(4)],
        "effective_memory_writes_affecting_future": effective,
        "anchor_similarity_audit": {"eligible_writes": anchor},
    }


def test_smoke_requires_native_identity_bitwise_and_future_effect():
    native = case()
    identity = case()
    fusion = case(effective=2, anchor=3)
    ok = verify_smoke_outputs(
        {"native": native, "identity": identity, "fusion": fusion},
        require_anchor=True)
    assert ok["native_identity_logits_bitwise_equal"] is True
    assert ok["fusion_future_effective_writes"] == 2
    identity["predictions"][1]["logits_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="diverged"):
        verify_smoke_outputs({"native": native, "identity": identity})


def test_smoke_rejects_ineffective_fusion_and_missing_tamp_anchor():
    native, identity = case(), case()
    with pytest.raises(RuntimeError, match="no BF16-visible"):
        verify_smoke_outputs({
            "native": native, "identity": identity,
            "fusion": case(effective=0),
        })
    with pytest.raises(RuntimeError, match="TAMP anchor"):
        verify_smoke_outputs({
            "native": native, "identity": identity,
            "fusion": case(effective=1, anchor=0),
        }, require_anchor=True)

"""MedSAM2-Latest must never be mistaken for the separate Hiera-B+ Lean result."""
import hashlib
import pytest
from modeling.medsam2_latest import (
    MEDSAM2_LATEST_SHA256,
    verify_medsam2_latest,
)


def test_hiera_t_medical_weights_are_distinct_from_hiera_b_plus_lean():
    lean_hiera_bplus_sha256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
    assert MEDSAM2_LATEST_SHA256 != lean_hiera_bplus_sha256


def test_wrong_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "MedSAM2_latest.pt"
    checkpoint.write_bytes(b"not-the-medical-pretrained-weights")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MEDSAM2_LATEST_SHA256
    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        verify_medsam2_latest(checkpoint)

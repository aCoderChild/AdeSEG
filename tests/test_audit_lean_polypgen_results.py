import numpy as np
from PIL import Image
import pytest

from scripts.audit_lean_polypgen_results import check_mask


def test_mask_audit_recomputes_dice_and_absence(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(path)
    truth = np.array([[True, True], [False, False]])
    row = {"scores": {"dice": 2 / 3}, "present": True, "absent_fp": None}
    assert check_mask(path, truth, row) == pytest.approx(2 / 3)
    row["scores"]["dice"] = 0.9
    with pytest.raises(ValueError, match="Dice does not match"):
        check_mask(path, truth, row)
    empty = {"scores": {"dice": 0.0}, "present": False, "absent_fp": False}
    with pytest.raises(ValueError, match="False-alarm"):
        check_mask(path, np.zeros_like(truth), empty)

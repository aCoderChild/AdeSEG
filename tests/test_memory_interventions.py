import numpy as np

from modeling.memory_interventions import corrupt_mask


def test_mixed_corruption_is_deterministic_and_cumulative():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    outputs = [corrupt_mask(mask, level, "mixed", frame_index=2)
               for level in (0.25, 0.50, 0.75, 1.00)]
    assert all(np.array_equal(value, corrupt_mask(mask, level, "mixed", 2))
               for value, level in zip(outputs, (0.25, 0.50, 0.75, 1.00)))
    assert not np.array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[1], outputs[2])
    assert not np.array_equal(outputs[2], outputs[3])

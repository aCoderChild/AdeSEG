import pytest
from modeling.lean_memory_shadow import memory_shadow_observer

def test_known_invalid_shadow_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='real-video replay parity failed'):
        with memory_shadow_observer(None, [], {1}, tmp_path, []):
            pass

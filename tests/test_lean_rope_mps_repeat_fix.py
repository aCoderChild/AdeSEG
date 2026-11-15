import pytest
import torch
from modeling.lean_rope_mps_repeat_fix import corrected_mps_key_frequency_repeat


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS required')
def test_mps_repeats_whole_complex_grid_and_restores_original():
    class Module:
        @staticmethod
        def _repeat_freqs_for_k(freqs, repeats):
            return freqs.unsqueeze(1).expand(-1, repeats, -1).reshape(-1, freqs.shape[-1])

    base = torch.view_as_complex(torch.randn(5, 3, 2)).to('mps')
    original = Module._repeat_freqs_for_k
    with corrected_mps_key_frequency_repeat(Module):
        actual = Module._repeat_freqs_for_k(base, 3).cpu()
        torch.testing.assert_close(actual, base.cpu().repeat(3, 1), rtol=0, atol=0)
        assert not torch.equal(actual, original(base, 3).cpu())
    assert Module._repeat_freqs_for_k is original

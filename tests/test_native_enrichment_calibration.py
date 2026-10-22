import torch
import torch.nn.functional as F

from scripts.calibrate_native_enrichment import NativeEnrichmentObserver
from test_region_bias import setup


def test_observer_uses_native_outputs_and_measures_all_layers():
    model, state, logits, q, v = setup()
    native = F.scaled_dot_product_attention(q[:, None], v[:, None], v[:, None])[:, 0]
    original = F.scaled_dot_product_attention
    with NativeEnrichmentObserver(model, "zero", "seq2", 0) as observer:
        observer.capture(state, 0, logits)
        for block in model.memory_attention.layers:
            actual = block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
            assert torch.equal(actual, native)
        assert len(observer.observer.rows) == 4
        for row in observer.observer.rows:
            assert row["pointer_tokens"] == 4
            assert sum(r["tokens"] for r in row["regions"].values()) == 4096
    assert F.scaled_dot_product_attention is original

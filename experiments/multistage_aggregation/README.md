# Multistage aggregation candidate

**Learned model:** `modeling/multiscale_channel_value_refiner_v1.py` blends early and deep features using channel-value Q/K/V attention and foreground/background similarity; its residual head modifies native logits. Separate `modeling/lean_channel_value_routing_v1.py` hooks the predictor for output-only (B), memory-only (C) and both (D). Historical runtime certification depends on exact original hashes and artifacts. Use the untouched backup commit to replay hash-bound experiments.

**Memory-only context ablation:** `modeling/lean_multiscale_anchor.py`, `scripts/run_lean_auto_multiscale.py`. The pooled-anchor score did not improve Dice materially on the matched eight-sequence study and is **not** evidence for learned channel-value aggregation.

**Training:** `scripts/train_aggregation.py` is a clean, offline trainer for labeled training feature captures. It does not automatically reproduce original online video routing or establish clinical generalization. Follow the frozen [method protocol](../../protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md).

# TTMR — Target-Aware Temporal Memory Refinement

## Objective

TTMR is a candidate fixed-budget, causal historical-token selector that jointly considers target similarity, native memory relevance, redundancy and temporal diversity.

**Distinction from the baseline:** Lean-SAM2 already supplies TAMP (target-anchored memory pruning), TCIM (temporal condensation/insurance) and TARR (risk-aware routing). TTMR studies a precisely specified *incremental selection criterion* rather than renaming those native mechanisms.

## Memory-token selection

For candidates \(m_i\in\mathbb R^{C_m}\), with \(t_i<t\), reference target anchors \(A_0\), native usage \(u_i\in[0,1]\) and temporal decay \(\tau\), define

\[
q_i = \lambda_a\max_{a\in A_0}\cos(m_i,a)
      +(1-\lambda_a)u_i+\lambda_r\exp(-(t-t_i)/\tau).
\]

At each step select the token with the largest diversity-adjusted score:

\[
i^*=\arg\max_{i\notin S}\left[q_i-
\lambda_d\max_{j\in S}\max(0,\cos(m_i,m_j))\right].
\]

Selection has a fixed budget \(|S|=B_m\), unique token indices, and source-frame retention quotas where feasible. The token source times must be **strictly before** frame \(t\).

The original FGPI pseudo-anchors are **image-feature-space** vectors; TTMR anchors must reside in **memory-token space**. A shared target representation therefore needs explicit encoder/projection alignment—channel dimensions are not presumed identical.

## Multiscale scoring without additional memory tokens

The first implementation path uses pooled context to **score existing native
spatial memory tokens**, not to append coarse-scale vectors to memory. For a
dense, frame-major memory tensor `[F,C,H,W]`, `score_multiscale_memory`
computes local anchor cosine similarities and pooled-scale (default 2x/4x)
similarities. Coarse scores are interpolated to the original `H x W` grid;
the output contains exactly `F*H*W` scalar scores in native flattening order.

With `multiscale_weight=\rho`, TTMR blends anchor evidence as
`(1-rho) * local_score + rho * multiscale_score`. The
`rank_spatial_memory` / `TTMR.select_spatial` reference entrypoint
passes these scores through the existing causal, fixed-budget selector.
`multiscale_weight=0` reproduces local-anchor reference ranking.

This remains **offline/shadow selection**. It does not allocate new memory
tokens, modify RoPE, change the native memory encoder, or establish that
multiscale scoring improves video segmentation. It also does not yet include
a validated corruption-reliability estimator; partially corrupted frames
remain a research hypothesis rather than a demonstrated capability.

The original `modeling/lean_multiscale_anchor.py` installer expects
`_cache_tarr_affinity_from_new_memory`, which is absent from the pinned
Lean-SAM2 submodule. It now reports an explicit compatibility error rather
than silently being treated as a working live hook. A future adapter should
intervene at Lean's native score-fusion / `pending_keep_idx` selection path
and preserve paired feature/position indices and per-frame metadata.

### Offline usage example

```python
from adeseg.modules.ttmr import TTMR, TTMRConfig

selector = TTMR(TTMRConfig(budget=128, multiscale_weight=0.5))
result = selector.select_spatial(
    memory_maps,      # [F,C,H,W], native historical spatial memory
    memory_anchors,   # [A,C], in the SAME memory-token space
    usage_scores,     # [F*H*W], native attention usage in [0,1]
    frame_ids,        # [F], strictly earlier than current_frame
    current_frame=t,
)
indices = result["indices"]  # original frame-major, row-major indices
```

The current CPU reference selector permits at most 4,096 candidates and
at most 512 selected tokens; these are prototype limits, not claims about
global Lean-SAM2 memory capacity. For fixed-budget comparisons, match
actual native retained-token counts and include all protected-memory costs.

## Research interfaces

- `adeseg/modules/ttmr/__init__.py`: TTMR reference API.
- `adeseg/modules/ttmr/selection.py`: reproducible candidate-ranking algorithm and spatial wrapper.
- `adeseg/modules/ttmr/multiscale.py`: stateless contextual scoring of original-resolution tokens.
- `modeling/lean_multiscale_anchor.py`: multiscale anchor similarity experiments.
- `adeseg/inference/lean_memory_routing.py`: existing mask/memory-logit routing, **not** the TTMR token hook.

The current TTMR code selects indices in offline/shadow mode; full Lean-SAM2 integration must preserve the native memory layout, positional encodings, temporal source identifiers and TAMP/TCIM budget.

The relevant comparison is **native Lean memory versus TTMR at identical prompt and token budget**, measuring long-horizon Dice and empty-frame errors alongside token count, latency and memory footprint.

References: [Lean-SAM2](https://arxiv.org/abs/2607.19811), [SAM2Long (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html).

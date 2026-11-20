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

## Research interfaces

- `adeseg/modules/ttmr.py`: TTMR reference API.
- `adeseg/memory/target_aware_temporal.py`: reproducible candidate-ranking algorithm.
- `modeling/lean_multiscale_anchor.py`: multiscale anchor similarity experiments.
- `adeseg/inference/lean_memory_routing.py`: existing mask/memory-logit routing, **not** the TTMR token hook.

The current TTMR code selects indices in offline/shadow mode; full Lean-SAM2 integration must preserve the native memory layout, positional encodings, temporal source identifiers and TAMP/TCIM budget.

The relevant comparison is **native Lean memory versus TTMR at identical prompt and token budget**, measuring long-horizon Dice and empty-frame errors alongside token count, latency and memory footprint.

References: [Lean-SAM2](https://arxiv.org/abs/2607.19811), [SAM2Long (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html).

# RMFA — Region-Aware Multistage Feature Aggregation

## Objective

RMFA integrates high-resolution spatial features and deep memory-conditioned representations so that the model can refine foreground/background separation and lesion boundaries **without replacing the pretrained mask decoder**.

## Six-stage architecture

Let \(E_t\in\mathbb R^{B\times C_e\times H_e\times W_e}\) be early FPN features, \(H_t\in\mathbb R^{B\times C_d\times H_d\times W_d}\) the deep memory-conditioned features, and \(L_t^0\in\mathbb R^{B\times1\times H\times W}\) native logits.

1. **Projection and alignment:** separate 1×1 projections of \(E_t\) and \(H_t\), aligned to \((H,W)\) by interpolation, then concatenated into \(U_t\).
2. **Compact region context:** pool \(U_t\) to a bounded attention grid and compute \(p_t=\operatorname{Pool}(\sigma(L_t^0))\).
3. **Region-aware channel Q/K/V:** compute channel correlation separately for soft predicted foreground and background:
   \[
   A_t=\operatorname{Softmax}\left(\operatorname{Corr}(Q_t,K_t,p_t)-\operatorname{Corr}(Q_t,K_t,1-p_t)\right).
   \]
   Here \(\operatorname{Corr}(Q,K,w)=\operatorname{Norm}(\operatorname{Flatten}(Q\odot w))\operatorname{Norm}(\operatorname{Flatten}(K\odot w))^\top\). Weighting both Q and K makes its inner product effectively quadratic in the region probability.
4. **Feature fusion:** mix V with \(A_t\), project and upsample to the spatial feature resolution, then add a residual correction to \(U_t\).
5. **Region contrast:** compute soft FG/BG prototypes from the predicted region probability; their cosine-similarity difference supplies a spatial contrast map.
6. **Mask refinement:** a small Conv–GELU–Conv head returns \(\Delta L_t\) and the output is
   \[
   L_t=L_t^0+\Delta L_t.
   \]

The final convolution is initialized to zero, so the residual equals zero **at initialization**. The pretrained decoder parameters remain unchanged.

## Training strategy and controlled routing

The source-compatible structured adapter is `modeling/structured_feature_aggregation.py`, exposed under `adeseg/modules/rmfa.py`. The six stateless computation blocks live in `modeling/aggregation_blocks/`. No duplicate trainable implementation is created.

`scripts.train_lean` trains the adapter from frozen, source-authenticated training captures. The segmentation objective includes boundary-weighted BCE, weighted IoU and soft Dice; an optional preservation term restricts divergence from selected native predictions.

Start with **output-only refinement**. Sending corrected masks into the memory encoder is a separate, explicit ablation because a segmentation error could otherwise affect later frames. The routing controls are implemented in `modeling/lean_channel_value_routing_v1.py`.

## Scientific comparisons

Native decoder versus RMFA; global versus foreground/background channel attention; region contrast on/off; output-only versus memory feedback. Compare on matched prompts, backbone weights, source frames and image resolution.

The original Lean-SAM2 foundation already supplies memory-conditioned features; RMFA's contribution is the **specific region-conditioned multilevel interaction and residual output**, not the existence of attention or FPNs.

References: [SAM2](https://arxiv.org/abs/2408.00714), [CMSA-Net (MICCAI 2026)](https://papers.miccai.org/miccai-2026/0176-Paper1232.html).

## Default implementation mode

The public `RMFAConfig` defaults to `foreground_background_asinh`, matching `python -m scripts.train_lean`. In this mode foreground/background channel attention and region contrast are applied, while the logits provided to the learned residual head are transformed by `asinh`. This mode name is part of the checkpoint contract; the other attention modes remain available only as separately trained ablations.

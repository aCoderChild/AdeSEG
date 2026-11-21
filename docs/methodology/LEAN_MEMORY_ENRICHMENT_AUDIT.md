# Verified architecture and mathematical limits: Lean-SAM2 multiscale memory fusion

**Status:** source-level audit and CPU-regression contracts. This note does not
claim improved video segmentation, TAMP ranking, or clinical performance.
Source revision and a trained checkpoint must be frozen for every experiment.

## Sources, precise insertion point, and feature identity

- [SAM 2](https://arxiv.org/abs/2408.00714): streaming video
  segmentation uses image encoding, previous-memory attention, mask decoding,
  and memory writing. Its stride-4/stride-8 image features support mask decoding.
- [Lean-SAM2](https://arxiv.org/abs/2607.19811):
  TAMP scores historical tokens using attention usage and similarity to
  prompt-memory anchors; TCIM/TARR remain native in this experiment.
- [CMSA-Net](https://arxiv.org/abs/2602.22821)
  ([source](https://github.com/wangtong627/CMSA-Net/blob/main/cmsa_project/lib/model_arch/cmsa_net.py)):
  projects multiple **spatial image-encoder stages**, then employs
  scale-dependent **causal temporal cross-attention** to update its deep
  segmentation representation. Our module borrows the complementary-scale
  principle, **not** the cross-frame attention mechanism.
- [SAM2Refiner (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Towards_Fine-grained_Interactive_Segmentation_in_Images_and_Videos_ICCV_2025_paper.html):
  uses hierarchical multiscale SAM2 features for fine-grained decoding,
  corroborating the potential value of early features while investigating a
  different insertion point.

**Important implementation precision:** The features consumed by the live
hook are the earlier entries of SAM2's `backbone_fpn` **after**
`sam_mask_decoder.conv_s0` and `conv_s1` have projected those levels to
**32/64 channels** in the pinned `forward_image` path. They originate from
the image encoder but are not raw, unprojected Hiera stage activations.
The final image embedding is the deeper 256-channel level. The native
memory encoder produces a mask-conditioned **64-channel** memory map.

For a frame `t`, first *read* retained historical memory to obtain
a conditioned representation and predict the current mask. Separately,
the unconditioned deep image feature plus the predicted mask enter the native
memory encoder. The adapter then writes an enriched memory, without
replacing the decoder or the memory encoder:

```text
I_t -> Image encoder/FPN -> early E_t^0, E_t^1; deep D_t
                  D_t + bank_{<t} -> native memory attention
                  native attention output + early skip -> native decoder -> mask_t
                  D_t + mask_t -> native memory encoder -> M_t
                  E_t^0 + E_t^1 + M_t (+ predicted-mask prior) -> adapter -> Mtilde_t
                  Mtilde_t -> native memory bank; later reads -> native TAMP
```

Conditioning-frame memory remains native by default. The learned adapter
uses earlier FPN scales from **the same frame** and never accesses future
images or labels at inference time. No intermediate memory-encoder
activations are added as separate feature sources.

## Mathematical model

Let the native memory be
`M_t in R^(B x C_m x H_m x W_m)`, with `C_m=64`.
Let the two earlier image feature maps be
`E_t^(l) in R^(B x C_l x H_l x W_l)` where `C_0=32` and
`C_1=64` for the pinned projection path. Define
`N_m=H_m W_m` and
`A_l = P_l(AreaPool(E_t^(l), (H_m,W_m)))`,
where each `P_l` maps `C_l -> C_h`.
Let `Q(M_t)` map `C_m -> C_h` and
`P(mask_t) = AreaPool(sigmoid(mask_logits_t), (H_m,W_m))`.
The actual module computes a learned convolutional correction

```math
Z_t = Mix_\theta([A_0, A_1, Q(M_t), P(mask_t)])
R_t = OutputProj_\theta(Z_t)
s_t = max(\operatorname{Std}(M_t), 10^{-6})
\Delta_t = \alpha s_t \tanh(R_t)
\widetilde M_t = M_t + g_t\Delta_t
```

where the object-presence gate `g_t` is binary when
`preserve_native_no_object=True` (default) and `alpha=0.25`.
Ablation mode `no_mask` zeroes only the **explicit extra mask prior** in
the adapter; the native `M_t` remains mask-conditioned.

### Proposition 1. Exact tensor geometry and write-time token count

**Assumptions:** Valid source tensors, compatible 1x1 projections,
interpolation to the fixed native grid and the unchanged residual-output
projection. Then both `R_t` and `Delta_t` are in
`R^(B x C_m x H_m x W_m)`. Addition to `M_t` therefore produces
`Mtilde_t` with the **same** shape, channel dimension and token count
`N_m = H_m W_m`. Its original positional encoding is returned unchanged.
This follows directly from the projection output channels and the use of
stride-one, same-padded convolutions in the adapter.

This does **not** prove equal total runtime, equal peak VRAM, or equal
number of *resident* historical tokens over time, because changed features
can alter TAMP attention usage, compression, TCIM/insurance decisions, and
other state transitions.

### Proposition 2. Per-element correction bound

Before floating-point rounding, `|tanh(z)| <= 1`, hence for each
memory entry and channel/spatial position,

```math
|\widetilde M_t[j]-M_t[j]| \le g_t\alpha\max(\operatorname{Std}(M_t),10^{-6})
```

with `alpha=0.25` by default. If `g_t=0`, the output is exactly
the native no-object memory (the implementation checks this invariant).
Finite-precision casting can introduce an additional small rounding
difference; measure the actual BF16 stored values, not only FP32 deltas.

This bound controls the **magnitude** of the update, but it does not
guarantee a meaningful *angular* change or preserve any mask accuracy.

### Proposition 3. Causality under the native streaming protocol

Assume native SAM2 inference at frame `t` only uses frames through `t`,
stored memory from earlier frames, and legal prompts through `t`.
The adapter computes `Delta_t` solely from `E_t`, `M_t`, and the
current predicted mask. Thus `Mtilde_t` depends only on information
available by time `t`. By induction, all future memory reads of
enriched historical entries remain causal. The independently generated
GT-conditioned teacher memory is used **only in TRAIN capture/training**
and not written to the inference bank.

### Proposition 4. Conditional cosine-anchor stability bound

Let `x` be a nonzero native memory token, `delta` its correction,
and `y=x+delta` also nonzero. For any fixed unit target anchor `a`,

```math
|cos(y,a)-cos(x,a)| <=
||y/||y||-x/||x||||_2
<= 2||delta||_2/||x||_2.
```

The same bound holds when taking `max_a` over a fixed set of unit
anchors, because `max` is 1-Lipschitz in the sup-norm of its arguments.
If `||delta||_2 < ||x||_2` the updated token cannot be zero.
The actual adapter's per-element bound alone **does not guarantee** this
relative vector-norm condition for every token, especially low-norm tokens.

**Very important:** TAMP combines anchor similarity with attention-usage
scores that may themselves change when enriched memory is read. This
conditional cosine bound therefore **cannot prove preserved or improved
TAMP rankings/top-K**, nor can it prove higher Dice. Only direct comparisons
of actual TAMP indices, attention statistics and evaluated masks can.

### Proposition 5. Computational interpretation

The adapter uses only spatial interpolation and small convolutional
projections/mixing at the native `H_m x W_m` grid after the two early
feature maps have been pooled. It adds **no temporal KV tokens**.
Its additional spatial computation and activation memory are *nonzero*
and must be profiled. The feature values may change native memory-attention
outputs and consequently later memory management.

## Why this is multiscale (and what it is not)

The adapter has **two distinct earlier spatial-resolution image-feature
sources**. Both are projected into the native memory feature space and
fused with a memory embedding. This meets the architectural definition of
multiscale *spatial* representation fusion.

It does **not** preserve all original high-frequency detail after pooling,
does **not** perform CMSA-Net's cross-frame multiscale temporal attention,
and does **not** guarantee that a trained network actually uses either early
scale. The `no_early`, `early0_only` and `early1_only` arms exist to
test these alternatives. Check gradients and learned effects rather than
relying on the presence of tensor branches in a diagram.

## Failure modes, guardrails and testability

| Failure mode | Guard / observation | Limitation |
|---|---|---|
| Source FPN stage/channel mismatch | Runtime shape, hierarchy and per-stage channel checks | Requires pinned Lean backbone |
| Memory/positional encoding mismatch | Fail-closed native grid and positional checks | Cannot prove geometry is clinically meaningful |
| Invalid anchor geometry/NaN | Fail-closed channel, rank and finite checks | Anchor availability may be legitimately zero |
| No-object hallucinated memory correction | Exact native-memory preservation gate | May suppress useful recovery after false-negative presence |
| FP32 update rounded away at write | BF16-value telemetry; run-level future-effect guard | Not a Dice gain guarantee |
| Only last-frame memory changed | Causally effective future-write counter | Some sequences may legitimately have zero effective writes |
| Oracle teacher correction unreachable | TRAIN teacher-feasibility audit | Teacher reconstruction is not propagation optimization |
| Full adapter learns only mask/memory | Separately trained no-early control | Extra parameterization can explain gains |
| Only one early scale matters | Separately trained single-scale controls | No guarantee both scales are needed |
| Explicit mask prior duplicates native mask | Separately trained no-mask mode | No-mask mode still has mask-conditioned native memory |
| TAMP anchors unavailable | Optional `--require-target-anchors` | Does not prove top-K improvement |
| Prompt, weights, train split or GT leakage | SHA-bound manifests/checkpoints and inference/eval separation | External prompt-mask producer must itself be audited |
| Full end-to-end claim | Separate FGPI and Lean backbone integration not yet established | Existing memory ablations start from frozen automatic masks |

## Experiments needed for a thesis claim

1. Native Lean-SAM2 vs zero-residual identity; require bit-exact
   predictions and unchanged memory trajectory.
2. Native vs separately trained `full`, `no_early`,
   `early0_only`, `early1_only`, `no_mask` under the same prompts,
   split, backbone and hyperparameter policy.
3. On DEV, inspect BF16-effective writes that **precede** a later scored
   frame, teacher reachability, native/enriched anchor-cosine drift, and
   frame-level storage audits.
4. For any direct TAMP retention claim, instrument actual historical
   memory-attention usages, selected token indices, anchor similarities,
   token-retention overlap and cost at fixed budgets. A fixed-anchor
   write-time cosine diagnostic is insufficient.
5. Freeze DEV choices before TEST; report paired sequence-level Dice/IoU,
   GT-empty false alarms and resource metrics. Resample independent
   source groups rather than correlated frames for uncertainty.
6. If improvements vanish against `no_early`, do not claim
   multiscale information caused them. If `full` fails against the best
   single-scale model, do not claim a benefit of *combining* both scales.

**Conclusion:** The design is a plausible, controlled SAM2 memory-write
adapter motivated by CMSA-Net's complementary spatial scales. The
mathematics guarantees a causal same-shaped bounded memory update
under stated assumptions, **not** better temporal propagation.

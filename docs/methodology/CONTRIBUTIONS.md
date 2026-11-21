# Research contributions and evidence status

**Active experimental track:** automatic video-polyp segmentation using
detector-guided initialization as a separate RQ, **same-frame multiscale
enrichment of SAM2's mask-conditioned temporal memory**, and evaluation
under **native Lean-SAM2** TAMP/TCIM/TARR. PolypGen is a colonoscopy proxy:
this is not clinical adenoid-segmentation validation.

| RQ | Current intended contribution | Status and comparison |
|---|---|---|
| **RQ1 — Target initialization (FGPI)** | Combine detector box with feature-selected positive and negative points | FGPI exists as a separate track. **Not integrated inline** with the current Lean memory-fusion runner, which begins from an externally frozen automatic mask. Compare detector box only vs box+selected points in a separately source-matched experiment. |
| **RQ2 — Multiscale temporal-memory enrichment** | Pool/project early SAM2 image/FPN levels and add a learned, bounded same-shape residual **after native mask-conditioned memory encoding** | Adapter, training captures, model checkpoint authentication, live write hook and matched ablations exist; **held-out temporal gains remain unproven**. Compare native, identity, full fusion, no-early, both single-scale and no-extra-mask controls. |
| **RQ3 — Efficient native temporal-memory management** | Test enriched memory under the **existing** Lean-SAM2 TAMP/TCIM/TARR selection and storage constraints | Native token selector is **unchanged**. Measure causal BF16 memory writes, per-frame retention, anchor cosine drift, native TAMP retained-index overlap, Dice and runtime. Do **not** claim a new TAMP selector is contributed. |

## What is not yet established

- A validated end-to-end **YOLO → FGPI → Lean-SAM2 memory-write enrichment**
  pipeline on the same model weights and feature representation.
- An adapter trained by future-frame segmentation loss or unrolled video
  backpropagation. The current trainer uses **TRAIN-only same-frame
  oracle-memory distillation** as a surrogate objective.
- Better segmentation accuracy, better TAMP anchor selection, lower peak
  GPU memory or clinically meaningful adenoid segmentation.

These are research questions, not verified benefits. The source-specified
math is in [the active architecture](OVERVIEW.md) and
[the mathematical/code audit](LEAN_MEMORY_ENRICHMENT_AUDIT.md).
For run commands and strict controls, consult
[the Lean memory-fusion preflight](../experiments/LEAN_MEMORY_FUSION_PREFLIGHT.md).

## Historical/independent contributions (not enabled in the active runner)

The original project also proposed and investigated:

| Legacy mechanism | Original question | Important distinction |
|---|---|---|
| [FGPI](FGPI.md) | Feature-guided target prompting | May use another encoder/backbone without explicit integration. |
| [RMFA](RMFA.md) | Multistage feature fusion for **output-mask-logit refinement** | **Not** the post-memory-encoder residual enrichment adapter. |
| [TTMR](TTMR.md) | Additional target-aware temporal memory token selector | **Not** native Lean-SAM2 TAMP. |

Preserving these files supports historical reproducibility. Do not
conflate these mechanisms with the active memory-fusion implementation
or claim their end-to-end integration without a separate validated run.

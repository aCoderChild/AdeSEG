# Research contributions

The proposed framework studies three complementary, technically isolated interventions.

| Module | Proposed contribution | Core comparison |
|---|---|---|
| **FGPI** | Detector-conditioned pseudo-target-anchor selection using spatial/feature diversity and foreground/background contrast | Same YOLO box + geometric/random points versus feature-selected points |
| **RMFA** | Joint early/high-resolution and memory-conditioned representation with FG/BG channel-value attention and residual mask-logit correction | Native logits versus region-aware residual, plus attention/contrast ablations |
| **TTMR** | Target-aware, temporally balanced memory retention with redundancy suppression under fixed token budget | Native Lean TAMP/TCIM against additional ranking with the same memory and prompt settings |

**Design principles:** automatic first-prompt construction; explicit separation of spatial and temporal intervention; preservation of pretrained decoder; controlled memory budgets; causal inference without GT labels.

The contributions are formulated as testable methods. Their mathematical details are given in [FGPI](FGPI.md), [RMFA](RMFA.md), [TTMR](TTMR.md), and the [unified framework](OVERVIEW.md).

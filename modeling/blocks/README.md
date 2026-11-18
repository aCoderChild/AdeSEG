# AdeSEG: scientific block registry

Each block has **one responsibility** and is importable independently.
The public blocks below use stable descriptive names. Historical source
modules remain in `modeling/` because their paths/hashes are part of saved
experiment receipts and weight-loading checks.

| Block folder | Responsibility | Trainable? | Used with |
|---|---|---|---|
| `medsam2_backbone/` | Load the SHA-verified **MedSAM2 Hiera-T/512** pretrained backbone | Frozen | Medical contribution |
| `high_resolution_adapter/` | Add a bounded correction to MedSAM2 decoder's first high-res skip | **Yes, five-layer adapter** | Medical contribution |
| `teacher_preserving_kl/` | Keep a detached native teacher close to the student via KL loss | **No parameters; training-only** | Medical contribution |
| `lean_sam2_backbone/` | Retain the pinned **Lean-SAM2 Hiera-B+** and spatial RoPE corrections | Pretrained frozen | Lean contribution / comparator |
| `multiscale_feature_aggregation/` | Six-stage early/deep Q/K/V feature aggregation and residual mask refinement | **Yes, Lean refiner** | Lean contribution |
| `temporal_memory/` | Multiscale memory-anchor scoring, position lookup and CPU RoPE control | Separate ablation | Lean comparison |

## How the blocks connect

**Medical method (proposed contribution 1):**

```text
Video + initial prompt
        │
        ▼
[MedSAM2 Hiera-T/512 backbone] ───────────────────┐
        │                                         │
        ▼                                         ▼
[High-resolution adapter]                    [Native teacher]
        │                                         │
        ▼                                         │
[Original decoder] ─── segmentation logits ──────┤
        │                                         │
        └──── TRAIN ONLY ──► [Teacher-preserving KL]
                                 + mask loss (GT)
```

**Lean method (proposed contribution 2):**

```text
Video + initial prompt
        │
        ▼
[Lean-SAM2 Hiera-B+ backbone] ── early + deep + mask logits
        │
        ▼
[Multi-scale feature aggregation]
        │
        ├── feature projection / spatial context
        ├── region-conditioned channel attention Q/K/V
        ├── feature fusion / foreground-background contrast
        └── residual mask decoder
        │
        ▼
Output masks / optional, separately controlled memory routing
```

The historical **0.742071** 21-video result is a *separate corrected-Lean
baseline*, not the learned aggregation model or medical KL adapter.

## Reproducibility boundaries

- The teacher KL is a **loss**, not an inference-time neural feature layer.
- `high_resolution_adapter/` is a reconstruction whose original historical
  forward code is missing; checkpoint **shapes** match the recovered weights,
  but numerical historical parity is not established.
- `multiscale_feature_aggregation/` preserves original class hierarchy,
  parameter names, and source-bound checkpoint validation.
- Never interchange MedSAM2 Hiera-T weights with Lean-SAM2 Hiera-B+.
- Keep source-specific prompts, datasets, checksums and evaluation frames
  together with reported scores.

# Experiment registry

The workstreams are separated so that a reported gain can be traced to **one architectural change**, a specific prompt budget, sequence/frame IDs, checkpoint and evaluation protocol.

| Workstream | Research question |
|---|---|
| **[Contribution 1: MedSAM2 teacher-preserving high-resolution refinement](medical_high_resolution_adapter/README.md)** | Can early/deep decoder-skip fusion retain correct native predictions under a training-only teacher loss? |
| **[Contribution 2: Lean-SAM2 multi-stage channel-value aggregation](multistage_feature_aggregation/README.md)** | Can trainable cross-channel value mixing improve masks or memory quality? |
| [Reference models](reference_models/README.md) | How does unchanged Lean-SAM2 compare with SAM2, SAM2Long, SAMURAI and CMSA-Net? |
| [Modified Lean-SAM2 (0.7421)](modified_lean_742/README.md) | What exactly produced the verified 21-sequence automatic-prompt result? |
| [Teacher preservation](teacher_preservation/README.md) | Can training preserve confident positive teacher predictions without test-time supervision? |
| [Lean memory and positional ablations](lean_memory_ablations/README.md) | Do memory-anchor pooling or positional RoPE corrections change performance? |

The full original experiment history and saved source-hash receipts remain on the backup branch and in the Drive experiment outputs. Scores from GT-initialized and automatic-initialized videos should **never** be represented as a matched model comparison.

# Teacher-preservation training objectives

Use `modeling/teacher_preservation.py` for the **training-only** positive-teacher consistency loss, one-sided foreground preservation loss, and deterministic training-logit expansion.

The scientifically named training script `training/train_feature_aggregator.py` accepts `--teacher-weight`. This is a **new combination** of the learned channel-value aggregator and a prior preservation objective; the archived teacher studies have not been reproduced by the new runner.

Historical preservation protocol: `protocols/SOURCE_GROUP_PRESERVATION_V1.md`. No inference-time GT, inference-time label-derived gate or clinical claim is introduced.

This older positive-preservation objective is **not** the same as the native MedSAM2 Bernoulli teacher-KL loss (`adeseg.losses.teacher_kl`). Both are training-only; no GT-derived inference gating is permitted.

# Teacher-preservation training objectives

Use `modeling/teacher_preservation.py` for the **training-only** positive-teacher consistency loss, one-sided foreground preservation loss, and deterministic training-logit expansion.

The scientifically named training script `training/train_feature_aggregator.py` accepts `--teacher-weight`. This is a **new combination** of the learned channel-value aggregator and a prior preservation objective; the archived teacher studies have not been reproduced by the new runner.

Historical preservation protocol: `protocols/SOURCE_GROUP_PRESERVATION_V1.md`. No inference-time GT, inference-time label-derived gate or clinical claim is introduced.

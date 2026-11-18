# Research documentation

- [Medical adapter architecture](MEDICAL_HIGHRES_ARCHITECTURE.md): learned
  decoder-skip correction with training-only native-teacher KL.
- [Lean feature aggregation](MODEL_ARCHITECTURE.md): six-step channel-value
  aggregation and residual-mask refinement.
- [Checkpoint provenance](CHECKPOINT_PROVENANCE.md): MedSAM2 Hiera-T/512
  versus Lean-SAM2 Hiera-B+; identities and historical limits.
- [Experiments](../experiments/README.md): four separate research questions.
- [Code map](../adeseg/README.md): short, explicit package APIs.

Historical filenames containing `_v1` record **old experiment versions**;
legacy `audit_*` scripts are independent **result validators**. Both are
retained only where modifying their names would break provenance. For new
work, use `adeseg.*` imports and `scripts/validate_*.py` commands.

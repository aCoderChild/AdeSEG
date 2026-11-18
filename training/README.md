# Offline feature aggregation training

`python -m training.train_feature_aggregator --help`

Requires an explicitly training-only JSON manifest listing `{"split":"train","tensor_file":"relative-file.pt"}` entries. Each feature file stores `early`, `deep`, `logits` and `target` float tensors from a source-verified TRAIN capture; shape assumptions are enforced. Training never opens test labels. Source and checkpoint SHA256 arguments are required.

The default objective combines boundary-weighted BCE/IoU and soft Dice; optional `--teacher-weight` adds one-sided positive teacher preservation. The script does **not** generate raw video captures, prove model efficacy, or bypass verifier-owned inference checks.

The public `feature_aggregation` module aliases the original model class; checkpoint `source_sha256` is computed from `modeling/multiscale_channel_value_refiner_v1.py` so the existing trained-state verification and video routing type checks stay aligned.

The new trainer instantiates `MultistageFeatureAggregator`, a six-block subclass of the original refiner. Its `state_dict` keys and strict source-hash loader remain bound to `modeling/multiscale_channel_value_refiner_v1.py`. The tensor features must match the actual selected backbone (SAM2.1 Hiera-B+ vs MedSAM2 Hiera-T); do not silently exchange checkpoints. See [checkpoint provenance](../docs/CHECKPOINT_PROVENANCE.md).

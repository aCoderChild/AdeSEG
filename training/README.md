# Training — two independent research contributions

## Contribution 1: MedSAM2 Hiera-T/512 medical adapter

`python -m training.train_medical_highres_adapter --help`

This new trainer consumes frozen TRAIN-only mask-decoder captures produced by a separately verified original MedSAM2 predictor. A manifest is a JSON list such as:

```json
[{"split":"train","sequence":"seq2","file":"captures/seq2_frame0.pt"}]
```

Every `.pt` capture contains exactly `image_embeddings`, `image_pe`, `sparse_prompt_embeddings`, `dense_prompt_embeddings`, `high_res_0`, `high_res_1`, and `target`. The target is for training only. The native pretrained decoder is frozen; the adapter optimizes against BCE, soft Dice and detached native-teacher KL (default weight 0.1). The training script does **not** produce decoder captures or reproduce historical four-frame video training. Strict `MedSAM2_latest.pt` SHA256 checking is mandatory. See [medical architecture](../docs/MEDICAL_HIGHRES_ARCHITECTURE.md).

## Contribution 2: Lean-SAM2 feature aggregation

`python -m training.train_feature_aggregator --help`

Requires an explicitly training-only JSON manifest listing `{"split":"train","tensor_file":"relative-file.pt"}` entries. Each feature file stores `early`, `deep`, `logits` and `target` float tensors from a source-verified TRAIN capture; shape assumptions are enforced. Training never opens test labels. Source and checkpoint SHA256 arguments are required.

The default objective combines boundary-weighted BCE/IoU and soft Dice; optional `--teacher-weight` adds one-sided positive teacher preservation. The script does **not** generate raw video captures, prove model efficacy, or bypass verifier-owned inference checks.

The public `feature_aggregation` module aliases the original model class; checkpoint `source_sha256` is computed from `modeling/multiscale_channel_value_refiner_v1.py` so the existing trained-state verification and video routing type checks stay aligned.

The new trainer instantiates `MultistageFeatureAggregator`, a six-block subclass of the original refiner. Its `state_dict` keys and strict source-hash loader remain bound to `modeling/multiscale_channel_value_refiner_v1.py`. The tensor features must match the actual selected backbone (SAM2.1 Hiera-B+ vs MedSAM2 Hiera-T); do not silently exchange checkpoints. See [checkpoint provenance](../docs/CHECKPOINT_PROVENANCE.md).

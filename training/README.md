# Offline feature aggregation training

`python -m training.train_feature_aggregator --help`

Requires an explicitly training-only JSON manifest listing `{"split":"train","tensor_file":"relative-file.pt"}` entries. Each feature file stores `early`, `deep`, `logits` and `target` float tensors from a source-verified TRAIN capture; shape assumptions are enforced. Training never opens test labels. Source and checkpoint SHA256 arguments are required.

The default objective combines boundary-weighted BCE/IoU and soft Dice; optional `--teacher-weight` adds one-sided positive teacher preservation. The script does **not** generate raw video captures, prove model efficacy, or bypass verifier-owned inference checks.

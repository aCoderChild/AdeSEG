# Offline channel-value aggregation training

The retained `scripts/train_aggregation.py` trains `MultiscaleChannelValueRefinerV1` on frozen precomputed feature captures; **this is not a raw-video training pipeline**. Each JSON manifest entry must be `{"split":"train","tensor_file":"relative/path.pt"}`. Each .pt must be a dict of float tensors named `early` [1,32,H,W], `deep` [1,256,h,w], `logits` [1,1,H,W], `target` [1,1,H,W]. The native logits and features must come from the same frozen, provenance-tracked prediction; `target` is an annotated **training-only** mask. Batch size is one. No validation/test labels are read.

Example: `python scripts/train_aggregation.py --manifest /path/to/train.json --output /path/to/adapter.pt --backbone-sha256 <64-hex> --feature-pipeline-sha256 <64-hex> --mode foreground_background_asinh --teacher-weight 1`.

The trainer writes an original-model-compatible checkpoint payload. Do not treat it as evaluated for inference: output-vs-memory routing requires the original runtime checks, a matching Lean checkpoint, real-video validation and separate held-out scoring. Preserve source hashes and do not modify the original archived scripts if reproducing a published or frozen run.

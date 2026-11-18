# Complete historical experiment archive

The original `experiment` branch is backed up **unchanged**:

- [backup/experiment-full-20261118](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118), commit `21c598266bd6f6fec2b2181f2a03d92691d20405`
- [backup/main-before-cleanup-20261118](https://github.com/aCoderChild/AdeSEG/tree/backup/main-before-cleanup-20261118)

The full original experiment includes the 1,055 historical scripts, 393 protocols, comprehensive tests and source-hash contracts that are not all present in the focused current working tree. Restore its exact layout to reproduce any saved checkpoint/run. Do not delete either backup while the desired ~0.742 modified-Lean score or full video inference have not been verified.

**Important gap:** the high-resolution MedSAM2 teacher adapter and original video runner named in Drive are not in this backup snapshot. The training-only teacher loss and the original MedSAM2 vendor predictor are present, but the separate high-res adapter must be recovered from the user's local source or Drive before claiming runnable preservation.

The focused `main` does **not** include `protocols/` or unsupported automatic-memory experimental scripts; all original files are retained in this backup. The retained author-model runner has a new minimal setting under `experiments/reference_baselines/checkpoint.json`, so new output checksums and provenance receipts will differ from the old experiment.

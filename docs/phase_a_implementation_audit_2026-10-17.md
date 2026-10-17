# Phase A implementation audit — 2026-10-17

This records deviations discovered while Kaggle script version `359186175` was
still running. That version is immutable and remains an exploratory pilot; code
changes made after its launch do not change its measurements. Do not use its
summary for the preregistered hypothesis go/no-go decision.

## Verified recording path

- `generate_utility_labels.py` atomically saves `utility_records.pt`,
  `frames.jsonl`, and `status.json` after each completed sequence-object.
- The Kaggle runner writes root `status.json` and append-only `events.jsonl`.
  Kaggle does not expose the working-directory files through `kernels files`
  while the notebook is running. Export the completed or failed version before
  claiming Drive delivery.
- Script version `359186175` passed `--wandb-mode disabled`; its “W&B
  initialized” message was misleading and does not signify a live run. The
  next runner version selects online tracking when the Kaggle secret and Forge
  connection check work, otherwise offline tracking for later sync. A saved
  version can also be backfilled with provenance marked `not-live`.
- The backfill artifact now includes the feature/label tensor file, not just
  JSON summaries. Check completion status and file counts after export.

## Deviations from frozen protocol v2

1. Counterfactual utility is computed from hard-mask Dice, while v2 specifies
   soft Dice loss as the primary utility and hard Dice as a separate diagnostic.
2. Attention saliency averages over every query and over layers. V2 requires
   predicted-foreground queries and per-layer measurements.
3. The reported cached-state-restore error duplicates the no-op rerun error;
   it is not an independent restore check. Full DAVIS validation J&F reference
   reproduction is not established by this run.
4. The E3 runner samples `c0`, `c2`, `c4`, and natural memory only. Its analysis
   does not perform the preregistered c0–c4 sequence-random-effect trend test.
   Accordingly, `E3_pass` is now null in later versions of this analysis script.
5. The c4 false-positive blob is a geometric proxy, not a blob placed on the
   most similar distractor. The image encoder cache retains the latest frame,
   not an entire sequence as v2 specifies.
6. This Phase A runner does not execute the full E2 oracle-bank and horizon
   comparisons, E4 held-out linear-probe controls, E5 causal readout
   comparisons, or E7 PolypGen transfer. It cannot establish C0–C5.

Do not modify the frozen v2 protocol to conceal these differences. Correct and
version the implementation, rerun sanity checks and test controls, then use
only the corrected experiment for the preregistered decision.

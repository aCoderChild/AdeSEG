# Full-propagation false-alarm follow-up

Status: **implemented, not yet run**. This is a new exploratory propagation
experiment, separate from the completed fixed-bank exclusions. No architecture,
training, detector or extra prompt is added.

## Protection before inference

The mounted source is now `outputs/experiment`. A verified snapshot copy is
being attempted under `AdeSEG/backups/experiment_snapshot_2026-10-24_v2` on the
same Drive account. Regenerable ordered-frame staging and W&B upload caches are
excluded, not scientific evidence. Copy completion requires a SHA256 manifest;
an incomplete destination is not a backup. This is not an independent device
or account backup. Google Drive web upload remains unverified.

`protection_receipt.json` is written only after a successful copy comparison.
It initially has `drive_web_verified=false`. An observed folder URL and actual
web verification must be recorded before the propagation runner will execute.
Do not flip this flag to bypass a missing check. User-facing cloud verification
requires working browser control or access to the actual Drive folder.

## Fixed implementation

- Reuse pinned author SAM2 and SAMURAI predictors and their native writes.
- Previous processed-frame score≤0 masks only stored foreground-interior keys.
- Positive score restores the next read; background, boundary and pointers stay.
- Native256 mixed cells use the existing Efficient-SAM2 occupancy integration.
- Every output propagates through normal memory writing; no fixed-bank replay.
- No-op parity and repeated fresh-model replay are strict hash checks.
- GT is used only for the one initial prompt and offline evaluation.

References, source revisions, metrics and stopping conditions are in the
[frozen protocol](../protocols/full_propagation_false_alarm_v1.json).
No replacement author attention, encoder, decoder or memory writer was written.
The new score-to-mask policy and workflow adapters are project-specific glue.

## Ordered execution after protection passes

All output directories must be new. Set the following to verified mounted paths;
the receipt must not change between the gate and full runs.

```sh
set -e
TASK_ROOT='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG'
TASK_CHECKPOINT="$TASK_ROOT/outputs/experiment/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt"
TASK_RUN="$TASK_ROOT/outputs/experiment/19_stale_foreground_propagation_v1"
TASK_RECEIPT="$TASK_ROOT/backups/experiment_snapshot_2026-10-24_v2/protection_receipt.json"
```

Four validation conditions, each on **complete seq4/seq9**, repeated with fresh
model/state. These sequences include GT-empty runs and returns; seq2/seq3 do not.

```sh
for TASK_CONDITION in sam2 samurai; do
  .venv/bin/python scripts/run_stale_foreground.py --condition "$TASK_CONDITION" --smoke --checkpoint "$TASK_CHECKPOINT" --protection-receipt "$TASK_RECEIPT" --output-dir "$TASK_RUN/validation/$TASK_CONDITION"
done
.venv/bin/python scripts/run_stale_foreground.py --condition samurai_noop --smoke --checkpoint "$TASK_CHECKPOINT" --protection-receipt "$TASK_RECEIPT" --native-reference-dir "$TASK_RUN/validation/samurai" --output-dir "$TASK_RUN/validation/samurai_noop"
.venv/bin/python scripts/run_stale_foreground.py --condition samurai_fg_gate --smoke --checkpoint "$TASK_CHECKPOINT" --protection-receipt "$TASK_RECEIPT" --output-dir "$TASK_RUN/validation/samurai_fg_gate"
```

Use a shell with fail-on-error enabled (`set -e`) for sequential execution.
Any failed command stops the workflow; do not run the following full cohort.
The runner additionally requires all four verified, delivered gates before any
full condition and rejects changed sources, protection receipt or identities.

```sh
for TASK_CONDITION in sam2 samurai samurai_fg_gate; do
  .venv/bin/python scripts/run_stale_foreground.py --condition "$TASK_CONDITION" --checkpoint "$TASK_CHECKPOINT" --protection-receipt "$TASK_RECEIPT" --validation-dir "$TASK_RUN/validation/$TASK_CONDITION" --gate-suite-dir "$TASK_RUN/validation" --output-dir "$TASK_RUN/full/$TASK_CONDITION"
done
```

`frames.jsonl`, `foreground_gate.jsonl`, per-sequence `false_alarm_summary.json`,
native parameter/replay checks and archives are saved in Drive and tracked
online in W&B. The final gated full run also writes/uploads
`propagation_comparison.json` after checking source-frame/GT agreement across
all three complete methods. Six fixed sequence-level tests use Holm correction.

Return latency is sampled-frame delay to any positive GT/prediction overlap.
Undetected returns retain censor status and follow-up length. Report GT-empty
false alarms, present-frame Dice and return failures/delays together; suppressing
every prediction would not be a successful method. No outcome-driven tuning.

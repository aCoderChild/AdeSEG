> **Superseded goal/work plan:** Follow [ACTIVE_RESEARCH_GOAL.md](ACTIVE_RESEARCH_GOAL.md), updated and resumed on 2026-11-11. The material below preserves the previous scope and decisions.

> **Active amendment — 2026-11-11:** The user authorised testing feature-based
> grouping, gradient-based attribution and alternative matching before further
> ToMe efficacy work. Follow [the staged identification study](TOME_REGION_IDENTIFICATION_STUDY.md).
> Its aim is a more reliable gain than Run155. The historical stop decisions
> below remain valid; new diagnostics do not reopen those failed runs.

# Codex work plan: ToMe memory-token utilization

Follow [the current research goal](MEMORY_ATTENTION_GOAL.md). The active
candidate is training-free, region-constrained local ToMe inside SAM2's spatial
memory attention. Keep the native frame schedule and SAM2 model unchanged; do
not add good/bad frame selection or confidence gates.

## Current state

The position-consistent implementation passed its seq2/seq3 engineering and
persistent-ID gates. Its frozen PolypGen efficacy run stopped after 19 of 21
matched sequences because the predeclared win criterion became unreachable and
the false-alarm condition failed. The apparent Dice gain is sensitive to seq8.
Run155 and DAVIS results are exploratory and do not establish reliable benefit.
No ToMe inference process is active.

Temporal audit: each selected memory frame keeps its native temporal slot and
is merged independently; within-frame feature evidence and spatial detail are
still averaged, so this is not lossless temporal-memory preservation.

## Work order

1. Preserve the source, protocol, gate receipts, and completed results. Record
   source-provenance limits explicitly; do not present unverified runtime
   dependencies as exact reproduction.
2. Do not restart, tune, or extend the stopped candidate on the analyzed
   PolypGen or DAVIS validation cohorts.
3. Start another efficacy study only if a distinct mechanism is supported by
   new evidence, untouched annotated video data are available, and the split,
   controls, token budget, metrics, practical-effect threshold, and stopping
   rules are frozen before inference.
4. Report Dice as primary, with boundary quality, false alarms, re-detection,
   compute, uncertainty, and sensitivity to seq8. Keep exploratory results
   separate from independent confirmation.

The concise method and status are in [`docs/REGION_TOKENS.md`](../docs/REGION_TOKENS.md)
and [`docs/REGION_TOKENS_RESULTS.md`](../docs/REGION_TOKENS_RESULTS.md).

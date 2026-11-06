# Codex Luna: regional memory-reader work plan

Prepared 2026-11-06. Planning only: this file launches no experiments and is not
a frozen efficacy protocol. Work through the stages sequentially.

## Objective and constraints

Test whether one complementary regional reader improves full-video PolypGen Dice
over native SAM2 **and** a parameter/training-matched generic reader. A reliable
negative answer is a valid deliverable; improvement is not guaranteed.

- Freeze every original SAM2 weight; retain native frame selection and pointers.
- No additional confidence gates, tuned decision thresholds, hard token deletion
  or good/bad frame selection. Statistical criteria are not inference gates.
- Reuse `modeling/complementary_memory/` and its pinned Cutie implementation.
  Do not revive failed ToMe variants or start an architectural sweep.
- Save artifacts, temporary data and tracker files in the configured Drive tree.
  A Drive-backed path is not proof of cloud upload; verify upload separately.
- PolypGen and previously examined DAVIS validation are exploratory. Sequence
  separation does not establish patient independence.

## Starting point

Run 165 records 509 passing tests, exact zero-residual replay on 75 outputs per
arm, learning gradients and a two-object isolation fixture. **Efficacy is untested.**
Revalidate receipts and source hashes before reusing this evidence.

Preserve the bytes of the recorded [engineering protocol](complementary_memory_v1.json)
and [dataset plan](dataset_evaluation_plan_v1.json); use new versioned protocols
for new work.

## Sequential stages

| Step | Work and deliverable | Required before advancing |
|---|---|---|
| 0. Establish state | Inspect instructions, Git changes, processes, Drive artifacts and receipts; record checkpoint/source hashes and available data. | No duplicate job; compatible evidence identified; unrelated work preserved. |
| 1. Freeze diagnostic | Write exact sequence splits, sample selection, supervision target, probe, fitting policy, seed and metrics. | Commit before computing diagnostic outcomes. |
| 2. Run diagnostic | Compare mask-only, memory-feature-only and combined probes on identical held-out sequences/positions; save per-sequence results and error cases. | Verified alignment and coverage; evidence that features add useful information on stored-mask errors. Otherwise stop for review. |
| 3. Freeze pilot | Write one executable training/evaluation recipe, practical-effect criteria, seeds, controls and stopping rules. | Commit before training; no unresolved metric, split or checkpoint-selection decision. |
| 4. Verify runner | Check autoregressive training, native parity, matched arms, frozen weights, instance isolation and safe resume on real frames. | All relevant tests/checks pass with saved receipts. |
| 5. Run pilot | Native reference, generic and regional full propagation; one fixed seed first, then pre-specified repeat seeds if the screen passes. | Complete coverage and integrity; no result-driven settings changes. |
| 6. Decide and report | Per-sequence/aggregate tables, paired uncertainty, cost and failures; update the two research documents. | Apply the frozen rule. Stop this candidate if it fails; distinguish negative from inconclusive evidence. |
| 7. Conditional transfer | Freeze the PolypGen model; verify DAVIS/LVOS data, instance handling and official evaluators, then evaluate all three arms. | Primary evidence warrants transfer; dataset-specific checks pass. |

### Diagnostic safeguards

Use an existing documented probe/library, not a new learning algorithm. Match the
probe family, samples, regularization policy and training allowance; disclose
different input dimensions. Fit normalization and model selection on training/
validation sequences only.

Use GT targets only for supervision and scoring. Predicting the same mask-derived
labels supplied as inputs is circular. Analyze both correct and incorrect stored
mask positions, retaining the full sampled population in the main result.
Feature-only memory is already mask-conditioned by SAM2; it is not image-only.
Probe accuracy measures decodable information, not causal utility or Dice gain.

Audit archived sample selection. Outcome-selected snapshots are debugging evidence,
not an unbiased evaluation. Predefine representative sampling if needed. Treat
sequences as units, not thousands of independent tokens. Freeze the diagnostic's
continuation criterion before outcomes; inconclusive evidence requires review,
not automatic authorization of a larger experiment.

### Pilot freeze checklist

- [ ] Exact train/validation/evaluation sequence IDs; exclude seq1/seq7. No sequence
  spans split roles. State which sequences contribute to each reported mean.
- [ ] Same native checkpoint, prompts, geometry, precision and scored frames.
  Evaluation GT after initialization is used only for scoring.
- [ ] Generic/regional arms have the same initialization seed, parameter/summary
  count, optimizer, schedule, training samples and checkpoint-selection policy.
  Generic uses the existing constant-0.5 external cue.
- [ ] Each arm propagates its own predictions into its own memory bank. Fixed
  native-bank replay is diagnostic, not the final efficacy experiment.
- [ ] Equal-sequence present-frame Dice is primary. Specify prompt handling,
  empty-frame conventions, paired sequence bootstrap, paired tests and multiplicity.
- [ ] Report per-sequence differences, wins/ties/losses, leave-one-sequence-out
  sensitivity, IoU, Boundary F, false alarms/denominators, re-detection and cost.
  Fix metric implementations and definitions before outcomes.
- [ ] Set numeric practical-effect and tolerated-harm criteria prospectively.
  Suggested discussion targets: +0.02 absolute Dice versus native and +0.01 versus
  generic. These are proposed research targets, not established clinical relevance
  or already-frozen pass criteria. Finalize them in the pilot protocol.
- [ ] Fix first-seed continuation rule, repeat-seed list and compute budget.
  One seed is a feasibility screen only. Include all planned seeds; never select
  the best seed. Sequence, not frame/token/seed, remains the statistical unit.
- [ ] Final positive interpretation requires the frozen practical and uncertainty
  criteria against both controls, within agreed harm limits.

## Execution and tracking

For every stage save setup, logs, status, metrics and a completion receipt on
Drive. Record run ID, protocol commit, source/checkpoint hashes, exact command,
PID, timestamps, expected/completed units, exit code and artifact paths. Write
status atomically. Resume only with compatible hashes/settings and validated
completed units; preserve failed attempts with their validity clearly marked.

On each continuation inspect the actual process and recent output progress before
saying a job is running. Report an evidence timestamp and completed/expected units.
Estimate remaining time from measured throughput, with uncertainty. Do not claim
unattended monitoring unless a monitor actually exists.

If a check fails, stop dependent runs, diagnose it, add a targeted test, document
the fix and invalidate affected results. Re-run affected checks before resuming.
Unexpected performance alone is not a software bug. Changes to frozen methods
require a dated amendment and a new run; never silently overwrite provenance.

Keep this ledger current. Keep `docs/REGION_TOKENS.md` short and methodological;
put verified completed tables and brief analysis in `docs/REGION_TOKENS_RESULTS.md`.
Do not turn either document into a training log. Commit only reviewed task-related
changes; never blanket-stage the existing dirty worktree.

## Progress ledger

Update each state with a date and evidence link. Initial entries reflect project
records, not a fresh process or hash verification.

| Item | State | Evidence / next action |
|---|---|---|
| Run 165 engineering gate | Recorded complete | Drive `165_complementary_memory_v1/gate_retry1/`; revalidate at Step 0 |
| Current-state inventory | Complete (2026-11-06) | No experiment process active; Run 165 receipt passed, `pilot_authorized_by_this_receipt=false`; all 228 snapshotted sources match the worktree; checkpoint hash matches; Run 143 bank receipts verify all 21 sequences / 2,012 post-prompt frames. seq13 bank is referenced from `gate/observed/seq13`. |
| Diagnostic protocol | Frozen before diagnostic outcomes | [Complementary memory diagnostic v1](complementary_memory_diagnostic_v1.json); use exact committed protocol and read-only Run 143 banks. |
| Diagnostic results | Not started | Step 2; test sequences were examined in prior studies, so result remains exploratory. |
| Pilot protocol | Pending | Step 3 |
| Pilot runner verification | Pending | Step 4 |
| First-seed full propagation | Not started | Conditional on Steps 1–4 |
| Repeated seeds | Not started | Conditional on frozen screen |
| Primary decision/report | Pending | Step 6 |
| DAVIS/LVOS transfer | Not started | Conditional on evidence and dataset checks |

## Handoff instruction

Complete one small reviewable task at a time; record evidence and the next action.
Ask only when a missing decision blocks correctness. Never invent scores, bypass
checks or substitute a method mid-run. If the diagnostic or pilot fails its frozen
criteria, report the finding and a justified next proposal. Do not automatically
start another architecture.

# Region-token experiment

Active branch: `experiment`. Effective approved preregistration: `protocols/region_tokens_v1_1.json` plus the
pre-E2 CholecSeg8k data-definition amendment `protocols/region_tokens_v1_2_cholec_amendment.json`;
the original `protocols/region_tokens_v1.json` is retained unchanged.
This study tests region-level memory contributions; it does not reuse the old
latent-capability conclusions as evidence.

## Entry points

- `datasets/io.py`: existing shared mask decoding moved out of the removed clinical `adenoid` package, without changing behavior.
- `modeling/region_tokens.py`: category maps, matched sampling, attention aggregation and native persistent-propagation wrapper.
- `modeling/utility_memory.py`: existing additive attention hook, extended to token masks/biases.
- `scripts/run_region_token_pilot.py`: independently restored counterfactuals, E0 and E1.
- `scripts/analyze_region_token_pilot.py`: sequence-level pilot decision.
- `kaggle/region-tokens/region_tokens.ipynb`: pinned GitHub revision bootstrap.
- `kaggle/region-tokens/runner.py`: credentials, data checks, E0 → E1 orchestration.
- `scripts/collect_region_results.py`: terminal-status polling, direct Drive export and checksum receipt.

Read the workflow in this order: `runner.py` → `run_region_token_pilot.py` →
`analyze_region_token_pilot.py` → the two collectors. The model stays in the
pinned upstream repositories; project code supplies experiment controls and
recording, not a replacement SAM2 implementation.

All three `modeling` files are needed: `medsam2.py` loads the existing upstream
predictor; `utility_memory.py` is the reused intervention hook; `region_tokens.py`
adds the study's taxonomy, matched sampling and aggregation. `training/tracking.py`
is required even for frozen inference. `training/utility_trainer.py` remains only
because the design explicitly reuses it for the optional learned-method arm.
Frozen protocol ancestors are retained for their split/provenance references;
the authorized amendment replaces the redundant proposal file.

The notebook requires the **attached** Kaggle secret `WANDB_API_KEY`. It checks
Forge access before inference. It creates online orchestrator/E0/E1 runs in
`adenoid-hypertrophy/ade-seg`; raw results are uploaded as versioned artifacts.
Kaggle retains outputs under `/kaggle/working/region_tokens_v1`. The Mac exports
these directly to the existing Drive directory, without a second local results
directory. The local Drive client must sync them to the cloud. Kaggle working
storage is remote scratch, not the user's local disk. Drive delivery occurs
through live W&B artifact collection; a separate terminal export verifies the
saved Kaggle version. W&B provides live progress.

```sh
.venv/bin/python scripts/collect_region_results.py --wait --output-root \
  '/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026'
```

Launch the pinned notebook from the authenticated Kaggle editor with its secret
attached. API-pushed versions have previously failed to receive the secret;
do not launch a duplicate GPU job to work around an interactive queue.

## Scientific status

Implemented: E0/E1 pilot, all four categories, two banks, five uniform and
distance-matched random controls, pointer/spatial bounds, per-layer/query-region
saliency, Dice/IoU/boundary F/HD95, low-resolution output and logit changes.
E1 samples up to ten fixed frame indices per dev sequence (98 targets across
the ten amended pilot sequences). Categories with
fewer than k available tokens are skipped and counted. The distance-matched
control can overlap the reference; overlap is reported.

**Not yet implemented/validated:** E2–E6 orchestration, GPU validation of persistent propagation, value-only
substitution and cross-sequence swap, CholecSeg8k adapter and source-video split,
RAFT condition analysis, complete external-baseline adapters, and method tuning.
An E1 pass permits developing/running those stages; it does not make this pilot
a full study. An E0 failure stops inference; an E1 failure stops/pivots the study.

PolypGen mirror equivalence to Synapse remains unverified. The runner records
per-file checksums, names and counts and explicitly retains that limitation.
The existing illumination helper uses luminance, not the HSV rule described in
the supplied design; the preregistration records the actual reused detector.

The MA-SAM2 submodule is pinned to its public repository, which currently ships
one predictor source file. It is not yet a validated runnable baseline. Existing
SAM2Long/Lean-SAM2 repositories remain the source for their adapters.

Historical v2 protocols and the exploratory Phase A report retain their original
provenance. Old development notebooks are not the active region-study workflow.

## Current deployment

The approved-amendment rerun completed on T4 x2 at revision `ef1195b39352`,
with online orchestrator `7orfgsk2` and group `region-tokens-v1-ef1195b39352`.
E0 run `oo2f3bjc` passed on seq2 frame 7: detached/empty/zero-bias no-ops,
independent mutation and restoration, full-token/frame equivalence, and attention
reconstruction (maximum absolute error 0.0). E1 run `wg2nrbsd` completed all ten
promptable sequences (2,744 rows; 2,463 feasible and 281 explicit skips). The
reviewed sequence-level gate is `proceed_to_E2`: masking recent boundary tokens
caused mean Dice harm 0.0322 (95% CI 0.0044–0.0695) and boundary-F harm 0.0413
(95% CI 0.0167–0.0727), beyond every uniform-random seed mean. This only gates
E2; it does not confirm H1–H4 or mark the full study complete. The Drive delivery
receipt is valid, and the collector has verified artifact checksums.

The next revision adds the E2 value-only path. It preserves keys, RoPE positions,
and pointers while replacing selected spatial values; an identity replacement is
now an E0-required bit-exact check before any E2 inference is counted. The first
E2 job is a dev-only calibration pass: it produces category means and validates
that exact value replacement is bit-identical. It is not an E2 result.

That calibration completed on Kaggle T4 x2 at revision `07e1b199dc77` with
online W&B runs `cyls6h3w` (orchestrator), `5udert26` (E0), and `o2cischg`
(calibration). The E0 artifact passed all seven checks, including value identity
exactness. The c0 dev means are finite 64-dimensional vectors for boundary
(129,240 tokens), foreground (159,556), near-background (163,681), and
far-background (2,357,379). The final artifact checksum-verified delivery is
in `region_tokens_e2_calibration_v1`; this validates the substitution operator
but makes no H1--H4 claim.

Before any CholecSeg8k E2 outcome is generated, its semantic-mask object policy
and source-video method split are frozen in the v1.2 data-definition amendment.
Every non-background semantic class is an independently prompted clip-class unit;
results will retain the source video and never be presented as instance tracking.

The collector verifies every downloaded file against the W&B manifest MD5,
removing only a mismatched downloaded copy before refetching. This avoids the
SDK's cached-checksum shortcut retaining a stale same-length tracking file.

Kaggle notebook: `bunnoob2005/adeseg-region-memory-tokens`. Version 1 failed
at Kaggle secret retrieval before creating a W&B run. The secret was verified
attached in the editor. A CPU-only interactive execution of the existing
`verify_wandb_connection.py --check-only` printed `connected` for the Forge
project. Version 2 was cancelled after 14 minutes queued; an interactive T4
session then executed the bootstrap at revision `12919339aa4e` with live W&B
orchestrator `w3f2tstp` and E0 run `s3owx6fo`. E0 stopped at prompt validation:
seq1's first mask is empty. No instrumentation check or hypothesis metric ran.
Kaggle's available accelerator selector offered T4 x2 and TPU; P100 was not
available. No duplicate GPU job was launched to bypass that queue.

`scripts/collect_region_live.py --watch --group region-tokens-v1-12919339aa4e
--output-root <Drive output directory>` delivers stage artifacts while the
interactive session runs. Artifact downloads use `skip_cache=True`; SDK staging
paths also point into Drive. The resulting receipt distinguishes W&B artifact
delivery from a saved Kaggle notebook version export.

The collector recomputes E1 with the reviewed analysis code (matching feasible
frames before averaging), retains the original notebook decision, uploads
`decision_verified.json` to the same W&B orchestrator run, and records both
analysis and inference revisions. A passing pilot never sets
`full_study_complete` to true.

## Initialization blocker discovered on Kaggle

The subsequent audit of all 23 sequences found no labeled foreground anywhere
in seq1 (36 frames) or seq7 (48 frames). Several other sequences have delayed
first labels. Only eight of the fifteen development sequences have foreground
at frame zero, so ten first-frame-prompted dev sequences cannot be drawn from
this mirror under the current rule.

The user approved the documented amendment before any token-intervention
outcomes. Effective protocol `protocols/region_tokens_v1_1.json` initializes at
the first labeled frame, selects seq2–6 and seq8–12, and reports unpromptable
sequences separately. Targets start at prompt offset + 7 for every method.
The original `region_tokens_v1.json` remains unchanged for provenance.

Kaggle version 3 was Quick Saved with **Always save output** enabled. Its
initialization-audit JSON downloaded from Kaggle was byte-identical to the file
downloaded from the W&B run. The idle interactive GPU session was stopped while
the initialization decision was pending. The amended rerun is now authorized.
E2–E6 remain incomplete, and the local
legacy branch/runs have not been removed because the requested successful-GPU
validation condition has not been met.

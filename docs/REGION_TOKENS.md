# Region-token experiment

Active branch: `experiment`. Effective approved preregistration: `protocols/region_tokens_v1_1.json` plus the
pre-E2 CholecSeg8k data-definition amendment `protocols/region_tokens_v1_2_cholec_amendment.json`
and the approved actual-source split correction `protocols/region_tokens_v1_3_cholec_split.json`;
the original `protocols/region_tokens_v1.json` is retained unchanged.
This study tests region-level memory contributions; it does not reuse the old
latent-capability conclusions as evidence.

## Review corrections (2026-10-18)

`protocols/region_tokens_review_v1.json` records the post-interim-review
amendment. Existing results and the running pinned Kaggle job are unchanged.
This is not a retrospective claim of outcome-blind preregistration.

The registered default remains mixed-coverage boundaries with pooled-distance
random controls. E2/E3 can additionally run independent sensitivity shards:

- `--boundary-definition contour_union` includes geometric grid-aligned edges.
- `--spatial-match frame_distance` preserves each memory frame's removed-token
  count and integer-distance histogram, controlling temporal composition too.

The Kaggle `REGION_JOB` JSON accepts the same `boundary_definition` and
`spatial_match` keys. Use a separate output folder for each variant. A fresh E0
is still mandatory. Variants cannot be pooled in analysis or select registered
E6 weights; optional variants are not automatically submitted by this change.

E5's existing `agreement` is explicitly labeled whole-object boundary-F harm;
`query_specific_agreement` instead uses signed Dice harm in the same query
region. Missing local effects remain unavailable. E3/E4 retain the registered
mixed model and add a separate statsmodels sequence-clustered OLS sensitivity,
including small-sample correction, cluster-count warnings and its own Holm
family. A sensitivity p-value never replaces the registered primary result.
Analysis JSON also reports feasible/skipped rows by cluster, bank, category
and measured size stratum; size is `unmeasured` without condition data.

Cholec method development uses video01, video09, video12, video17, video18,
video20, video24, video25, video26, video27, video28 and video35. Held-out method
testing uses video37, video43, video48, video52 and video55. All clips/classes
from a source video remain together. The previous consecutive-ID split is
retained only as superseded provenance, not used for new tuning or testing.

## Entry points

- `datasets/io.py`: existing shared mask decoding moved out of the removed clinical `adenoid` package, without changing behavior.
- `modeling/region_tokens.py`: category maps, matched sampling, attention aggregation and native persistent-propagation wrapper.
- `modeling/utility_memory.py`: existing additive attention hook, extended to token masks/biases.
- `scripts/run_region_token_pilot.py`: independently restored counterfactuals, E0 and E1.
- `scripts/analyze_region_token_pilot.py`: sequence-level pilot decision.
- `kaggle/region-tokens/region_tokens.ipynb`: pinned GitHub revision bootstrap.
- `scripts/run_region_experiments.py`: gated sequence-object shards for full-target E2, E3, calibration and persistent E6.
- `scripts/measure_region_conditions.py`: E4 contrast, existing luminance/specular measurements and torchvision RAFT-small motion.
- `scripts/analyze_region_experiments.py`: matched-frame E2 comparisons, E3/E4 mixed models, E5 sequence-bootstrap agreement, E6 paired comparisons and dev-only method freezing.
- `scripts/run_region_baseline.py`: isolated-process adapters to upstream SAM2Long and Lean-SAM2 TAMP, without reimplementing their models.
- `kaggle/region-tokens/runner.py`: credentials, data checks, E0 → E1 and explicit study-shard orchestration.
- `scripts/collect_region_results.py`: terminal-status polling, direct Drive export and checksum receipt.

Read the workflow in this order: `runner.py` → `run_region_token_pilot.py` →
`analyze_region_token_pilot.py` → the two collectors. The model stays in the
pinned upstream repositories; project code supplies experiment controls and
recording, not a replacement SAM2 implementation.

All three `modeling` files are needed: `medsam2.py` loads the existing upstream
predictor; `utility_memory.py` is the reused intervention hook; `region_tokens.py`
adds the study's taxonomy, matched sampling and aggregation. `training/tracking.py`
is required even for frozen inference. The old optional utility training adapter
has been removed; the active E0–E6 implementation is training-free. The optional
learned follow-up mentioned in the frozen execution supplement is not available.
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

`kaggle/region-tokens/tracking_preflight.ipynb` is the CPU-only saved-version
credential diagnostic. Submit it with no datasets and GPU disabled, not with
the E0 GPU metadata. Version 6 (script version 359528058) started its first cell
but failed when Kaggle's secret service returned HTTP 400. The editor still
shows the existing `WANDB_API_KEY` checked; that does not prove the API-submitted
saved version can retrieve it. Its diagnostic log is in Drive under
`region_tokens_tracking_preflight_v6`. Validate an editor-submitted preflight
before allocating another GPU session. Never embed the key in notebook source.
Editor-submitted Version 7 (script version 359528953) subsequently passed:
secret retrieval succeeded and the existing authenticated project check returned
connected at 11.1 seconds. Kaggle marked it successful (20.7-second run). Its
output, log and checksum receipt are in Drive under
`region_tokens_tracking_preflight_v7`. This proves credential/project access,
not model correctness or a completed W&B experiment run. Version 8 was then
submitted through that same editor path for fresh E0 checks, with PolypGen only.
The default metadata now attaches only PolypGen for E0; add Cholec/DAVIS only
for a stage that needs that dataset.

For standalone E0 terminal export, select its actual output folder; do not apply
the E1 pilot verifier to an E0-only run:

```sh
.venv/bin/python scripts/collect_region_results.py --wait \
  --folder-name region_tokens_execution_e0_v2 --output-root \
  '/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026'
```

## Scientific status

Implemented: E0/E1 pilot, all four categories, two banks, five uniform and
distance-matched random controls, pointer/spatial bounds, per-layer/query-region
saliency, Dice/IoU/boundary F/HD95, low-resolution output and logit changes.
E1 samples up to ten fixed frame indices per dev sequence (98 targets across
the ten amended pilot sequences). Categories with
fewer than k available tokens are skipped and counted. The distance-matched
control can overlap the reference; overlap is reported.

**Implemented for validation, not yet GPU-validated:** the remaining E2 operators
and persistent mode, full-target multiclass Cholec/DAVIS shards, E3 corruption,
E4 RAFT conditions, E5 agreement, upstream SAM2Long/TAMP adapters, and the
training-free E6 tuning/evaluation path. Prospective operational choices are in
`protocols/region_tokens_execution_v1.json`. CPU tests verify token swaps,
corruption resolution, independent rollout states, cached features, no-op bias,
matched statistics and split guards. This is implementation evidence only.
The optional learned E6 head and optional MedSAM2 transfer are not implemented
by this extension; the incomplete MA-SAM2 release is still unavailable.
The E2 calibration and bounded mean-value smoke validate the value-only operator,
but not the full E2 design.
An E1 pass permits developing/running those stages; it does not make this pilot
a full study. An E0 failure stops inference; an E1 failure stops/pivots the study.

## Remaining-stage execution

The existing running notebook remains pinned to its original revision. After
that gate finishes, GPU-validate each new operator and baseline on a dev shard
before the full matrix. Set `REGION_STAGE=study_shard` in a new pinned notebook
execution and supply one explicit JSON `REGION_JOB`, for example:

```json
{"stage":"E3","dataset":"polypgen","split":"test","sequences":["seq2"],"memory_condition":"c1"}
```

`split=test` is the existing PolypGen manifest label; actual dev/test eligibility
uses sequence IDs, not that label. Cholec uses `split=analysis`; DAVIS uses its
frozen `train`, `holdout`, `val` splits. The runner enforces the registered
first-labeled-frame prompt and uses every target from prompt+7 onward. Set
`mode=persistent` for accumulated E2 effects, `intervention=mean_substitution`
or `swap` for substitution controls, and `band`/`thresholds` for the registered
sensitivity settings. Pointer-only and all-spatial references are recorded.
Infeasible categories and missing exact-position donors are explicit skips.

Run `stage=E2_calibration` separately on dev sequences of each dataset, then
reference its completed `calibration_artifact` for mean/swap jobs. Earlier
mean-only smoke artifacts do not include swap donors and are not accepted as
complete calibration. E3 conditions are c0/c1/c2/natural; reuse matching E2 c0
rows rather than concatenate duplicate c0 reruns. E4 condition files join by
dataset, sequence, object and frame. Analysis rejects repeated completed arms.

For E6, prepare the fixed weight/lambda candidates from dev E2 rows:

```sh
python scripts/analyze_region_experiments.py --prepare-method \
  --input /kaggle/working/dev_E2/seq2__object1/frames.jsonl \
  --input /kaggle/working/dev_E2/seq3__object1/frames.jsonl \
  --input /kaggle/working/dev_E2/seq4__object1/frames.jsonl \
  --output /kaggle/working/method_candidates
```

Run each candidate through `stage=E6_dev`. The preparation command publishes the
candidate JSONs to a W&B artifact; reference that `method_artifact` and the chosen
`method_file` in the job. The dev
runner executes only baseline/full arms; test also runs the specified controls
and ablations. Freeze lambda only after every grid candidate completes on the
thirteen promptable PolypGen dev sequences and ten DAVIS holdout sequences:

```sh
python scripts/analyze_region_experiments.py --freeze-method \
  --candidate-dir /kaggle/working/method_candidates \
  --trial-dir /kaggle/working/completed_lambda_trial \
  --output /kaggle/working/method_frozen.json
```

Repeat `--trial-dir` for all sequence/dataset shards of all five candidates.
`E6_test` rejects an unfrozen choice or a different checkpoint. For upstream
comparators use `stage=E6_baseline`, `baseline=SAM2Long` or `Lean-SAM2`, and
`partition=dev`/`test`. Both adapters use the author code in a separate process
to prevent their `sam2` packages from colliding. MA-SAM2 is not replaced by a
guessed SAM2 package alias.

Every child run requires online W&B, records setup/checkpoint/protocol hashes,
and uploads raw JSONL plus status artifacts. Each notebook execution has its
own W&B group. The existing live collector supports `study_shard_complete`,
downloads directly to the requested Drive root and validates component delivery.
Its receipt still distinguishes Drive files from Google cloud-sync status and
saved Kaggle-version export; it never declares the whole study complete.

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

The subsequent non-identity E2 **mean-substitution smoke** completed at
revision `0ffc3840e115` with online W&B orchestrator `a6evelfm` and
intervention run `591rm6ni`. It used one fixed target in each of the ten dev
sequences, two banks and boundary tokens only (20 paired rows). E0 passed again
on every sequence, including the value-identity check. Its whole-sequence mean
Dice changes were -0.00406 for attention masking (95% bootstrap CI
[-0.01388, 0.00604]) and -0.00577 for mean value substitution
([-0.01254, 0.00059]). The conditioning-bank means were slightly positive
(+0.00204 and +0.00340, respectively), while recent-bank means were negative
(-0.01017 and -0.01493). The checksum-verified Drive receipt is valid and the
full result is `region_tokens_e2_mean_smoke_v1/E2_mean_smoke/summary.md`.
This validates the exercised non-identity value path, not E2 efficacy or any
H1--H4 conclusion; the full E2 matrix remains required.

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

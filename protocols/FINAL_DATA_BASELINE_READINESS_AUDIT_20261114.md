# Data and baseline readiness audit

Date: 2026-11-14. This is a review of local source and saved receipts, not a
claim of independent validation or a complete rescore of historical runs.

## Cohort and splits

`data/polypgen_sequence.jsonl` contains 2,225 image/mask rows: 1,793 in 15
`dev` sequences and 432 in eight `test` sequences. Each row has only `split`,
`video_id`, `frame_index`, `image` and `mask`; it does not establish patient
or center identity. The current five-epoch and 30-epoch fixed-read studies
trained on seq2/3/4 and evaluated seq9/10, with two seq4 training examples
selected after label inspection. Seq9/10 were already inspected and reused;
seq11/20/23 have been repeatedly inspected as difficult cases. None of those
outcomes should be described as blinded external confirmation. The checkpoint
and original detector's patient-level training overlap remain unverified.

The manifest's `frame_index` is a contiguous staging ordinal. Original image
filenames have non-unit and sometimes irregular numeric gaps, e.g. seq4 starts
357, 361, 368; seq20 starts 0, 20, 40; seq23 starts 0, 80, 160. Their precise
timebase and what the suffixes denote are not established here. Treat the
inputs as the provided ordered samples, not necessarily adjacent original
video frames, and do not claim video timing or real-time throughput from the
sample ordinal alone. `datasets/video.py` sorts by ordinal and stages the
files as consecutive predictor inputs.

The earlier author comparisons use 21 selected sequences, excluding seq1 and
seq7 for the historical ground-truth prompt protocol; see
`protocols/region_mass_polypgen_split_v1.json` and
`protocols/full_propagation_false_alarm_v1.json`. Those exclusions were made
for that protocol, not established here as an exhaustive assessment of an
automatic detector on every available sequence. Dataset `readme.md` states
that negative-only sequences supply images without masks. Consequently the
annotated empty frames within the selected positive sequences should not be
represented as an exhaustive negative-video evaluation.

Local release files disagree on licensing: `data/PolypGen2021_MultiCenterData_v3/readme.md`
names CC BY 4.0 and also mentions same-terms licensing, while the bundled
`license.txt` names CC BY-NC-SA 4.0. The dataset publication describes CC BY;
verify the authoritative terms for this exact release before redistribution
or a data-availability statement. No legal determination is made here.

## Prompt schedule and eligibility

Historical automatic runs detect with YOLO on each chronological image until
the first detection, then initialize and propagate without subsequent
redetection (`scripts/run_lean_auto_multiscale.py:111–116`;
`scripts/run_seq23_viewport.py:64–70`). Thus they perform **one initialization**,
but may make several detector calls before it. This is not equivalent to running
the detector exactly once on the first frame; whether a proposed new protocol
permits the former must be made explicit. Labels do not choose the detection
frame. For example, the archived uncropped seq4 run initialized at index 9,
after ten YOLO calls and nine labeled-present pre-prompt frames; the archived
uncropped seq20 run initialized at index 13 after fourteen calls. These are
historical records, not claims about any newly run matched comparison.

For every automatic method, freeze detector bytes, settings, crop, prompt
image and method, and report initial detection failures and any labeled-present
frames before initialization. The seq23 inset-excluded crop is retrospective
and must be applied consistently; its original frame-0 prompt differs from the
uncropped prompt. The selected fixed-read captures use a 16,384-token project
window, indexed RoPE, multiscale anchor and archived box-plus-feature-point
prompts; these should not be labeled as unmodified author Lean.

## Metrics and comparators

The exploratory fixed-read receipt averages seven present frames, not videos;
three empty frames all come from seq10. It cannot supply patient-level
uncertainty or an independent false-alarm estimate. The 17/10 five-epoch
multilevel refiner scored 0.920678 against 0.921285 for its captured baseline
and produced 3/3 empty-frame false alarms; the 30-epoch development follow-up
scored 0.859409 with 3/3 false alarms. Only one seed and the mask-only
capacity control were run. The project protocol's fixed-frame candidate gate
therefore has not been met; a new candidate and other feature/contrast controls
remain necessary before efficacy claims.

Use equal per-video present-frame Dice, labeled-empty false-alarm rates and
false-positive area with denominators, and report full-frame and pre-prompt
outcomes as described in `MICCAI_2027_EXPERIMENT_PROTOCOL.md`. Verify label
completeness; labeled-empty is not automatically proven absence. A scan of
the 432 `test` masks found 108 with grayscale maximum zero and 324 with
maximum above 127; no test-mask maximum lay between 1 and 127. This check does
not prove annotation completeness.

Historical native SAM2, Lean, SAMURAI and SAM2Long use GT-mask prompts; they
are not matched automatic-prompt controls. The automatic SAMURAI comparison
is preparatory only (`outputs/experiment/samurai_automatic_priority_v1/progress.md`
on the configured Drive). SAM2Long's released final path selection is
noncausal; CMSA-Net is a distinct clip model with released column-wise
normalization. `COMPARATOR_AUDIT_2026_11.md` documents source-level issues,
temporal access, incomplete rescoring and historical runner hashes. Compare
prompt and causal tracks separately and do not claim matched full-video
results without independently verified, matching runs.

Before a confirmatory evaluation, define and validate patient/center identity,
training exposure, disjoint independent data, exact prompts and source frames,
matched controls, multi-seed training, label completeness and a predeclared
analysis. If those are unavailable, the work can still be run and reported as
an explicitly exploratory development study.

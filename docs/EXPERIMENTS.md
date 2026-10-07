# Experiment record

PolypGen is a proxy dataset for the adenoid videos. All inference uses the tight
box of the first non-empty ground-truth mask as the prompt, MedSAM2 frozen, one
seed. Run outputs are on Google Drive under `AdeSEG/outputs/` (layout at the end).

**Reproducibility rule.** Every PolypGen and CholecSeg8k table marked
*(regenerated)* is rebuilt from the saved per-frame CSVs by
`scripts/analyze_runs.py` (outputs in `polypgen/c6/analysis/` and
`polypgen/all23/analysis_{all23,dev}/` on Drive) and was checked against the earlier
numbers. Results marked *(archived)* come from code that has since been removed;
their numbers are kept as a record but cannot be regenerated from this
repository. Two aggregations appear and are always named: **per-sequence** (mean
over sequences of the per-sequence mean; used for all paired tests) and
**pooled** (mean over frames).

## What the method is, as run

```text
frame t ──► MedSAM2 image encoder (frozen) ──► memory attention (frozen) ──► mask decoder (frozen)
                                                 reads [anchor | S] + object pointers
observe:  C_t = MedSAM2 memory encoder(F_t, mask_t)
update:   P⁻ = P + q;  R_t = r + a·(1 − p_t);  K_t = P⁻/(P⁻ + R_t)
          S ← S + K_t (C_t − S);  P ← (1 − K_t) P⁻
```

`modeling/kalman_memory.py` replaces MedSAM2's 7-frame FIFO bank with the prompt
frame's memory (anchor) plus one recursive state `S` (64 × 32 × 32). The code
has per-pixel machinery — a variance map `P`, a learned spatial correction of
`R`, and an uncertainty embedding `u` that adds `u·log(1+P)` to the read — but
**in every reported run it is untrained**: the correction's last layer and `u`
are zero, so `R_t` is one scalar per frame, `P` stays spatially uniform, and `P`
is never read. As run, the method is a **scalar Kalman filter on a 1-slot
memory whose measurement noise switches with MedSAM2's presence probability
`p_t`** (q = 1, r = 0.1, a = 10, P₀ = 1). Its steady-state gain is
`K* = M*/(M* + R)` with `M* = (q + √(q² + 4qR))/2`: 0.92 when present
(R ≈ 0.1) and 0.27 when absent (R ≈ 10.1). The measured dev means are 0.835 and
0.266.

The constant-gain EMA (`modeling/ema_memory.py`, `--fixed_gain 0.835 0.266`)
uses the Kalman run's own mean dev gains, so a tie between them is expected by
construction. They differ only in (i) the transient gain right after a presence
switch, (ii) soft presence `p_t` (Kalman) versus a hard 0.5 threshold (EMA), and
(iii) `--skip_absent`, under which an absent frame is a missing measurement
(K = 0, the variance grows, so the gain jumps on reappearance). The
`--skip_absent` comparison is therefore the test of what is specific to the
Kalman form. Under `--skip_absent` the spatial state is not written, but the
absent frame's object pointer still enters memory attention through MedSAM2's
pointer policy (true of every reported run).

`--untrained` builds this module fresh; it reproduces the earlier untrained
checkpoint (now deleted) frame for frame on C6 (maximum Dice difference
0.0), because the zero-initialised last layer makes it independent of the
random seed.

**Memory and speed** (`scripts/infer.py` records both in `setup.json`; C6 rerun,
Apple M-series MPS, propagation only; `polypgen/c6/efficiency/`):

| | Retained memory at the end of a video | Frames per second |
|---|---|---|
| native MedSAM2 | grows with the video: 4.5–8.1 MiB on C6 (mean 6.2), ~0.13 MiB per frame | 20.2 |
| Kalman (1 slot) | 0.754 MiB, fixed (float32 anchor, position code, mean, variance) | 22.1 |
| EMA (1 slot) | 0.754 MiB, fixed | 22.3 |

Native stores every frame's memory but attends to at most 7; those 7 bf16 maps
are 7 × 64 × 32 × 32 × 2 B = 0.875 MiB. The 1-slot state is therefore no smaller
than a trimmed bank; its advantages are a fixed size and two attended memory
maps instead of seven. The Kalman state also keeps the previous frame's image
features (1 MiB) for the motion diagnostic. Earlier figures (0.88 MiB, 36.3 MiB,
13.3 / 17.0 FPS) came from removed measurement code and are superseded.

## Data protocol

| Role | Source |
|---|---|
| Development (protocol (a)) | `sequenceData/positive` seq1–15, centres C1–C5 (13 sequences with a polyp, 1,640 propagated frames) |
| Test (protocol (a)) | seq16–23, centre C6 (8 sequences, 372 propagated frames, 56 empty) |
| First protocol | all 23 sequences as the test set (2,012 propagated frames: 1,689 polyp, 323 empty) |

`scripts/build_polypgen_manifest.py` writes these splits as `dev` and `test`. It
has no `train`/`val` split, so `scripts/train_kalman.py` (which trains on
natural clips from a manifest's train and val splits) cannot be run on it as
is. All reported results use the untrained module.

**Archived training runs.** Earlier training used pseudo-videos assembled from
PolypGen single frames (`data_C1`..`data_C6`, 1,399 frames after removing 12
duplicates of test frames) and `sequenceData/negativeOnly` (23 sequences, 4,275
frames). That pipeline (`datasets/pseudo_video.py`) has been deleted, and its
runs also used detector observations. Those runs improved pseudo-video
validation but not real-video results; they describe the old setup, not the
current `train_kalman.py`.

## How often C6 has been used

C6 is called "held out" below only in the sense that no design choice was
*calibrated* on it after protocol (a) was defined. It was not untouched:

1. The first protocol tested on all 23 sequences, so C6 results of native,
   prompt + 1 frame and Kalman (`polypgen/all23/`) and of earlier memories were seen
   before protocol (a) existed (`8ae385c`).
2. A C6 rerun of the Kalman memory with motion logging (folder since deleted).
3. `6ca458a`: native / EMA / Kalman on C6 with the misordered manifest
   (discarded).
4. `063853e`: the corrected rerun, reported below. The output presence gate
   first appears in this commit, after the earlier C6 run had shown that empty-
   frame false positives were the problem.
5. The presence Kalman filter below was evaluated on C6 and failed.

So C6 has been used for several frozen comparisons, and the gate's design was
informed by C6 results. The adenoid data will be the first clean test.

## C6 results *(regenerated)*

`polypgen/c6/`; analysis in `polypgen/c6/analysis/`. Per-sequence means over
propagated frames. Lost = polyp frames with Dice < 0.1; reappearance = polyp
frames 1–10 after an absence; high motion = polyp frames with image-feature
change > 0.117 (the dev tercile cut, from the Kalman run's log). "+ gate" empties
the output mask where MedSAM2's object score is below τ.

| Method | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance | High motion |
|---|---|---|---|---|---|---|
| native (7-frame bank) | **0.617** | 0.599 | 0.486 | 0.283 | 0.462 | 0.563 |
| EMA (constant-gain 1-slot) | 0.603 | **0.623** | 0.587 | **0.268** | **0.525** | **0.588** |
| Kalman (1-slot) | 0.602 | 0.613 | 0.555 | 0.284 | 0.483 | 0.579 |
| native + gate | **0.628** | 0.573 | **0.332** | 0.326 | 0.391 | 0.516 |
| EMA + gate | 0.620 | **0.607** | 0.433 | **0.288** | **0.493** | **0.574** |
| Kalman + gate | **0.628** | 0.598 | 0.388 | 0.303 | 0.457 | 0.558 |

Paired Wilcoxon over the 8 sequences (A − B; sequences where A is higher / lower;
sequence-bootstrap 95% interval):

| Comparison | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance |
|---|---|---|---|---|---|
| Kalman − native | −0.014 (7/1), p = 0.20, [−0.082, +0.029] | +0.015 (6/2), p = 0.31 | +0.069 (2/1), p = 0.50 | +0.001 (2/2), p = 0.88 | +0.021 (4/3), p = 0.58 |
| Kalman − EMA | −0.001 (7/1), p = 0.20 | −0.010 (5/3), p = 0.74 | −0.032 (0/3), p = 0.25 | +0.016 (2/0), p = 0.50 | −0.042 (3/4), p = 0.22 |
| Kalman+gate − native+gate | −0.000 (4/3), p = 0.94 | +0.025 (5/2), p = 0.30 | +0.056 (2/0), p = 0.50 | −0.023 (1/3), p = 0.38 | +0.066 (3/3), p = 0.69 |

Nothing is significant at n = 8. Kalman's Dice is higher than native's in 7
sequences, but its mean is lower because of seq21 (below). Pooled lost-episode
metrics on C6: native lost 0.247 of polyp frames, mean episode 7.1 frames,
recovered 0.64, lost until the end 0.36; Kalman 0.244, 7.7, 0.60, 0.40. **On C6
there is no error-accumulation effect.**

**Gate thresholds.** τ = 1.557 (native), 0.651 (EMA), 1.078 (Kalman) were
calibrated on dev runs of the corrected pipeline that were not saved, with a
rule that was not recorded. They cannot be regenerated: on the archived dev runs
(`polypgen/all23/`), maximising per-sequence dev Dice gives 1.69 / 0.43 / 1.16
and Youden's J gives 3.57 / 2.86 / 2.98. The "+ gate" rows are reproducible
*given* these τ (`analyze_runs.py --tau`); the τ themselves are not.

**seq21.** The sequence where Kalman falls furthest below native (0.448 vs
0.685) is not a tracking collapse: on its polyp frames Kalman is on par with
native (polyp-frame Dice 0.751 vs 0.757; lost fraction 0.083 vs 0.042); the loss
is a 21-frame empty tail
where native's object score turns negative but the 1-slot memories keep it
positive. MedSAM2's object score calls 71–74% of dev empty frames present for
every method; on C6, 38% (native), 62% (EMA) and 57% (Kalman) of 56 empty
frames.

**Presence Kalman filter (pre-registered; failed on C6, not adopted)**
*(archived; `scripts/train_presence_filter.py` was deleted)*. A scalar Kalman
filter on the object-score log-odds over time (`x⁻ = x`, `P⁻ = P + ρ`,
`K = P⁻/(P⁻ + 1)`, emit the mask iff `x_t > τ`; `ρ = ∞` is the per-frame gate),
selected by leave-one-sequence-out on dev. Rule: beat the per-frame gate on dev
Dice and FP without losing more than 0.005 polyp-frame Dice. On the Kalman memory
it passed narrowly on dev (Dice +0.0005, FP −0.010; selected ρ = 0.01,
τ = 1.63) but on C6 it was worse than the per-frame gate (Dice −0.005,
FP +0.107, worse in 3/8 sequences and better in none): heavy smoothing reacts
slowly when the polyp leaves.

**Update-rule test with `--skip_absent` (dev, pre-registered; failed, not run on
C6)** *(archived; these runs are not on Drive)*. Kalman+skip vs EMA+skip
(constant gain 0.833 = Kalman+skip's mean present-frame gain). Rule: Kalman+skip
must raise reappearance Dice and lower the lost fraction. Dice +0.006
(p = 0.24), polyp-frame Dice +0.004, empty-frame FP −0.031 and lost fraction
−0.016 (Kalman lower in the 4 sequences that differ, p = 0.13), reappearance
0.000. **Fail.** This is the comparison that isolates the Kalman form, and it
shows no measurable benefit.

## First protocol and dev split *(regenerated)*

Archived runs in `polypgen/all23/` (`native`, `native_prompt_plus_1frame`,
`kalman`, `ema`; older pipeline, frames in correct
order; the current pipeline reproduces their dev and C6 Dice). These results
informed the design, and the episode metrics were defined after seeing them;
they are development evidence, not a test.

**All 23 sequences:**

| | Dice (per-seq / pooled) | Polyp-frame Dice (per-seq / pooled) | Empty-frame FP (per-seq / pooled) | Lost (pooled) | Mean episode | Recovered | Lost until end |
|---|---|---|---|---|---|---|---|
| native | 0.627 / 0.507 | 0.652 / 0.537 | 0.649 / 0.650 | 0.374 | 25.3 | 0.60 | 0.36 |
| prompt + 1 frame | 0.616 / 0.513 | 0.644 / 0.552 | 0.652 / 0.687 | 0.326 | 19.0 | 0.55 | 0.31 |
| Kalman | 0.634 / 0.531 | 0.671 / 0.574 | 0.654 / 0.693 | 0.304 | 16.1 | 0.75 | 0.22 |
| EMA | 0.632 / 0.523 | 0.674 / 0.569 | 0.686 / 0.718 | 0.327 | 18.4 | 0.67 | 0.23 |

Lost episodes are runs of consecutive polyp frames with Dice < 0.1 (empty frames
do not break a run); recovered = a later polyp frame of the sequence reaches
Dice ≥ 0.5; lost until end = the run reaches the last polyp frame.

Paired per-sequence tests (21 sequences with a polyp; 17 with empty frames):

| Comparison | Dice | Polyp-frame Dice | Empty-frame FP | Lost fraction | High motion |
|---|---|---|---|---|---|
| Kalman − native | +0.007 (14/7), p = 0.14, [−0.024, +0.032] | +0.019 (14/7), p = 0.070 | +0.005 (4/3), p = 0.74 | −0.024 (3/6), p = 0.21 | +0.026 (15/5), p = 0.027 |
| Kalman − prompt + 1 | +0.018 (14/7), p = 0.34 | +0.027 (17/4), p = 0.014 | +0.002 (4/3), p = 0.74 | −0.025 (0/5), p = 0.043 | +0.039 (14/6), p = 0.030 |
| Kalman − EMA | +0.002 (15/6), p = 0.12 | −0.003 (12/9), p = 0.92 | −0.032 (0/6), p = 0.028 | −0.001 (4/3), p = 0.74 | −0.006 (12/8), p = 0.57 |

**Correction.** Earlier versions of this record and the report paired pooled
differences (Dice +0.025, polyp-frame Dice +0.037) with per-sequence p-values,
and reported the pooled lost-fraction drop (−0.070) with p = 0.016 against native
and p = 0.011 against prompt + 1 frame. Those two p-values cannot be reproduced
from the saved per-frame data with any standard test tried (per-sequence
Wilcoxon on fractions or counts, one- or two-sided, sign test, paired t-test).
The per-sequence test gives p = 0.21 against native. **Fewer lost frames on the
long videos is a pooled, development-set observation, not a significant
result.**

**Dev split (seq1–15, 13 sequences with a polyp):**

| | Dice (per-seq / pooled) | Polyp-frame Dice (per-seq / pooled) | Lost (pooled) | Mean episode | Recovered | Lost until end |
|---|---|---|---|---|---|---|
| native | 0.633 / 0.481 | 0.685 / 0.518 | 0.403 | 39.6 | 0.57 | 0.36 |
| prompt + 1 frame | 0.614 / 0.488 | 0.670 / 0.534 | 0.343 | 23.6 | 0.60 | 0.25 |
| Kalman | 0.653 / 0.514 | 0.706 / 0.559 | 0.318 | 19.9 | 0.82 | 0.14 |
| EMA | 0.649 / 0.504 | 0.705 / 0.551 | 0.350 | 26.7 | 0.78 | 0.17 |

Kalman − native on dev: Dice +0.020 (7/6), p = 0.46; lost fraction −0.040 (1/4),
p = 0.19; high-motion polyp Dice +0.033 (9/3), p = 0.052. Kalman − EMA on dev:
Dice +0.004, p = 0.41; lost −0.012, p = 0.63; reappearance −0.015 (1/9),
p = 0.020 (EMA better). Pre-set rule (Kalman must beat EMA on dev lost fraction
and high-motion Dice): not met.

**Camera motion (dev, pooled polyp frames; motion = 1 − cosine similarity of
MedSAM2 image features with the previous frame; tercile cuts 0.033 / 0.117):**

| Dev | Low: Dice / lost | Medium | High |
|---|---|---|---|
| native | 0.614 / 0.297 | 0.541 / 0.377 | 0.397 / 0.540 |
| prompt + 1 frame | 0.615 / 0.220 | 0.559 / 0.326 | 0.427 / 0.487 |
| Kalman | 0.628 / 0.209 | 0.583 / 0.302 | 0.464 / 0.446 |

(The earlier version of this table differed by at most 0.003.)

## Robustness analyses *(regenerated)*

`scripts/analyze_reappearance.py` (polyp frames 1–10 after at least one empty
frame, mean per offset then per sequence; 23 sequences): native 0.544, prompt + 1
frame 0.561, Kalman 0.554, EMA 0.579; Kalman − native +0.010 (11/6), p = 0.27;
EMA − native +0.035 (10/7), p = 0.10.

`scripts/analyze_illumination.py` (luminance terciles over polyp frames, low ≤
0.300 < mid ≤ 0.361 < high):

| Bucket | native | prompt + 1 frame | Kalman | EMA | Kalman − native |
|---|---|---|---|---|---|
| low | 0.609 | 0.599 | 0.601 | 0.601 | −0.008, p = 0.65 |
| mid | 0.665 | 0.661 | 0.688 | 0.703 | +0.023, p = 0.055 |
| high | 0.651 | 0.615 | 0.694 | 0.700 | +0.043, p = 0.75 |

Both robustness analyses are post hoc and use the all-23 runs.

## Gain variants *(archived; code removed, Drive folders 21–23 deleted)*

Each was judged on dev by a rule fixed beforehand; none went to C6. The code is
in git history (reliability `e0c1e44`, motion noise `1b520c7` / `628bcf1`, oracle
`8422a52`, score gates `9d676ff` / `992b4b8`; removed in `ccb4fd4` and
`f256be9`).

| Gain change | Dev result | Verdict |
|---|---|---|
| R from calibrated reliability of the object score (LOSO AUROC 0.745) | Dice 0.493, lost 0.348 | worse than Kalman |
| Process noise ∝ per-pixel feature change | Dice 0.490, lost 0.339 | worse |
| Measurement noise ∝ feature change | Dice 0.502, lost 0.353 | worse |
| Prototype innovation gate | real-video AUROC 0.43 | uninformative |
| Prompt-anchor distances (memory, pointer, image) | pooled AUROC 0.76 / 0.63 / 0.74 | no gain over the object score |
| Object score (raw logit) | AUROC 0.887 | best available signal |
| Hard score gate (τ = 5.74), all frames / present only | Dice 0.527 / 0.497; empty-frame FP 0.914 / 0.809 | fail |

**Oracle gain** (diagnostic): write a frame only if it matches the ground truth
(IoU ≥ 0.5 or correctly empty), otherwise K = 0. Dev pooled Dice 0.614, lost
0.183. This bounds what an informed write decision could gain on PolypGen dev;
it is not reproducible from the current code.

## CholecSeg8k *(partly regenerated)*

Gallbladder class, 88 sequences of 80 frames from 16 videos, split by video
(dev 60 sequences, test 28 from 6 videos); `outputs/cholecseg8k/`. The
converter `scripts/prepare_cholecseg8k.py` was removed (it is in commit
`9bb801a`), so the runs cannot be recreated from this repository, but the
tables below are regenerated from their saved CSVs.

| | dev Dice (per-seq / pooled) | test Dice (per-seq / pooled) | dev lost (pooled) | test lost (pooled) |
|---|---|---|---|---|
| native | 0.916 / 0.916 | 0.860 / 0.856 | 0.015 | 0.069 |
| prompt + 1 frame | 0.926 / 0.926 | 0.875 / 0.872 | 0.002 | 0.055 |
| Kalman | 0.925 / 0.925 | 0.876 / 0.873 | 0.004 | 0.054 |
| EMA (0.911 / 0.284) | 0.926 / 0.926 | 0.877 / 0.875 | 0.002 | 0.052 |

Kalman − native per sequence: dev +0.009 (38/22), p = 0.22; test +0.016
(17/11), p = 0.26. The earlier table mixed per-sequence (dev) and pooled (test)
Dice. Tracking rarely fails on dev (0.2–1.5% of frames lost) but more often on
test (5–7%); every 1-slot variant and prompt + 1 frame are within 0.002 of each
other.

## Two-region stand-in: REFUGE *(regenerated)*

The adenoid data are expected to follow Cai et al. 2024: grey masks with 0
background, 128 unobstructed nasopharyngeal airway, 255 adenoid, graded by
A/N = adenoid / (adenoid + airway). REFUGE (fundus) uses the same 0/128/255
coding (255 background, 128 optic-disc rim, 0 cup), so cup / (cup + rim) has
the form of A/N. Masks converted with `build_mask_manifest.py --value_map 255:0
128:1 0:2`; native MedSAM2, ground-truth box prompts; scored by
`two_region.py --region_a cup:2 --region_b rim:1 --ratio_mode fraction_of_total`.

| REFUGE val (400 images) | Cup Dice | Rim Dice | Ratio MAE | RMSE | Pearson | Spearman |
|---|---|---|---|---|---|---|
| one pass per label (rim box = disc box) | 0.734 | 0.886 | 0.109 | 0.138 | 0.74 | 0.86 |
| `--nested_labels` (prompt disc and cup; rim = disc − cup) | **0.918** | **0.924** | **0.019** | **0.024** | **0.98** | **0.98** |

The first design is flawed: the rim's ground-truth box is the whole disc's box,
and with one pass per label and "highest logit wins" the rim prediction absorbs
the cup. The predicted cup fraction was below the truth on **all 400** images
(mean signed error −0.109), and the ratio error tracked cup Dice (Spearman
ρ = −0.74; images with both Dice ≥ 0.85 had MAE 0.034). That was a prompting
artefact, not a Dice–measurement mismatch. Prompting the outer region and the
inner one (`--nested_labels`) matches A/N, where nasopharynx = adenoid + airway.
With it, the signed error is +0.008 and images with both Dice ≥ 0.9 (249) still
have MAE 0.015. **REFUGE does not show a Dice–measurement mismatch**; it does not
motivate a measurement-aware loss. Results: `outputs/refuge/val_nested/`; the
first design is reproduced by running `infer.py` without `--nested_labels`.

## Drive folders (`AdeSEG/outputs/`)

| Folder | Content |
|---|---|
| `polypgen/c6/{native,ema,kalman}` | C6 runs, current pipeline |
| `polypgen/c6/analysis`, `efficiency`, `gate_thresholds.json` | regenerated C6 tables; memory and FPS; gate thresholds |
| `polypgen/all23/{native,native_prompt_plus_1frame,kalman,ema}` | archived all-23 runs (first protocol) |
| `polypgen/all23/analysis_all23`, `analysis_dev` | regenerated tables for all 23 sequences and for dev |
| `cholecseg8k/{native,native_prompt_plus_1frame,kalman,ema}` | archived CholecSeg8k runs; `split.json` |
| `refuge/val_nested` | REFUGE val with `--nested_labels` |

Overlay images were removed (rerun `evaluation/temporal.py` without
`--no_overlays` to regenerate them). Deleted as failed or superseded: the dev
gain variants (reliability, motion noise, skip + reliability), the C6 motion
rerun, the oracle and score-gate CholecSeg8k runs, the first-design REFUGE run,
the untrained checkpoint, and the early summary tables. `outputs/README.md`
describes each folder.

## Caveats

- Any run made with a manifest built before commit `c8df175` (alphabetical
  frame order) is invalid.
- An MPS inference bug (MedSAM2's non-blocking offload read before the copy
  finished) corrupted early results; fixed in `f96742d`. All numbers here are
  after the fix.
- One proxy video dataset plus CholecSeg8k, one seed, ground-truth box prompts.
- Missing baselines: a trimmed 7-slot bank, SAMURAI, SAM2Long, DAM4SAM, and
  EMA-SAM (the closest prior work: a confidence-weighted moving-average memory
  for medical SAM2).

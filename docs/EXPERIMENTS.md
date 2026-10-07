# Experiment record

PolypGen is a proxy dataset for the adenoid videos. Numbers below are on the
23 `sequenceData/positive` sequences (2,012 propagated frames: 1,689 with a
polyp, 323 empty), prompted with the tight box of the first non-empty ground
truth mask, MedSAM2 frozen, one seed. Results are on Google Drive under
`outputs/Kalman/<run>/`.

The repository scope was trimmed on 2026-10-06: the detector observations,
presence fusion, and all DOK-Mem code and runs were removed. What remains is
the plain Kalman spatial memory replacing MedSAM2's 7-frame FIFO bank, and the
constant-gain EMA baseline it is compared with (`modeling/ema_memory.py`).
**The current results are in "Held-out test" and "Two-region stand-in:
REFUGE" below**; later sections record the archived runs and analyses that
led to them.

Two protocols were used, in this order:

1. **All 23 sequences as the test set** (first-protocol baselines).
2. **Protocol (a)** (`datasets/splits/polypgen/protocol_a.json`): dev =
   seq1–15 (centres C1–C5) for every calibration and design choice; test =
   seq16–23 (centre C6, confirmed by frame names), run once per frozen method.
   CholecSeg8k is a second real-video dataset reported in its own section.

## Architecture: 1-slot Kalman spatial memory

```text
frame t ──► MedSAM2 image encoder (frozen) ────────────► memory attention (frozen)
                                                               reads [anchor | S + u·log(1+P)]
                                                                                     │
                                                                              mask decoder (frozen)
                                                                                     │
                                               observe:  C_t = MedSAM2 memory encoder(F_t, mask_t)
                                                                                     │
                                               Kalman:   P⁻ = P + q
                                                         R   = (r + a·(1 − p_t)) · c(spatial map)
                                                         K   = P⁻ / (P⁻ + R)
                                                         S  ← S + K(C_t − S)
                                                         P  ← (1 − K) P⁻
```

| Component | Code | Learned? |
|---|---|---|
| Kalman memory (prompt anchor + mean `S` + per-pixel variance `P`) | `modeling/kalman_memory.py` | noise level hand-set (`q=1`, `r=0.1`, `a=10`, `P0=1`); bounded spatial correction is trainable but untrained in these runs |
| Object pointers | `modeling/native_pointers.py` | MedSAM2's own selection |

Kalman step (identity transition, diagonal covariance shared across 64
channels at each pixel, identity observation model; RKN's update equations):

```text
predict:  P⁻ = P + q
read:     memory attention sees [anchor | S + u·log1p(P⁻)] and MedSAM2's object pointers
observe:  C_t = MedSAM2 memory encoder(F_t, mask_t)
update:   R_t = (r + a · (1 − p_t)) · c(D_t − mean(D_t))
          K_t = P⁻ / (P⁻ + R_t);  S ← S + K_t(C_t − S);  P ← (1 − K_t) P⁻
```

`c(x) = 10^tanh(x)`, so the untrained module is the hand-set filter: a polyp
frame writes about 95% of the memory, an empty one about 17%. The anchor uses
MedSAM2's conditioning-frame temporal code (`maskmem_tpos_enc[num_maskmem−1]`),
the state the most-recent-frame code (`maskmem_tpos_enc[0]`). Retained memory:
about 0.88 MiB per object, constant in video length.

Options (`scripts/infer.py` flags):

| Flag | Effect |
|---|---|
| `--skip_absent` | A frame MedSAM2 calls empty is a missing measurement (`K = 0`, variance grows). |
| `--fixed_gain PRESENT ABSENT` | EMA baseline: the constant-gain memory in `modeling/ema_memory.py` instead of the Kalman update. |

The constant-gain EMA baseline (`ConstantGainMemory`) shares the Kalman read
path but replaces the adaptive gain with a fixed present/absent gain, with the
variance kept consistent (`P ← (1 − k)P⁻`). Exploratory gain variants that did
not beat the baseline (hard score gates, reliability-
weighted and motion-scaled noise, the oracle and prototype gates) were removed
from the code; their earlier runs remain on Google Drive and are summarised
under "Protocol (a)" below.

## Data protocol

Training (when used) reads only `data_C1`..`data_C6` single frames and
`sequenceData/negativeOnly`; it never reads `sequenceData/positive`. All
inference numbers in this document use GT-box prompts from the first polyp
frame.

| Role | Source |
|---|---|
| Positives (training frames) | `data_C1`..`data_C6` single frames (1,399 after excluding 12 frames that duplicate test frames) |
| Absences (training) | `sequenceData/negativeOnly` (23 sequences, 4,275 frames) |
| Test | all 23 `sequenceData/positive` sequences |

## Held-out test (protocol (a), C6 = seq16–23)

Frozen native, EMA and Kalman run once on C6 with the manifest pipeline
(`outputs/Kalman/heldout_test/`, 8 sequences, 372 propagated frames). Frames
are in numeric temporal order: an earlier manifest sorted frame files
alphabetically and fed 18 of 23 sequences out of order, so all results from it
were discarded (fixed in `c8df175`). The corrected pipeline reproduces the
archived runs exactly (pooled propagated-frame Dice: dev native 0.481 / Kalman
0.514; C6 native 0.621 / Kalman 0.609). EMA uses the Kalman update's own dev
gains (present 0.835, absent 0.266). "+ gate" is an output presence gate (empty
mask when MedSAM2's object score < τ, τ calibrated on each method's dev run:
native 1.557, EMA 0.651, Kalman 1.078). Per-sequence means; paired Wilcoxon (n = 8).

| Method | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance | High motion |
|---|---|---|---|---|---|---|
| native (7-frame bank) | **0.617** | 0.599 | 0.486 | 0.283 | 0.462 | 0.563 |
| EMA (constant-gain 1-slot) | 0.603 | **0.623** | 0.587 | **0.268** | **0.525** | **0.588** |
| Kalman (adaptive-gain 1-slot) | 0.602 | 0.613 | 0.555 | 0.284 | 0.483 | 0.579 |
| native + gate | **0.628** | 0.573 | **0.332** | 0.326 | 0.391 | 0.516 |
| EMA + gate | 0.621 | **0.607** | 0.433 | **0.289** | **0.493** | **0.574** |
| Kalman + gate | **0.628** | 0.598 | 0.388 | 0.303 | 0.457 | 0.558 |

| Comparison | Dice | Polyp-frame Dice | Empty-frame FP | Reappearance |
|---|---|---|---|---|
| Kalman − native | −0.014 (Kalman better in 7/8; the loss is seq21's empty tail), p = 0.20 | +0.015, p = 0.31 | +0.069, p = 0.50 | +0.021, p = 0.58 |
| Kalman − EMA | −0.001, p = 0.20 | −0.010, p = 0.74 | −0.032, p = 0.25 | −0.042, p = 0.22 |
| Kalman+gate − native+gate | −0.000, p = 0.94 | +0.025, p = 0.30 | +0.056, p = 0.50 | +0.066, p = 0.69 |

The 1-slot memories track polyp frames and high-motion frames slightly better
than native's bank but raise empty-frame false positives, ending slightly below
native on overall Dice. The Kalman adaptive gain ties the constant-gain EMA. The
output presence gate lowers false positives for every method; with it, Kalman
ties native on Dice with better polyp-frame Dice, while native keeps the lowest
false-positive rate. Nothing is significant at n = 8.

**Update-rule test (dev, pre-registered; failed, not run on C6).** With
`--skip_absent`, absent frames are missing measurements, so the Kalman variance
grows during an absence and the gain jumps on reappearance, which a constant
gain cannot do. Kalman+skip vs EMA+skip (constant gain 0.833, Kalman+skip's mean
present-frame gain): Dice +0.006 (p = 0.24), polyp-frame Dice +0.004, empty-frame
FP −0.031 and lost fraction −0.016 (Kalman lower in every one of the 4 differing
sequences, p = 0.13), reappearance Dice 0.000. The rule (reappearance up and lost
fraction down) fails. Without skip, EMA beats Kalman on reappearance (−0.015,
9/10 sequences, p = 0.02).

**Failure analysis of seq21** (the C6 sequence where Kalman falls furthest
below native, 0.448 vs 0.685): not a tracking collapse. On its polyp frames
Kalman matches or beats native (e.g. 0.86 vs 0.15 and 0.86 vs 0.32); the loss
is a 21-frame empty tail where native's object score turns negative but the
1-slot memories keep it positive and emit false positives. MedSAM2's object
score is a weak presence signal in general: on dev it calls 71–74% of empty
frames present for every method, native included (C6: native 38%, EMA 62%,
Kalman 57%, of 56 empty frames, 21 of them seq21's tail).

**Presence Kalman filter (pre-registered; failed on C6, not adopted).** A
scalar Kalman filter on the object-score log-odds over time (`x⁻ = x`,
`P⁻ = P + ρ`, `K = P⁻/(P⁻+1)`, emit the mask iff `x_t > τ`; `ρ = ∞` is the
per-frame gate), selected by leave-one-sequence-out on dev. Rule: beat the
per-frame gate on dev Dice and FP without losing more than 0.005 polyp-frame
Dice. On the Kalman memory it passed narrowly on dev (Dice +0.0005, FP −0.010;
selected ρ = 0.01, τ = 1.63) but on C6 it was worse than the per-frame gate
(Dice −0.005, FP +0.107, worse in 3/8 sequences and better in none): the heavy
smoothing reacts slowly when the polyp leaves. The per-frame gate remains the
presence module.

## Two-region stand-in: REFUGE (current)

The target adenoid data are expected to follow Cai et al. 2024: fiberoptic
nasopharyngoscopy images (516×531), 3-class grey masks (0 background, 128
unobstructed nasopharyngeal airway, 255 adenoid), graded by the A/N ratio
(<50% small, 50–75% medium, >75% large). REFUGE (`data/REFUGE`, fundus images,
400 per split) uses the same 0/128/255 coding (255 background, 128 optic-disc
rim, 0 cup) with two adjacent regions, so cup / (cup + rim) has the form of an
A/N ratio. It tests the two-label path (mask conversion, two-label inference,
per-region scores, ratio error) but, as static images, not the video memory.

Masks converted with `scripts/build_mask_manifest.py --value_map 255:0 128:1 0:2`
(all 1,200 images contain background, rim and cup); native MedSAM2 with a GT box
per region; scored by `evaluation/two_region.py --region_a cup:2 --region_b rim:1
--ratio_mode fraction_of_total`. Results: `outputs/REFUGE/native_val_two_region/`.

| REFUGE val (400 images) | Cup Dice | Rim Dice | Cup IoU | Rim IoU | Ratio MAE | Ratio RMSE | Pearson | Spearman |
|---|---|---|---|---|---|---|---|---|
| native MedSAM2, GT-box prompts | 0.734 | 0.886 | 0.589 | 0.801 | 0.109 | 0.138 | 0.74 | 0.86 |

Dice of 0.73–0.89 still leaves an 11-point ratio error on a 0–1 scale, enough
to change an A/N grade near the 0.50 or 0.75 thresholds: first evidence, on a
stand-in, of the Dice–measurement mismatch that motivates a measurement-aware
objective. The Kalman memory also runs the two-label path (checked on three
images). Before evaluating Kalman on adenoid data, the data must be video: Cai
et al.'s dataset is single images, which gives the memory nothing to propagate.

## Main results — plain Kalman memory (no detector, no fusion)

All 23 sequences, propagated frames.

| Run | Dice | IoU | Polyp-frame Dice | Empty-frame FP | Polyp frames detected | ≥10 frames after reappearance | Memory | FPS |
|---|---|---|---|---|---|---|---|---|
| `01_native_medsam2` | 0.507 | 0.466 | 0.537 | 0.650 | 0.906 | 0.452 | grows | 13.3 |
| `02_native_medsam2_prompt_plus_1frame` | 0.513 | 0.469 | 0.552 | 0.687 | 0.926 | 0.466 | grows | 15.7 |
| **`10_kalman_memory`** | **0.531** | **0.487** | **0.574** | 0.694 | 0.927 | **0.519** | **0.88 MiB** | **17.0** |

Paired tests (21 sequences with a polyp, 17 with empty frames; Wilcoxon
signed-rank; bootstrap 95% intervals of the mean difference):

| Comparison | Metric | Difference [95% CI] | Better / worse | p |
|---|---|---|---|---|
| Kalman vs native | Dice | +0.025 [−0.009, +0.063] | 14 / 7 | 0.137 |
| | polyp-frame Dice | +0.037 [+0.000, +0.080] | 15 / 6 | 0.070 |
| | empty-frame FP | +0.044 [−0.019, +0.117] | 7 / 7 | 0.325 |
| Kalman vs prompt + 1 frame | polyp-frame Dice | +0.027 | 17 / 21 better | 0.014 |

**What is and is not supported standalone:**

| Claim | Status |
|---|---|
| Kalman memory raises overall Dice | not significant (p = 0.14) |
| Kalman memory lowers empty-frame false positives | not supported (FP 0.694 vs native 0.650) |
| Kalman memory reduces error accumulation on long videos | supported — see protocol (a) below |
| Kalman memory stores one compressed state instead of a bank | supported in size and recursion; the Kalman adaptive gain itself is matched by a constant-gain EMA |
| Kalman memory is more robust under motion / illumination / reappearance | partially supported — see protocol (a) and robustness analyses |

Memory: native "grows" is what the code stores (36.3 MiB on average). MedSAM2
attends to only 7 frame memories, and one is 64 × 32 × 32 bf16 = 0.125 MiB, so
a trimmed bank would hold about 0.875 MiB, close to the Kalman memory.

## Protocol (a): dev / C6 split

Here the memory is the only difference from native MedSAM2.
Untrained Kalman checkpoint: `outputs/Kalman/kalman_memory_untrained.pt`.

**Dev (seq1–15) vs C6 test, frozen plain Kalman memory:**

| | Dice | Polyp frames lost | Mean episode | Recovered | Lost until end |
|---|---|---|---|---|---|
| dev native | 0.481 | 0.403 | 39.6 | 0.57 | 0.36 |
| dev prompt + 1 frame | 0.488 | 0.343 | 23.6 | 0.60 | 0.25 |
| dev Kalman | 0.514 | 0.318 | 19.9 | 0.82 | 0.14 |
| C6 native | 0.621 | 0.247 | 7.1 | 0.64 | 0.36 |
| C6 prompt + 1 frame | 0.623 | 0.253 | 8.9 | 0.44 | 0.44 |
| C6 Kalman | 0.609 | 0.244 | 7.7 | 0.60 | 0.40 |

C6, Kalman vs native: Dice −0.014 (7 / 8 sequences better; the loss is
seq21's empty tail, see the held-out section, p = 0.20); lost fraction +0.001
(p = 0.88). The dev error-accumulation benefit
does not reach significance on the short C6 videos.

**Error accumulation** (lost episode = consecutive polyp frames with
Dice < 0.1), all 23 sequences:

| | Polyp frames lost | Mean episode (frames) | Episodes recovered | Lost until the end |
|---|---|---|---|---|
| native | 0.374 | 25.3 | 0.60 | 0.36 |
| native, prompt + 1 frame | 0.326 | 19.0 | 0.55 | 0.31 |
| Kalman | 0.304 | 16.1 | 0.75 | 0.22 |

Lost-frame fraction, Kalman vs native −0.070 (p = 0.016); vs prompt + 1 frame
−0.023 (p = 0.011).

**Camera motion.** Motion = per-pixel change of MedSAM2's image features since
the previous frame (1 − cosine). Dev terciles (cuts 0.033 / 0.117), polyp frames:

| Dev | Low motion: Dice / lost | Medium | High |
|---|---|---|---|
| native | 0.612 / 0.300 | 0.541 / 0.376 | 0.400 / 0.538 |
| prompt + 1 frame | 0.614 / 0.221 | 0.558 / 0.326 | 0.429 / 0.484 |
| Kalman | 0.628 / 0.210 | 0.581 / 0.303 | 0.466 / 0.444 |

High-motion polyp Dice, Kalman vs native: +0.033, 9 / 12, p = 0.052. On C6
with the dev cut fixed beforehand (high > 0.117): native 0.539, Kalman 0.559,
+0.016, 6 / 8, p = 0.25.

**Is the Kalman gain responsible?** The untrained gain is nearly constant on
present frames (PolypGen dev mean 0.835, std 0.144; CholecSeg8k 0.911, std
0.033): with constant q and presence-only R the variance converges and a
steady-state Kalman filter is an exponential moving average. Ablation
`19_kalman_fixed_gain_ema` uses constant gains measured on dev
(present 0.835, absent 0.266):

| | dev Dice | dev lost | dev high-motion Dice | C6 Dice | C6 lost |
|---|---|---|---|---|---|
| Kalman | 0.514 | 0.318 | 0.464 | 0.609 | 0.244 |
| EMA | 0.504 | 0.350 | 0.460 | 0.607 | 0.231 |

Kalman vs EMA on dev: lost −0.012 (p = 0.63), Dice +0.004 (p = 0.41). Pre-set
rule (Kalman must beat EMA on dev lost fraction and high-motion Dice): not
met. **The 1-slot recurrent structure helps; the Kalman adaptive gain itself
has not been shown to beat a constant gain.**

Attempts to make the gain informative (each judged on dev by a rule fixed
beforehand; none went to C6):

| Gain change | Dev result | Verdict |
|---|---|---|
| R from calibrated reliability ρ = σ(0.423·s − 2.619) of the object score s (LOSO AUROC 0.745) | Dice 0.493, lost 0.348 | worse than Kalman |
| Process noise ∝ per-pixel feature change (d_ref 0.0958) | Dice 0.490, lost 0.339 | worse |
| Measurement noise ∝ feature change (`20_dev_kalman_motion_measurement`) | Dice 0.502, lost 0.353 | worse |
| Prototype innovation gate | real-video AUROC 0.43 | uninformative |
| Prompt-anchor distances (memory, pointer, image) | pooled AUROC 0.76 / 0.63 / 0.74; no gain over the object score in LOSO | uninformative |
| Object score (raw logit) | AUROC 0.887 overall, 0.88–0.98 within sequences | the best available signal |

**Oracle gain** (`21_dev_kalman_oracle_gain`, diagnostic): a frame is written
only if it matches the ground truth (IoU ≥ 0.5 or correctly empty), otherwise
K = 0. Dev: Dice 0.614, lost 0.183. Against EMA: Dice +0.056; against native:
Dice +0.072 (10/3, p = 0.048). **An informed gain is worth about +0.11
pooled Dice and half the lost frames** over a constant gain on PolypGen; this
is the headroom any new gain scheme must try to realize.

**Hard score-validation gate** (object score below τ ⇒ K = 0;
τ = 5.74 from dev by Youden's J):

| Dev | Dice | Polyp-frame Dice | Empty-frame FP | Lost |
|---|---|---|---|---|
| `22_kalman_score_gate` (all frames) | 0.527 | 0.613 | 0.914 | 0.277 |
| `23_kalman_score_gate_present_only` | 0.497 | 0.556 | 0.809 | 0.336 |

The all-frames gate recovers 21% of the oracle's pooled Dice headroom but
also stops writing empty frames, so empty-frame FP rises to 0.91 and the per
sequence paired test is not better. Both variants fail the pre-registered rule.

## Robustness analyses (no re-inference)

### Reappearance window

Mean Dice per sequence over polyp frames at offsets 1..10 after a reappearance
of at least one empty frame, pooled over the 23 PolypGen sequences
(`scripts/analyze_reappearance.py`).

| | Mean Dice (window 1..10) | Paired vs native |
|---|---|---|
| native | 0.544 | — |
| prompt + 1 frame | 0.561 | +0.017, 8 / 8, p = 0.47 |
| Kalman | 0.554 | +0.010, 11 / 6, p = 0.27 |
| EMA | 0.579 | +0.035, 10 / 7, p = 0.10 |

The 1-slot recurrent memory helps slightly over native in the first 10
reappearance frames, but the Kalman adaptive gain is not better than EMA.

### Illumination

Per-frame mean luminance (Rec. 709 Y) computed from the PolypGen JPEGs
(`scripts/analyze_illumination.py`). Luminance terciles over all polyp frames:
low ≤ 0.300 < mid ≤ 0.361 < high. Dice is per sequence, averaged.

| Bucket | native | prompt + 1 frame | Kalman | EMA | Kalman vs native |
|---|---|---|---|---|---|
| low (dark) | 0.609 | 0.599 | 0.601 | 0.601 | −0.008, p = 0.65 |
| mid | 0.665 | 0.661 | 0.688 | 0.703 | +0.023, p = 0.055 |
| high (bright) | 0.651 | 0.615 | 0.694 | 0.700 | +0.043, p = 0.75 |

The claim that the Kalman memory is more robust under non-ideal lighting is
supported in mid luminance (marginal) and in direction on high luminance (not
significant); low-luminance frames show no advantage.

## CholecSeg8k (second real-video dataset)

CholecSeg8k (CC BY-NC-SA 4.0; 101 clips × 80 consecutive laparoscopic frames,
17 Cholec80 videos, dense masks) was converted into the sequence layout for
one target class (gallbladder: 88 sequences from 16 videos, 191 of 7,040
frames empty; split by whole video: dev 60 sequences, test 28, 6 videos).
Runs under `outputs/Kalman/cholecseg8k/`. The converter script
(`prepare_cholecseg8k.py`) was removed with the earlier scope trim; runs are
preserved on Google Drive but cannot be re-produced from the current repo.

| | dev Dice | dev IoU | test Dice | test IoU |
|---|---|---|---|---|
| native | 0.916 | 0.868 | 0.856 | 0.805 |
| prompt + 1 frame | 0.926 | 0.877 | 0.872 | 0.822 |
| Kalman | 0.925 | 0.877 | 0.873 | 0.823 |
| EMA (gains 0.911 / 0.284 from CholecSeg8k dev) | 0.926 | 0.877 | 0.875 | 0.825 |
| oracle gain (dev only) | 0.920 | 0.871 | — | — |
| score gate τ = 5.74 | 0.915 | — | 0.878 | — |

Kalman vs native: dev +0.009 (38 / 60, p = 0.22), high-motion +0.017
(p = 0.054); test +0.016 (p = 0.26). Kalman vs prompt + 1 frame: dev −0.0004
(p = 0.007). Kalman vs EMA: no difference. Tracking almost never fails here
(lost polyp frames 0.2–1.5%), so no gain policy has room to help.

**Across both datasets:** a 1-slot memory beats or ties MedSAM2's 7-frame bank
in Dice; it reduces error accumulation on long polyp videos; the Kalman
adaptive gain as built equals a moving average; an informed gain has large
headroom only where the tracker drifts (PolypGen).

## Drive folders (`AdeSEG/outputs/Kalman/`)

| Folder | Run |
|---|---|
| `../REFUGE/native_val_two_region/` | **current results**: two-region REFUGE val run (`summary.json`, `per_frame.csv`) |
| `heldout_test/` | **current results**: frozen native / EMA / Kalman on C6 with the corrected manifest (`comparison.md`, `summary.json`, per-method metrics and diagnostics) |
| `01_native_medsam2`, `02_native_medsam2_prompt_plus_1frame` | archived native baselines (older pipeline) |
| `10_kalman_memory` | archived plain Kalman memory |
| `14`, `16`, `17`, `20` (`*_dev*`) | archived dev-only Kalman variants and diagnostics |
| `18_test_kalman_motion_logged` | archived C6 rerun of the Kalman memory with motion logging |
| `19_kalman_fixed_gain_ema` | archived constant-gain EMA ablation |
| `cholecseg8k/` | archived CholecSeg8k runs |
| `summary_before_renaming/` | old summary tables |

The archived folders come from the older pipeline, which ordered frames
correctly; the corrected manifest pipeline reproduces their dev and C6 Dice.
Folders for the oracle gain (`21`) and score gates (`22`, `23`) were removed;
their numbers are kept in the sections above. `26_soft_gain_smoke_seq1` is a
smoke test from the removed soft gain, made with the misordered manifest.

## Negative results (plain Kalman scope)

| Idea | Result |
|---|---|
| Hand-designed informative gains (reliability R, motion process or measurement noise, prototype gate, anchor signals) | none beats a constant gain on dev |
| Hard score gates (`22`, `23`) | trade tracking against false positives; fail per sequence |
| Kalman adaptive gain beating EMA | not shown on PolypGen or CholecSeg8k |

Diagnostics behind the design: with the ground-truth box on every frame,
frozen MedSAM2 reaches 0.906 Dice on polyp frames, against 0.574 when
tracking; 514 of 1,689 polyp frames are tracked with Dice < 0.1, and in 390
of them the tracker segments a different object. The oracle-gain diagnostic
quantifies the room an informative gain has.

## Caveats

- The held-out C6 section uses the current manifest pipeline
  (`scripts/build_polypgen_manifest.py`, numeric frame order); every other
  section reports archived runs from the older pipeline, which the current one
  reproduces. Any run made with a manifest built before commit `c8df175`
  (alphabetical frame order) is invalid.
- The protocol (a) dev analyses (error accumulation, camera motion, the
  reappearance and illumination buckets) use metrics defined after earlier
  results.
- The all-23-sequence results informed later designs; protocol (a) keeps C6
  for one run per frozen method, but C6 values of runs 01 / 02 / 10 had been
  seen in all-sequence tables before the split. C6 is 8 short videos, so it
  cannot test long-term error accumulation.
- One proxy dataset plus CholecSeg8k (laparoscopy, easy for every method),
  one seed. The adenoid data is the clean test of a frozen design.
- An earlier inference bug on MPS (MedSAM2's non-blocking offload read before
  the copy finished) corrupted earlier results; fixed in `f96742d` (state
  stays on device on non-CUDA). All numbers above are after the fix.
- Missing baselines: SAMURAI (Kalman filter on boxes), SAM2Long, DAM4SAM,
  EMA-SAM, TinySAM 2.

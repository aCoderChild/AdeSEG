# Experiment record

PolypGen is a proxy dataset for the adenoid videos. All inference uses the tight
box of the first non-empty ground-truth mask as the prompt, MedSAM2 frozen, one
seed. Run outputs are on Google Drive under `AdeSEG/outputs/` (layout at the end).

**Reproducibility rule.** Every PolypGen and CholecSeg8k table marked
*(regenerated)* is rebuilt from the saved per-frame CSVs by
`scripts/analyze_runs.py` (outputs in `polypgen/update_rules_new/c6/analysis/`,
`polypgen/c6/analysis/` and `polypgen/all23/analysis_{all23,dev}/` on Drive) and was checked against the earlier
numbers. Results marked *(archived)* come from code that has since been removed;
their numbers are kept as a record but cannot be regenerated from this
repository; results of the removed hand-set Kalman and EMA memories are
regenerated from saved CSVs but their code is only at commit `fa3df57`. Two
aggregations appear and are always named: **per-sequence** (mean
over sequences of the per-sequence mean; used for all paired tests) and
**pooled** (mean over frames).

## What the method is, as run

```text
frame t ──► MedSAM2 image encoder (frozen) ──► memory attention (frozen) ──► mask decoder (frozen)
                                                 reads [prompt memory | state (| last memory frame)] + object pointers
measure:  C_t = MedSAM2 memory encoder(F_t, mask_t)
update:   state ← update rule(state, C_t)
```

`modeling/recurrent_memory.py` replaces MedSAM2's 7-frame bank with the prompt
frame's memory plus one recurrent state (64 × 32 × 32). Two update rules:

- **Modified RKN** (`modeling/kalman_memory.py`, `modeling/recurrent_memory.py`;
  under training): the official `KalmanFilter.step` of
  `external/RecursiveKalmanNet/Algo` on every memory element (F = H = 1), with each
  update weighted by the probability β that the measurement is the prompted
  object (probabilistic data association: `x = x⁻ + βK(z − x⁻)`,
  `P = βP⁺ + (1 − β)P⁻ + β(1 − β)(Kν)²`), so an absent or unreliable frame only
  predicts. Four official `GRUNetwork`s learn presence p (MedSAM2's object score
  corrected inside its decoder by the YOLOv8n detector's confidence and box IoU
  with the mask), reliability ρ (expected IoU of the tracker's mask, from
  detector confidence, box IoU and pointer similarity to the prompt), Q per pixel
  (image-feature change) and R per pixel (mask probability, detector box,
  distance to the prompt memory). β = p·ρ; the frame's object pointer is gated by
  ρ. Re-detection: when the detector (confidence ≥ 0.5) disagrees with the mask
  (box IoU < 0.5, or the gate is closed), the frame is decoded from the detector
  box on memory-free features and that mask is output and written (β = its
  presence × detector confidence). The trigger was checked on the training
  videos only (fires on 39% of polyp frames, 65% of them lost; catches 82% of
  lost frames; fires on 3.4% of empty frames). The state is read at
  its effective age. No input compares the new memory with the state. Training
  (through frozen MedSAM2, 24-frame clips, half containing an absence): later
  frames' mask loss + presence BCE weighted by what a wrong gate costs
  (Dice − 0.5·FP) + soft BCE of ρ toward the tracker mask's IoU + 0.02 ×
  `GaussianLikelihoodLoss` against the carried ground-truth memory; checkpoints
  selected on held-out polyp-frame Dice − 0.5 × empty-frame FP. Inference uses
  past frames only.
- **RDE-VOS** (`modeling/rde_memory.py`): the key branch of the official
  `MemCrompress` (`SAM` = 3D non-local + 3D ASPP, then the 2x3x3
  `compress_key`), with official names and initialisation:
  `RDE_t = compress_key(SAM([RDE_{t−1}, C_t]))`. Memory is read and written as in
  the official `two-frames-compress` mode with `mem_every` 3: the prompt memory
  (twice, as in `MemoryBank.match_memory`), the RDE and the last memory frame,
  with the RDE rewritten on frames with index divisible by 3. Untrained, the RDE
  slot is a random compression; the prompt and last-frame slots are MedSAM2's
  own memories.

All runs use no hole filling: MedSAM2 fills small mask holes only when its
compiled `_C` extension is installed, which it was not on the machines used, and
`modeling/medsam2.py` sets `fill_hole_area = 0` so results do not depend on the
machine. The refactor into `recurrent_memory.py` reproduces the archived RDE
C6 run and native MedSAM2 frame for frame (object-score difference 0).

**Memory and speed** (`scripts/infer.py` records both in `setup.json`; Apple
M-series MPS, propagation only; native from the earlier efficiency rerun in
`polypgen/c6/efficiency/`, RDE-VOS from its C6 run, separate sessions):

| | Retained memory at the end of a video | Frames per second |
|---|---|---|
| native MedSAM2 | grows with the video: 4.5–8.1 MiB on C6 (mean 6.2), ~0.13 MiB per frame | 20.2 |
| RDE-VOS | 1.000 MiB, fixed (float32 prompt memory, position code, RDE, last memory frame) | 15.2 |

Native stores every frame's memory but attends to at most 7; those 7 bf16 maps
are 7 × 64 × 32 × 32 × 2 B = 0.875 MiB, so the recurrent memory is not
smaller than a trimmed bank; its advantage is a fixed size. It also keeps the
previous frame's image features (1 MiB) for the motion diagnostic.

## Data protocol

| Role | Source |
|---|---|
| Development (protocol (a)) | `sequenceData/positive` seq1–15, centres C1–C5 (13 sequences with a polyp, 1,640 propagated frames) |
| Test (protocol (a)) | seq16–23, centre C6 (8 sequences, 372 propagated frames, 56 empty) |
| First protocol | all 23 sequences as the test set (2,012 propagated frames: 1,689 polyp, 323 empty) |

`scripts/build_polypgen_manifest.py` writes these splits as `dev` and `test`.
`scripts/train_memory.py` trains on natural clips; on PolypGen it holds out dev
videos for validation (`--train_split dev --val_videos seq13 seq14 seq15`), with a
tight ground-truth box on frame 0 as at inference.

**Archived training runs.** Earlier training used pseudo-videos assembled from
PolypGen single frames (`data_C1`..`data_C6`, 1,399 frames after removing 12
duplicates of test frames) and `sequenceData/negativeOnly` (23 sequences, 4,275
frames). That pipeline (`datasets/pseudo_video.py`) has been deleted, and its
runs also used detector observations. Those runs improved pseudo-video
validation but not real-video results; they describe the old setup, not the
current `train_memory.py`.

## How often C6 has been used

C6 is called "held out" below only in the sense that no design choice was
*calibrated* on it after protocol (a) was defined. It was not untouched:

1. The first protocol tested on all 23 sequences, so C6 results of native,
   prompt + 1 frame and Kalman (`polypgen/all23/`) and of earlier memories were seen
   before protocol (a) existed (`8ae385c`).
2. A C6 rerun of the Kalman memory with motion logging (folder since deleted).
3. `6ca458a`: native / EMA / Kalman on C6 with the misordered manifest
   (discarded).
4. `063853e`: the corrected rerun, reported below.
5. Output-presence variants evaluated on C6 during development (no longer part
   of the method).
6. Earlier trained update rules (hand-set-Kalman noise head and a re-implemented
   RDE): two training objectives, RDE steps 200 and 500 and Kalman step 500;
   runs replaced.
7. RDE-VOS at its selected step 0, reported below, and the official-code RKN
   (results discarded; replaced by the modified RKN).

So C6 has been used for several frozen comparisons. The adenoid data will be the
first clean test.

## Training the update rules *(archived, not currently reproducible)*

The following RDE-VOS training description refers to an older run directory that
is no longer present under the canonical Drive output root. It is retained as
historical context only and must not be cited as a current trained-RDE result.
The current code has a matched RDE training path, but a new run is required.

- **RDE-VOS:** the official `BootstrappedCE` (plain CE, then the hardest 15% of
  pixels; warm-up scaled from 20k–70k of 150k iterations to steps 67–233) on
  `RDE_VOS.aggregate` of MedSAM2's mask, plus the official `KLDivLoss` (weight
  10, temperature 1) from the decoder input read from the compressed memory to
  the one read from native MedSAM2's uncompressed bank. Adam, weight decay 1e-7,
  lr 1e-4 (the official 1e-5 is for 150k iterations). Both terms are averaged over
  frames; the official `LossComputer` halves its running total at each frame,
  which would weight frame 1 of a 16-frame clip by 2^-15.

Selection = best held-out per-sequence polyp-frame Dice with empty-frame FP not
above step 0, else step 0. Outputs: `polypgen/update_rules_new/`.

Held-out validation (whole seq13–15 through the inference path), polyp-frame
Dice / empty-frame FP / lost:

| Step | 0 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| RDE-VOS | **0.505** / 0.890 / 0.408 | 0.451 / 0.678 / 0.462 | 0.466 / 0.540 / 0.472 | 0.443 / 0.515 / 0.455 | 0.340 / 0.225 / 0.607 | 0.368 / 0.204 / 0.577 |

Selection keeps step 0. **Training did not help.**

- **RDE-VOS** cut empty-frame false positives (0.89 to 0.20) by losing polyp
  frames (Dice 0.505 to 0.368): with no term protecting present frames, the
  bootstrapped CE rewards suppressing the object. The KL term has median 4e-4, so
  at weight 10 it is about 2% of the loss.

## C6 results *(regenerated)*

`polypgen/c6/native` and `polypgen/update_rules_new/c6/rde`; analysis in
`polypgen/update_rules_new/c6/analysis/`. RDE-VOS is its selected step 0, i.e.
untrained. Per-sequence means over propagated frames. Lost
= polyp frames with Dice < 0.1; reappearance = polyp frames 1–10 after an
absence; high motion = polyp frames with image-feature change > 0.117 (the dev
tercile cut).

| Method | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance | High motion |
|---|---|---|---|---|---|---|
| native (7-frame bank) | 0.617 | 0.599 | **0.486** | 0.283 | 0.462 | 0.563 |
| RDE-VOS (untrained) | **0.623** | **0.638** | 0.533 | **0.237** | **0.542** | **0.616** |

Pooled lost-episode metrics: native lost 0.247 of polyp frames, mean episode 7.1
frames, recovered 0.64, lost until the end 0.36; RDE-VOS 0.199, 3.7, 0.82, 0.18.

Paired Wilcoxon over the 8 sequences (A − B; sequences where A is higher / lower;
sequence-bootstrap 95% interval):

| Comparison | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance |
|---|---|---|---|---|---|
| RDE-VOS − native | +0.006 (5/3), p = 0.46, [−0.079, +0.079] | +0.039 (5/3), p = 0.38, [−0.045, +0.116] | +0.047 (3/1), p = 0.88 | −0.046 (3/4), p = 0.47 | +0.080 (5/2), p = 0.47 |

**Reading.** Nothing differs significantly from native at n = 8. Because RDE-VOS
is untrained, the comparison is between memory layouts, not a learned update
rule: RDE-VOS reads the prompt memory and the last memory frame (its random RDE
slot aside), which roughly matches the native 7-frame bank.

## Modified RKN with detector observations (dev, held-out seq13–15)

Trained on dev seq1–12 (`scripts/train_memory.py --memory_update rkn
--rkn_weight_factor 1.0 --clip_length 24 --absence_fraction 0.5 --nll_weight 0.02
--fp_weight 0.5`, 500 steps x 4 clips, seed 0), validated on whole seq13–15
every 100 steps; checkpoint = max polyp-frame Dice − 0.5 × empty-frame FP on
seq13–15 (so these numbers are optimistic: selection and report use the same
three videos). Per-sequence means; native MedSAM2 on the same videos for
reference. Outputs: Drive `outputs/polypgen/rkn_detector/` (run script,
checkpoints, histories, `validation.json`, per-frame `check_*.json`). C6 not run.

| seq13–15 | Polyp-frame Dice | Empty-frame FP | Lost |
|---|---|---|---|
| native MedSAM2 | 0.393 | 0.681 | 0.558 |
| presence only (`--presence_only`, step 500) | 0.433 | 0.011 | 0.511 |
| full, re-detection gated by presence and ρ (step 500; run overwritten) | 0.458 | 0.000 | 0.478 |
| full, detector-triggered re-detection, untrained (step 0) | 0.797 | 0.878 | 0.107 |
| **full, detector-triggered re-detection (step 500, selected)** | **0.778** | **0.022** | **0.129** |

- The data-association update stopped the copying of earlier versions:
  effective gain 0.53 on right polyp frames, 0.19 on wrong ones, 0.06 on empty
  frames (presence only: 0.61 everywhere).
- Re-detection never fired, in training or on seq13–15: the corrected presence
  closes the gate on 44% of polyp frames (mostly frames where the tracker is
  lost, which its cost-weighted loss makes free to close) and vetoed it, and ρ
  never fell below 0.5 (min 0.62 on seq13–15; lost-frame median 0.66 on the
  training videos too). The full model's gain over presence only came from the
  memory, pointer gating and presence, and mostly from one video (pooled
  Dice seq13 0.256 vs 0.178; seq14 0.265 vs 0.271; seq15 0.854 vs 0.851).
- The re-detection trigger was then changed to the detector's confident
  disagreement alone, set on the training videos (see the method description).
  ρ remains miscalibrated (too high on lost frames); it still weights the
  write and gates pointers.
- With that trigger (`full/`, run on 2026-10-09): 28.5% of polyp frames are
  re-detected (pooled Dice 0.792 on them) and 1.1% of empty frames. The other
  polyp frames, tracked from the memory, reach pooled Dice 0.752 (presence
  only: 0.365), so the corrected memory carries the object forward. Pooled
  polyp Dice 0.763 (seq13 0.741, seq14 0.726, seq15 0.866); 9.4% of polyp
  frames are blank. Effective gain 0.41 on right polyp frames, 0.23 on wrong
  ones, 0.05 on empty frames.
- Most of the Dice gain is already present untrained (step 0: Dice 0.797,
  FP 0.878): it comes from detector-triggered re-detection plus a memory that
  propagates the corrected mask. Training cut the empty-frame FP from 0.878 to
  0.022 at a Dice cost of 0.019. Not yet separated: the same re-detection with
  native MedSAM2's bank or with a fixed-gain memory, and the detector box
  decoded on every frame without memory.

### Pre-registered: re-detection comparisons and the C6 test (written 2026-10-09, before any C6 run)

Script: Drive `outputs/polypgen/rkn_detector/comparisons/eval_methods.py`
(same prompt and scoring as the training validation: tight box of the first
annotated frame, propagated frames only). Methods, all fixed now:

| Method | Memory | Re-detection |
|---|---|---|
| `native` | MedSAM2's 7-frame bank | no |
| `native_redetect` | MedSAM2's 7-frame bank | yes |
| `detector_only` | none: the detector box decoded on memory-free features on every frame with confidence ≥ 0.5, empty otherwise | — |
| `rde` / `rde_redetect` | RDE-VOS (official, untrained, seed 0; its selected step was always 0) | no / yes |
| `kalman_untrained` | modified RKN, untrained (seed 0) | yes |
| `presence_only` | modified RKN, presence-only checkpoint (step 500) | no |
| `kalman_full` | modified RKN, `full/memory_update.pt` (step 500), unchanged | yes |

Re-detection everywhere uses the same trigger (detector confidence ≥ 0.5, box
IoU with the mask < 0.5) and memory-free box decoding.

1. All methods run on dev seq13–15 (the validation videos), then once on C6
   seq16–23. Nothing is changed between the two.
2. Primary C6 comparison: `kalman_full` − `native`, per-sequence polyp-frame Dice
   and empty-frame FP rate (paired Wilcoxon over the 8 sequences; n = 8).
3. Attribution: the learned Kalman memory is said to contribute only if
   `kalman_full` beats both `native_redetect` and `rde_redetect` on per-sequence
   polyp-frame Dice − 0.5 × empty-frame FP on seq13–15 **and** on C6. Otherwise
   the gain is reported as coming from detector re-detection.
4. Reproduction check: `native` on C6 must match the archived native C6 row
   (polyp-frame Dice 0.599, empty-frame FP 0.486, lost 0.283).

### Results of the pre-registered comparisons (run 2026-10-09)

Drive `outputs/polypgen/rkn_detector/comparisons/{dev_seq13-15,c6_seq16-23}/`
(per-frame JSON per method, `run_comparisons.sh`). Per-sequence means; J =
polyp-frame Dice − 0.5 × empty-frame FP. The native rows reproduce the training
validation on seq13–15 and the archived native C6 row exactly (0.599 / 0.486 /
0.283).

| Method | seq13–15 Dice / FP / lost / J | C6 Dice / FP / lost / J |
|---|---|---|
| native | 0.393 / 0.681 / 0.558 / 0.053 | 0.599 / 0.486 / 0.283 / 0.355 |
| native_redetect | 0.810 / 0.773 / 0.099 / 0.423 | **0.774** / 0.562 / **0.104** / 0.493 |
| detector_only | **0.815** / 0.023 / 0.117 / **0.804** | 0.742 / 0.086 / 0.193 / **0.699** |
| rde | 0.408 / 0.955 / 0.523 / −0.069 | 0.575 / 0.544 / 0.316 / 0.303 |
| rde_redetect | 0.805 / 0.920 / **0.093** / 0.345 | 0.761 / 0.587 / 0.118 / 0.468 |
| kalman_untrained | 0.794 / 0.804 / 0.113 / 0.392 | 0.711 / 0.523 / 0.188 / 0.450 |
| presence_only | 0.433 / **0.011** / 0.511 / 0.428 | 0.544 / **0.038** / 0.386 / 0.525 |
| kalman_full | 0.778 / 0.022 / 0.129 / 0.767 | 0.708 / 0.088 / 0.206 / 0.663 |

Paired Wilcoxon over the 8 C6 sequences (A − B; sequences higher / lower):

| Comparison | Dice | FP | J |
|---|---|---|---|
| kalman_full − native (primary) | +0.109 (7/1), p = 0.078 | −0.398 (0/6), p = 0.031 | +0.308 (8/0), p = 0.008 |
| kalman_full − native_redetect | −0.066 (2/6), p = 0.15 | −0.473 (0/6), p = 0.031 | +0.170 (6/2), p = 0.15 |
| kalman_full − rde_redetect | −0.054 (2/6), p = 0.039 | −0.499 (0/5), p = 0.062 | +0.196 (5/3), p = 0.15 |
| kalman_full − detector_only | −0.035 (1/7), p = 0.20 | +0.002 (1/1), p = 1.0 | −0.036 (1/7), p = 0.20 |
| kalman_full − presence_only | +0.164 (7/1), p = 0.016 | +0.050 (2/1), p = 0.50 | +0.138 (6/2), p = 0.11 |
| kalman_full − kalman_untrained | −0.004 (3/5), p = 0.74 | −0.435 (0/5), p = 0.062 | +0.214 (6/2), p = 0.055 |

**Reading.**

- Primary: the full model beats native MedSAM2 on C6 (J +0.31, 8/8 sequences,
  p = 0.008; FP −0.40, p = 0.031; Dice +0.11, p = 0.078).
- The pre-registered attribution rule is met (kalman_full's J is above
  native_redetect's and rde_redetect's on both seq13–15 and C6), but the rule
  was too weak: the J advantage over those two is entirely fewer false
  positives (the detector-corrected presence), while their Dice is higher
  (C6: −0.066 and −0.054 for kalman_full). The rule did not include the
  memory-free detector_only baseline, which is better than kalman_full on both
  splits (C6 Dice 0.742 vs 0.708, J 0.699 vs 0.663, 7 of 8 sequences; not
  significant). So the data do not show that the learned Kalman memory adds
  to segmentation beyond decoding the detector's box; the gain over native
  MedSAM2 comes from the detector (re-detection and presence).
- Training changed false positives, not Dice (vs untrained: Dice −0.004, FP −0.44).
- The memory helps where the detector fails and hurts where it succeeds:
  seq18 Dice 0.58 (detector_only 0.19), seq23 0.49 (detector_only 0.87).
- C6 has now been used for this comparison too; further design choices must
  not be checked on it. The adenoid data remain the clean test.

## Archived: hand-set Kalman and EMA memories *(regenerated from saved CSVs; code removed)*

The hand-set Kalman filter (`K = P⁻/(P⁻ + R)`, `R` switched by MedSAM2's presence
probability, q = 1, r = 0.1, a = 10) and the constant-gain EMA
(`--fixed_gain 0.835 0.266`) were removed; both are at commit `fa3df57`. On C6
(`polypgen/c6/{ema,kalman}`): EMA Dice 0.603, polyp-frame Dice 0.623, empty-frame
FP 0.587, lost 0.268; Kalman 0.602, 0.613, 0.555, 0.284; neither differed
significantly from native or from each other. The sections below compare them
on the first protocol, dev and CholecSeg8k.

### First protocol and dev split *(regenerated)*

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

Kalman − native on dev: Dice +0.020 (7/6), p = 0.45; lost fraction −0.040 (1/4),
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

### Robustness analyses *(regenerated)*

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

### Gain variants *(archived; code removed, Drive folders 21–23 deleted)*

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

### CholecSeg8k *(partly regenerated)*

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
| `--nested_labels` (prompt disc and cup; rim = disc − cup) | **0.917** | **0.924** | **0.019** | **0.024** | **0.98** | **0.98** |

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
| `polypgen/c6/native` | native MedSAM2 on C6 |
| `polypgen/c6/{ema,kalman}`, `analysis`, `efficiency` | archived hand-set Kalman and EMA C6 runs and tables; memory and FPS |
| `polypgen/update_rules_new/rde_trained` | trained RDE-VOS update: checkpoints `kalman_memory*.pt`, `history.jsonl`, `validation.json`, logs |
| `polypgen/update_rules_new/c6/{rde,analysis}` | its C6 run (selected step 0) and the comparison with native |
| `polypgen/update_rules/` | earlier trained update rules (hand-set-Kalman noise head, re-implemented RDE); superseded |
| `polypgen/all23/{native,native_prompt_plus_1frame,kalman,ema}` | archived all-23 runs (first protocol) |
| `polypgen/all23/analysis_all23`, `analysis_dev` | regenerated tables for all 23 sequences and for dev |
| `cholecseg8k/{native,native_prompt_plus_1frame,kalman,ema}` | archived CholecSeg8k runs; `split.json` |
| `refuge/val_nested` | REFUGE val with `--nested_labels` |

Overlay images were removed (rerun `evaluation/temporal.py` without
`--no_overlays` to regenerate them). Deleted as failed or superseded: the dev
gain variants (reliability, motion noise, skip + reliability), the C6 motion
rerun, the oracle and score-gate CholecSeg8k runs, the first-design REFUGE run,
the untrained checkpoint, and the early summary tables. The official-code RKN
runs (`update_rules_new/kalman_trained`, `rkn_ab_trained`, `c6/rkn`) are
discarded and can be deleted. `outputs/README.md` describes each folder.

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
- RDE-VOS was trained by code before the `recurrent_memory.py`
  refactor (same computation; `update_rules_new/code.diff` records the code after
  it). For exact provenance, retrain from a commit.

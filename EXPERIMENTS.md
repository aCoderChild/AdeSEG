# Experiment record

PolypGen is a proxy dataset for the adenoid videos. Unless a section says
otherwise, numbers are on the 23 `sequenceData/positive` sequences (2,012
propagated frames: 1,689 with a polyp, 323 empty), prompted with the tight box
of the first non-empty ground-truth mask, MedSAM2 frozen, one seed. Results are
on Google Drive under `outputs/Kalman/<run>/`.

Two protocols were used, in this order:

1. **All 23 sequences as the test set** (sections "Current architecture" to
   "Training the Kalman update"). Several design choices were informed by these
   sequences (see Caveats).
2. **Protocol (a)** (`datasets/splits/polypgen/protocol_a.json`): dev =
   seq1–15 (centres C1–C5) for every calibration and design choice; test =
   seq16–23 (centre C6, confirmed by frame names), run once per frozen method.
   Section "Protocol (a): the Kalman memory without a detector". A second real-video
   dataset, CholecSeg8k, is in its own section.

## Current architecture: DOK-Mem (detection-observed Kalman memory)

```text
frame t ──► MedSAM2 image encoder (frozen) ─────────────────────────────┐
   │                                                                     ▼
   └──► YOLOv8n detector (training data only) ──► top box b_t, conf d_t   memory attention (frozen) reads
                                                     │                    [prompt anchor | detection slot | S + u·log(1+P)]
            d_t ≥ 0.5 ?  ── yes ──► clean observation: decode b_t on memory-free features (prompt-frame path)
                │                     object score += logit(d_t); memory -> detection slot + Kalman update
                no
                ▼
     mask decoder (frozen) ──► object score s_t, predicted IoU
                ▼
     presence fusion  σ(w·[s_t, IoU, logit d_t, log P̄] + b)  ──► gates the mask output and the memory write
                ▼
     Kalman update:  P⁻ = P + q;  R = (r + a(1−p)) · c(spatial map);  K = P⁻/(P⁻+R);  S ← S + K(C_t − S);  P ← (1−K)P⁻
```

| Component | Code | Learned? |
|---|---|---|
| Kalman memory (prompt anchor, detection slot, mean `S`, per-pixel variance `P`) | `modeling/kalman_memory.py` | noise level hand-set (`q=1`, `r=0.1`, `a=10`, `P0=1`); bounded corrections trainable, used untrained in the main result |
| Detector observations (clean decoding, detection slot, detector log-odds) | `modeling/detector_observation.py` (`DetectorObservationMixin`) | detector trained on training data |
| Presence fusion (4 inputs, 5 weights) | `PresenceFusion` in the same file | yes, cross-fitted (below) |
| Object pointers | `modeling/native_pointers.py` | MedSAM2's own selection; `scripts/verify_native_pointers.py` checks the tokens are identical to native MedSAM2 (max difference 0) |

Kalman step (identity transition, diagonal covariance shared across the 64
channels at each pixel, identity observation model; RKN's update equations):

```text
predict:  P⁻ = P + q
read:     memory attention sees [anchor | detection slot | S + u·log1p(P⁻)] and MedSAM2's object pointers
observe:  C_t = MedSAM2 memory encoder(F_t, mask_t)
update:   R_t = (r + a·(1 − p_t)) · c(D_t − mean(D_t)) · [c(g(logit d_t)) on detector frames]
          K_t = P⁻ / (P⁻ + R_t);   S ← S + K_t(C_t − S);   P ← (1 − K_t)P⁻
```

`c(x) = 10^tanh(x)`. `D_t` is a per-pixel map from `R_net(C_t, normalize(C_t − S),
F_t, mask_t, p_t)`, centred per frame so it only redistributes trust inside a
frame; `g` is a linear function of the detector confidence. Both start at zero
(`c = 1`), so the untrained module is the hand-set filter: a polyp frame writes
about 95% of the memory, an empty one about 17%. The prompt anchor and the
detection slot use MedSAM2's conditioning-frame temporal code
(`maskmem_tpos_enc[num_maskmem−1]`), the state the most-recent-frame code
(`maskmem_tpos_enc[0]`), as in MedSAM2/SAM2-Plus. Retained memory: about 1.0 MiB
per object, constant in video length.

Detector observations: the detector runs on every propagated frame. When its top
box has confidence ≥ 0.5 (56% of propagated frames: 65% of polyp frames, 6.8%
of empty frames), the box is decoded on memory-free features, as on a prompt
frame, so a drifted memory cannot pull the observation off the polyp; the
detector's log-odds are added to the object score; the resulting memory is
written to the detection slot (latest detection only) and into the Kalman state.

Presence fusion: on frames without a confident box, a logistic head replaces the
object score that gates the mask and the memory write:
`σ(w1·s + w2·IoU + w3·logit(d) + w4·log P̄ + b)`.

## Data protocol

Training never reads `sequenceData/positive`.

| Role | Source |
|---|---|
| Positives | `data_C1`..`data_C6` single frames with non-empty masks (1,399 after excluding 12 frames that duplicate test frames, `datasets/splits/polypgen/single_frame_test_overlap.txt`) |
| Absences | `sequenceData/negativeOnly` (23 sequences, 4,275 frames), inpainted polyps, look-away camera motion |
| Test | all 23 `sequenceData/positive` sequences |

Pseudo-videos (`datasets/pseudo_video.py`, `hard` preset): one single frame on a
mean-reverting camera path, 16 frames, gaps 1–80 raw frames, blur/defocus/noise/
JPEG/specular highlights/vignetting, occluders in 40% of clips, and with
probability 0.6 an absence of 2–12 frames (inpainted polyp, look-away, or a
negative-video splice).

Detectors (`scripts/build_detector_dataset.py`, `scripts/train_detector.py`,
YOLOv8n from COCO, 60 epochs, 640 px):

| Detector | Training data | Use |
|---|---|---|
| main | all centres: 1,258 positive images (1,434 boxes) + 1,258 negativeOnly frames; validation 141 / 157 / 167 (mAP@0.5 0.882, P 0.887, R 0.798) | test-time observations |
| fold 0 | centres C2, C4, C6 + half of the negative sequences | detections on C1/C3/C5 clips |
| fold 1 | centres C1, C3, C5 + the other half | detections on C2/C4/C6 clips |

The fold detectors give out-of-fold detections on training pseudo-videos
(cross-fitting), so the presence fusion and the Kalman training see realistic
detector errors rather than a detector that memorised the frames.

Presence-fusion data (`scripts/build_fusion_data.py`): 300 clips per fold run
through the full pipeline exactly as at test time; frames without a confident
box are logged (5,760 frames, 58% with a polyp). Fit
(`scripts/train_presence_fusion.py`): class-balanced logistic BCE with L2 1e-3.
Cross-fold AUROC 0.790 / 0.820 (object score alone 0.762 / 0.784). Weights:
object score 0.123, predicted IoU 1.300, detector logit 0.122, log prior
variance −0.660, bias −0.075.

No hyperparameter, threshold or checkpoint is chosen with test labels.

## Main results

Propagated frames (after the prompt). Dice on an empty frame is 1 for an empty
prediction and 0 otherwise. 95% intervals resample sequences.

| Run | Dice [95% CI] | IoU | Dice, polyp frames | Empty-frame FP | Polyp frames detected | Presence AUROC | ≥10 frames after reappearance | Memory | FPS |
|---|---|---|---|---|---|---|---|---|---|
| native MedSAM2 (`native_gtbox`) | 0.507 [0.387, 0.661] | 0.466 | 0.537 | 0.650 | 0.906 | 0.713 | 0.452 | grows | 13.3 |
| native, prompt + 1 frame (`native_k1`) | 0.513 [0.385, 0.671] | 0.469 | 0.552 | 0.687 | 0.926 | 0.656 | 0.466 | grows | 15.7 |
| Kalman memory alone (`kalman_untrained`) | 0.531 [0.400, 0.693] | 0.487 | 0.574 | 0.693 | 0.927 | 0.665 | 0.519 | 0.88 MiB | 17.0 |
| native + detector observations | 0.668 [0.549, 0.792] | 0.611 | 0.731 | 0.656 | 0.952 | 0.855 | 0.697 | grows | 12.8 |
| native k1 + detector observations | 0.655 [0.533, 0.787] | 0.599 | 0.724 | 0.706 | 0.970 | 0.842 | 0.690 | grows | 13.9 |
| Kalman + detector observations (`kalman_detobs_full`) | 0.658 [0.544, 0.779] | 0.600 | 0.724 | 0.690 | 0.977 | 0.859 | 0.689 | 1.13 MiB | 13.5 |
| native + detector + fusion (`native_detobs_fusion`) | 0.690 [0.583, 0.803] | 0.634 | 0.724 | 0.486 | 0.917 | 0.886 | 0.693 | grows | 12.4 |
| **DOK-Mem** (`kalman_detobs_fusion`) | **0.689 [0.590, 0.796]** | **0.633** | 0.716 | **0.449** | 0.941 | **0.893** | 0.681 | 1.13 MiB | 13.5 |
| DOK-Mem, trained Kalman (`kalman_dok_trained_v2`, absence 1.0, step 200) | 0.689 [0.593, 0.793] | 0.640 | 0.677 | 0.251 | 0.891 | 0.902 | 0.626 | 1.00 MiB | 13.2 |
| DOK-Mem, trained Kalman (absence 0.1, step 100) | 0.698 | 0.646 | 0.687 | 0.245 | 0.895 | — | 0.636 | 1.00 MiB | — |

Memory: native "grows" is what the code stores (36.3 MiB on average). MedSAM2
attends to only 7 frame memories, and one is 64×32×32 bf16 = 0.125 MiB, so a
trimmed bank needs about 0.875 MiB, the same as the Kalman memory
(0.88–1.13 MiB). The Kalman memory is constant in video length but saves no
memory over a trimmed bank.

Per-sequence paired tests (21 sequences with a polyp, 17 with empty frames;
Wilcoxon signed-rank; bootstrap 95% interval of the mean difference):

| Comparison | Metric | Difference [95% CI] | Better / worse | p |
|---|---|---|---|---|
| DOK-Mem vs native MedSAM2 | Dice | +0.084 [+0.002, +0.170] | 16 / 5 | 0.029 |
| | IoU | +0.080 [+0.007, +0.158] | 16 / 5 | 0.018 |
| | empty-frame FP | −0.266 [−0.435, −0.113] | 10 / 1 | 0.004 |
| | Dice, polyp frames | +0.054 [−0.066, +0.167] | 13 / 8 | 0.393 |
| DOK-Mem vs native + detector + fusion | Dice | −0.001 [−0.015, +0.011] | 12 / 9 | 1.000 |
| | empty-frame FP | −0.075 [−0.183, +0.027] | 7 / 4 | 0.248 |
| Kalman alone vs native MedSAM2 | Dice | +0.007 [−0.024, +0.031] | 14 / 7 | 0.137 |
| Kalman alone vs native prompt + 1 frame | Dice, polyp frames | +0.027 | 17 of 21 better | 0.014 |

## Ablations and robustness (DOK-Mem, paired against the full model)

| Variant | Dice | Empty-frame FP | Paired vs full |
|---|---|---|---|
| without presence fusion (`kalman_detobs_full`) | 0.658 | 0.690 | Dice −0.029 [−0.051, −0.008], p = 0.021; FP +0.257, 11/17 worse, 0 better, p = 0.001 |
| without the Kalman variance in the fusion (`_novar`) | 0.681 | 0.514 | Dice −0.008 [−0.013, −0.003], p = 0.011; FP +0.071 [+0.021, +0.136], p = 0.012 |
| without clean decoding and detection slot, no fusion (`kalman_detobs_noclean`) | 0.606 | 0.693 | vs `kalman_detobs_full` 0.658 (pooled) |
| detector threshold 0.25 | 0.672 | 0.517 | Dice −0.004, p = 0.494 |
| detector threshold 0.75 | 0.675 | 0.421 | Dice +0.005, p = 0.812; FP −0.083, p = 0.046 |
| detector on every 2nd frame | 0.678 | 0.446 | Dice −0.010, p = 0.229 |
| detector on every 5th frame | 0.661 | 0.440 | Dice −0.029, p = 0.076 |
| detector on every 10th frame | 0.643 | 0.437 | Dice −0.036, p = 0.082 |

Presence fusion on native memory: Dice +0.017 (p = 0.15), empty-frame FP
−0.213 (10/17 better, 0 worse, p = 0.002).

Detector baselines (YOLO on file paths, frozen MedSAM2 image predictor on the top
box; `gate2_results.csv`): tracker (Kalman alone) 0.531, detector box on every
confident frame without memory 0.598, switch to the detector mask when
confident 0.616, Kalman + detector observations 0.658. Detector observations vs
detector alone: +0.090 per sequence, 19/21 better, p = 0.003; vs switch: +0.020,
12/21, p = 0.393.

## Training the Kalman update

Setup (`scripts/train_kalman.py`, `training/kalman_trainer.py`): MedSAM2's
`SAM2Train`, `MultiStepMultiMasksAndIous` and `BatchedVideoDatapoint`; training
clips run exactly as at test time, with out-of-fold detector observations, clean
decoding and the detection slot. Every 10th single frame and every 10th
negative sequence are held out for 64 validation clips. 500 steps × 4 clips,
AdamW, learning rate 1e-3.

Objective on propagated frames:

```text
polyp frames, presence gate open:  20 · focal(mask) + 1 · Dice(mask)       # MedSAM2 weights; IoU term off (frozen head)
polyp frames:                      1 · BCE(object score, 1)
empty frames:                      w_absence · BCE(object score, 0)
```

Mask losses skip polyp frames whose gate closed (their logits are −1024, a
presence error left to the BCE). Training clips decoded entirely from detector
boxes carry no gradient for the memory and are skipped.

| Attempt | Trainable | Objective | Validation Dice (untrained 0.580) | Test |
|---|---|---|---|---|
| first (before detector observations) | unbounded Q/R heads | absence 0.1 + distillation to native | — | Dice −0.037 vs untrained (p = 0.042) |
| v1, lr 1e-4 | bounded Q/R corrections | absence 1.0 | 0.5796 at step 100 (flat) | stopped |
| v1, lr 1e-3 | bounded Q/R corrections | absence 1.0 | 0.566 at step 100; gains on all frames fell (0.82 → 0.27) | stopped |
| **v2** (`kalman_dok_trained_v2`) | per-frame-centred spatial map + detector trust; noise level fixed | absence 1.0 | 0.612 / **0.623** / 0.617 / 0.615 / 0.614 at steps 100–500 | step 200 selected: Dice −0.002 [−0.013, +0.011] vs untrained, p = 0.517 |

v2 validation, steps 0 → 200 → 500: polyp-frame Dice 0.651 → 0.629 → 0.572,
empty-frame FP 0.568 → 0.390 → 0.300. On the test set the trained model also
trades polyp frames for empty ones (polyp-frame Dice 0.716 → 0.677, empty-frame
FP 0.449 → 0.251), so Dice is unchanged. By construction about 28% of clip frames
are empty (an absence in 60% of clips, 2–12 of 15 propagated frames), against
16% of test frames, which favours that trade during training and validation.

Learning only what single-frame pseudo-videos can show (where in a frame to trust
the observation, how much to trust the detector) with the overall write level
fixed is what made training stop hurting; it did not yet make it help on real
video.

Absence weight 0.1 (`22_dokmem_trained_absence0.1`), checkpoint selection fixed
beforehand: the highest validation polyp-frame Dice among checkpoints whose
empty-frame FP is not above the untrained model's.

| Step | Validation Dice | Polyp-frame Dice | Empty-frame FP |
|---|---|---|---|
| 0 (untrained) | 0.5801 | 0.6507 | 0.568 |
| 100 | 0.5804 | 0.6510 | 0.568 |
| 200 | 0.5701 | 0.6543 | 0.606 |
| 300 | 0.5607 | 0.6543 | 0.635 |
| 400 | 0.5489 | 0.6492 | 0.661 |

The rule selects step 100, indistinguishable from untrained on validation. Its
test run (`23_dokmem_trained_absence0.1_step100`, fusion refitted): Dice 0.698,
IoU 0.646, polyp-frame Dice 0.687, empty-frame FP 0.245. Paired against
untrained DOK-Mem: Dice +0.006 (p = 0.37), IoU +0.008 (p = 0.13), polyp-frame
Dice −0.016 (p = 0.45), empty-frame FP −0.077 (p = 0.062). Against native
MedSAM2: Dice +0.090 [+0.012, +0.176], 17/21, p = 0.009; IoU +0.088, p = 0.013.

Control: untrained DOK-Mem with a fusion head refitted by the same (current)
code (`24_dokmem_untrained_refit_fusion`) reproduces untrained DOK-Mem exactly
(Dice 0.6892, empty-frame FP 0.4489). So the step-100 difference comes from the
100 training steps, not from refitting the fusion; it is a trade-off (fewer
false positives, worse polyp frames), not a significant Dice gain.

Summary: with single-frame pseudo-videos, training moves the model along the
false-positive / polyp-tracking trade-off (absence 1.0 one way, absence 0.1 the
other) but does not improve both.

## Protocol (a): the Kalman memory without a detector

Here the memory is the only difference from native MedSAM2 (no detector).
Untrained Kalman checkpoint: `20_dokmem_untrained/kalman_memory_untrained.pt`.

All 23 sequences, propagated frames (computed before protocol (a) was adopted):

| Run | Dice | IoU | Polyp-frame Dice | Empty-frame FP | Polyp frames detected | ≥10 frames after reappearance |
|---|---|---|---|---|---|---|
| `01_native_medsam2` | 0.5072 | 0.4661 | 0.5373 | 0.6502 | 0.9065 | 0.4518 |
| `02_native_medsam2_prompt_plus_1frame` | 0.5132 | 0.4692 | 0.5515 | 0.6873 | 0.9260 | 0.4660 |
| `10_kalman_memory` | 0.5313 | 0.4867 | 0.5743 | 0.6935 | 0.9266 | 0.5193 |
| `11_kalman_memory_skip_absent` | 0.5255 | 0.4805 | 0.5763 | 0.7399 | 0.9408 | 0.5182 |
| `12_kalman_memory_skip_absent_gate` (prototype gate 0.4446) | 0.5312 | 0.4839 | 0.5872 | 0.7616 | 0.9248 | 0.5463 |
| `03_native_medsam2_fusion` (detector-free presence fusion) | 0.5526 | 0.5125 | 0.5506 | 0.4365 | 0.7217 | — |
| `13_kalman_memory_skip_absent_fusion` | 0.5451 | 0.5043 | 0.5339 | 0.3963 | 0.6998 | — |

Absence skip: a frame whose presence gate is closed is a missing measurement
(predict only). Without it, ten consecutive empty frames leave only 7% of the
polyp's memory (gain rises from 0.10 to 0.27 per empty frame).

Paired: Kalman + skip vs native, polyp-frame Dice +0.023 [+0.006, +0.046],
15/21, p = 0.026 (vs prompt + 1 frame +0.031, p = 0.003), overall Dice not
significant. Detector-free presence fusion on native memory: Dice +0.026
[+0.009, +0.045], p = 0.017; IoU +0.026, p = 0.013; FP −0.19, p = 0.014.
Kalman + skip + fusion vs native + fusion: Dice −0.008, p = 0.73.

**Error accumulation** (lost episode = consecutive polyp frames with Dice < 0.1):

| All 23 sequences | Polyp frames lost | Mean episode (frames) | Episodes recovered | Lost until the end |
|---|---|---|---|---|
| native | 0.374 | 25.3 | 0.60 | 0.36 |
| native, prompt + 1 frame | 0.326 | 19.0 | 0.55 | 0.31 |
| Kalman | 0.304 | 16.1 | 0.75 | 0.22 |
| Kalman + skip | 0.300 | 14.9 | 0.68 | 0.21 |
| Kalman + skip + gate | 0.285 | 10.9 | 0.80 | 0.18 |

Lost-frame fraction, Kalman + skip vs native −0.032 (8/2, p = 0.037), vs prompt +
1 frame −0.033 (9/0, p = 0.004). These metrics were defined after earlier
results; all frames more than 100 frames after the prompt are in dev sequences
(seq5, seq11–15), and the C6 videos are short.

**Dev (seq1–15) vs C6 test, frozen plain Kalman memory:**

| | Dice | Polyp frames lost | Mean episode | Recovered | Lost until end |
|---|---|---|---|---|---|
| dev native | 0.481 | 0.403 | 39.6 | 0.57 | 0.36 |
| dev prompt + 1 frame | 0.488 | 0.343 | 23.6 | 0.60 | 0.25 |
| dev Kalman | 0.514 | 0.318 | 19.9 | 0.82 | 0.14 |
| C6 native | 0.621 | 0.247 | 7.1 | 0.64 | 0.36 |
| C6 prompt + 1 frame | 0.623 | 0.253 | 8.9 | 0.44 | 0.44 |
| C6 Kalman | 0.609 | 0.244 | 7.7 | 0.60 | 0.40 |

C6, Kalman vs native: Dice −0.014 (7/8 sequences better, one collapse,
p = 0.20); lost fraction +0.001 (p = 0.88). The dev error-accumulation benefit
does not show on the short C6 videos.

**Camera motion.** Motion = per-pixel change of MedSAM2's image features since
the previous frame (1 − cosine), logged by the Kalman runs. Dev terciles (cuts
0.033 / 0.117), polyp frames:

| Dev | Low motion: Dice / lost | Medium | High |
|---|---|---|---|
| native | 0.612 / 0.300 | 0.541 / 0.376 | 0.400 / 0.538 |
| prompt + 1 frame | 0.614 / 0.221 | 0.558 / 0.326 | 0.429 / 0.484 |
| Kalman | 0.628 / 0.210 | 0.581 / 0.303 | 0.466 / 0.444 |

High-motion polyp Dice, Kalman vs native: +0.033, 9/12, p = 0.052. On C6 with
the dev cut fixed beforehand (high > 0.117): native 0.539, Kalman 0.559,
+0.016, 6/8, p = 0.25; lost fraction unchanged. Camera motion is a major failure
cause for native MedSAM2 on both splits.

**Is the Kalman gain responsible?** The untrained gain is nearly constant on
present frames (PolypGen dev mean 0.835, std 0.144; CholecSeg8k 0.911, std
0.033): with constant q and presence-only R the variance converges and a
steady-state Kalman filter is an exponential moving average. Ablation
`19_kalman_fixed_gain_ema`: constant gains measured on dev (present 0.835,
absent 0.266):

| | dev Dice | dev lost | dev mean episode | dev high-motion Dice | C6 Dice | C6 lost |
|---|---|---|---|---|---|---|
| Kalman | 0.514 | 0.318 | 19.9 | 0.464 | 0.609 | 0.244 |
| moving average | 0.504 | 0.350 | 26.7 | 0.460 | 0.607 | 0.231 |

Kalman vs moving average, dev: lost −0.012 (p = 0.63), Dice +0.004 (p = 0.41),
high-motion −0.004 (p = 0.79); C6 no difference. Pre-set rule (Kalman must beat
it on dev lost fraction and high-motion Dice): not met.

Attempts to make the gain informative (each judged on dev by a rule fixed
beforehand; none went to C6):

| Gain change | Dev result | Verdict |
|---|---|---|
| R from calibrated reliability ρ = σ(0.423·s − 2.619) of the object score s (LOSO AUROC 0.745) | Dice 0.493, lost 0.348 | worse than Kalman + skip |
| Process noise ∝ per-pixel feature change (d_ref 0.0958) | Dice 0.490, lost 0.339, high-motion 0.437 | worse |
| Measurement noise ∝ feature change (`20_dev_kalman_motion_measurement`) | Dice 0.502, lost 0.353, high-motion 0.457 (vs Kalman −0.012, 11/12 worse, p = 0.009) | worse |
| Prototype innovation gate | real-video AUROC 0.43 (gate 0.085 on real dev vs 0.4446 on pseudo-video) | uninformative |
| Prompt-anchor distances (memory, pointer, image) | pooled AUROC 0.76 / 0.63 / 0.74; inverted in seq13; no gain over the object score in LOSO | uninformative |
| Object score (raw logit) | AUROC 0.887 overall, 0.88–0.98 within sequences | the best available signal |

**Oracle gain** (`21_dev_kalman_oracle_gain`, diagnostic only): a frame is
written only if it matches the ground truth (IoU ≥ 0.5 or correctly empty),
otherwise K = 0. Dev: Dice 0.614, IoU 0.559, polyp-frame Dice 0.685, empty-frame
FP 0.749, lost 0.183, mean episode 6.1. Against the moving average: Dice +0.056
(8/5, p = 0.19), lost −0.081; against native: Dice +0.072 (10/3, p = 0.048).
An informed gain is worth about +0.11 pooled Dice and half the lost frames over
a constant gain on PolypGen; it is worth nothing on CholecSeg8k (below).

**Measurement-validation gain** (object score below τ ⇒ K = 0; τ = 5.74 from
dev by Youden's J = 0.494, keeps 75% of correct frames, rejects 74% of wrong
ones):

| Dev | Dice | IoU | Polyp-frame Dice | Empty-frame FP | Lost | Mean episode |
|---|---|---|---|---|---|---|
| `22_kalman_score_gate` (all frames) | 0.527 | 0.479 | 0.613 | 0.914 | 0.277 | 14.1 |
| `23_kalman_score_gate_present_only` (0 < s < τ) | 0.497 | 0.454 | 0.556 | 0.809 | 0.336 | 13.2 |

The all-frames gate recovers 21% of the oracle's pooled Dice headroom and 44% of
its lost-frame headroom over the moving average, but per sequence it is not
better (Dice −0.022, p = 0.74; lost ≈0, p = 0.84) because it also stops writing
empty frames, so the memory keeps the polyp and empty-frame FP rises to 0.91.
The present-only variant loses the tracking gain too. Both fail the rule.

`25_kalman_score_gate_presence_fusion_dev` (gate + detector-free fusion) is
**invalid**: the gate compared the fused score with τ calibrated on raw scores,
so all 1,640 frames were rejected; its fusion data were also generated without
the gate. Fixed in `09287e8` (the gate always uses the raw object score;
`build_fusion_data.py --score_gate`). Not rerun yet.

## CholecSeg8k (second real-video dataset)

`scripts/prepare_cholecseg8k.py` converts CholecSeg8k (CC BY-NC-SA 4.0; 101
clips × 80 consecutive laparoscopic frames, 17 Cholec80 videos, dense masks)
into the sequence layout for one target class. Gallbladder (watershed value
22, checked on the files): 88 sequences from 16 videos, 191 of 7,040 frames
empty; split by whole video: dev 60 sequences, test 28 (6 videos;
`data/CholecSeg8k_gallbladder/split.json`). Runs under
`outputs/Kalman/cholecseg8k/`.

| | dev Dice | dev IoU | test Dice | test IoU |
|---|---|---|---|---|
| native | 0.916 | 0.868 | 0.856 | 0.805 |
| prompt + 1 frame | 0.926 | 0.877 | 0.872 | 0.822 |
| Kalman | 0.925 | 0.877 | 0.873 | 0.823 |
| moving average (gains 0.911 / 0.284 from CholecSeg8k dev) | 0.926 | 0.877 | 0.875 | 0.825 |
| oracle gain (dev only) | 0.920 | 0.871 | — | — |
| score gate τ = 5.74 | 0.915 | — | 0.878 | — |

Kalman vs native: dev +0.009 (38/60, p = 0.22), high-motion +0.017 (p = 0.054);
test +0.016 (p = 0.26). Kalman vs prompt + 1 frame: dev −0.0004 (19/60 better,
p = 0.007). Kalman vs moving average: no difference. Tracking almost never
fails here (lost polyp frames 0.2–1.5%), so no gain policy has room to help.

**Across both datasets:** a 1-slot memory beats MedSAM2's 7-frame bank
(consistently, mostly not significantly per test); the Kalman gain as built
equals a moving average; an informed gain has large headroom only where the
tracker drifts (PolypGen), and no realizable gain has captured it without a
false-positive cost.

## Drive folders

| Folder | Run |
|---|---|
| `01_native_medsam2`, `02_native_medsam2_prompt_plus_1frame`, `03_native_medsam2_fusion` | native baselines |
| `10`–`13_kalman_memory*` | Kalman memory without a detector |
| `14`, `16`, `17`, `20`, `21` (`*_dev*`) | dev-only Kalman variants and diagnostics |
| `18_test_kalman_motion_logged` | C6 rerun of the frozen Kalman memory with motion logging |
| `19_kalman_fixed_gain_ema`, `22`/`23_kalman_score_gate*`, `25_*` (invalid) | gain ablations |
| `20_dokmem_untrained`, `21_dokmem_trained_absence1.0`, `22_dokmem_trained_absence0.1`, `23_dokmem_trained_absence0.1_step100`, `24_dokmem_untrained_refit_fusion` | DOK-Mem (with detector) |
| `detectors`, `cholecseg8k/`, `summary_before_renaming` | detectors, CholecSeg8k runs, old tables |

The numbering overlaps between the two families (`20`–`24`); the full names
are unambiguous. Ablation and baseline folders from the first protocol
(stride, threshold, no-variance, no-clean, native + detector) were deleted;
their numbers are kept above.

## Negative results

| Idea | Result |
|---|---|
| Re-prompting MedSAM2 with a box from the tracker's own mask | +0.0025 polyp-frame Dice (10/21 sequences), no gain: tracking errors are wrong-object, not mask shape |
| Temporal presence filter (scalar Kalman filter on presence log-odds) | failed the pre-set pseudo-video rule (Dice 0.607 vs 0.608 and 0.677 vs 0.682 on the two folds); removed, not tested |
| Presence verifier from tracker signals alone | leave-one-sequence-out at chance |
| Two memory paths chosen by predicted IoU | 0.531 → 0.511 |
| Prototype re-localisation in frozen features | lands on the polyp in 53% of polyp frames |
| Training the Kalman update on single-frame pseudo-videos (4 variants) | no Dice gain on test; moves the FP / tracking trade-off only |
| Hand-designed informative gains (reliability R, motion process or measurement noise, prototype gate, anchor signals) | none beats a constant gain on dev |
| Score gates (`22`, `23`) | trade tracking against false positives; fail per sequence |
| Kalman memory with the detector (DOK-Mem) vs native memory with the detector | tie (Dice −0.001, p = 1.0) |

Diagnostics behind the design: with the ground-truth box on every frame, frozen
MedSAM2 reaches 0.906 Dice on polyp frames, against 0.574 when tracking; 514 of
1,689 polyp frames are tracked with Dice < 0.1, and in 390 of them the tracker
segments a different object.

## Verification

- `scripts/verify_kalman_parity.py --perturb`: training model vs inference
  predictor with perturbed learned layers: mask IoU 1.0, object-score and gain
  errors 0.
- `scripts/verify_native_pointers.py`: object-pointer tokens identical to native
  MedSAM2 (count and values, 50 frames).
- Code changes that should not change behaviour were checked by reproducing
  DOK-Mem on seq20 / seq23 (0.3649 / 0.6345) exactly, and the plain Kalman runs
  (run 16 reproduces run 10 on dev, run 18 on C6). Every new Kalman option is
  off by default and was checked to leave the default output unchanged.

## Caveats

- Clean decoding and the detection slot were designed after inspecting failures
  on two test sequences (seq20, seq23). They contain no tuned value, but the
  design was informed by the test set. The presence fusion and the training
  were not.
- The all-23-sequence results informed later designs; protocol (a) keeps C6
  for one run per frozen method, but C6 values of runs 01/02/10 had been seen
  in all-sequence tables before the split. C6 is 8 short videos, so it cannot
  test long-term error accumulation.
- The protocol (a) dev analyses (error accumulation, camera motion) use metrics
  chosen after earlier results.
- One proxy dataset plus CholecSeg8k (laparoscopy, easy for every method), one
  seed. The adenoid data is the clean test of a frozen design.
- An earlier inference bug on MPS (MedSAM2's non-blocking offload read before
  the copy finished) corrupted earlier results; fixed in `f96742d` (state stays
  on the device on non-CUDA). All numbers above are after the fix. An earlier
  feasibility check fed YOLO RGB arrays instead of BGR; the detector baselines
  above use file paths.
- Missing baselines: SAMURAI (Kalman filter on boxes), SAM2Long, DAM4SAM,
  EMA-SAM.
- Google Drive for desktop intermittently fails reads (`Operation canceled`)
  when the disk is nearly full; analyses read local copies under
  `~/Library/Caches/adseg_work`.

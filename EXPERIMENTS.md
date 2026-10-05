# Experiment record

PolypGen is a proxy dataset for the adenoid videos. All numbers below are on
the 23 `sequenceData/positive` sequences (2,012 propagated frames: 1,689 with a
polyp, 323 empty), prompted with the tight box of the first non-empty
ground-truth mask, MedSAM2 frozen, one seed. Results are on Google Drive under
`outputs/Kalman/<run>/`.

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
| DOK-Mem, trained Kalman (`kalman_dok_trained_v2`) | 0.689 [0.593, 0.793] | 0.640 | 0.677 | 0.251 | 0.891 | 0.902 | 0.626 | 1.00 MiB | 13.2 |

Native "Memory" is reported as stored by the code (36.3 MiB); MedSAM2 attends
to 7 frame memories, so that number overstates what a trimmed bank needs.

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

Running: `kalman_dok_aw0.1` (absence weight 0.1), selected with a rule fixed
beforehand: the highest validation polyp-frame Dice among checkpoints whose
empty-frame FP is not above the untrained model's, across this run and v2;
tested once only if a trained checkpoint is selected.

## Negative results

| Idea | Result |
|---|---|
| Re-prompting MedSAM2 with a box from the tracker's own mask | +0.0025 polyp-frame Dice (10/21 sequences), no gain: tracking errors are wrong-object, not mask shape |
| Temporal presence filter (scalar Kalman filter on presence log-odds) | failed the pre-set pseudo-video rule (Dice 0.607 vs 0.608 and 0.677 vs 0.682 on the two folds); removed, not tested |
| Presence verifier from tracker signals alone | leave-one-sequence-out at chance |
| Two memory paths chosen by predicted IoU | 0.531 → 0.511 |
| Prototype re-localisation in frozen features | lands on the polyp in 53% of polyp frames |

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
  DOK-Mem on seq20 / seq23 (0.3649 / 0.6345) exactly.

## Caveats

- Clean decoding and the detection slot were designed after inspecting failures
  on two test sequences (seq20, seq23). They contain no tuned value, but the
  design was informed by the test set. The presence fusion and the training
  were not.
- One proxy dataset, 21 sequences with a polyp, one seed. The adenoid data is
  the clean test of the frozen design.
- An earlier inference bug on MPS (MedSAM2's non-blocking offload read before
  the copy finished) corrupted earlier results; fixed in `f96742d` (state stays
  on the device on non-CUDA). All numbers above are after the fix. An earlier
  feasibility check fed YOLO RGB arrays instead of BGR; the detector baselines
  above use file paths.
- Missing baselines: SAMURAI (Kalman filter on boxes), SAM2Long, DAM4SAM.

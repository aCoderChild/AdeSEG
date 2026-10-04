# Experiment record

## Current architecture: Kalman spatial memory

MedSAM2 keeps a FIFO memory bank: the prompt frame plus the last six frame
memories. `modeling/kalman_memory.py` replaces it with two constant-size slots
read by the frozen memory attention, while the native object-pointer history is
unchanged:

- **anchor slot**: the prompt-frame memory, never overwritten, with
  MedSAM2's conditioning-frame temporal code (`maskmem_tpos_enc[num_maskmem-1]`);
- **Kalman state**: a mean memory map `S` and a per-pixel variance `P`, with
  the most-recent-frame temporal code (`maskmem_tpos_enc[0]`).

For frame `t`, with `C_t` the MedSAM2 memory-encoder candidate, `F_t` the image
feature, `M_t` the predicted low-resolution mask, `p_t = sigmoid(object score)`
and `dt` the raw-frame gap, each step runs predict -> read -> observe -> update,
as in RKN (`_predict`, `_update`) and KalmanNet (`step_prior`,
`step_KGain_est`, `KNet_step`):

```text
predict:  Q_t     = softplus(Q_net(proj F_t, |proj F_t - proj F_{t-1}|, log1p dt))
          P_prior = P_{t-1} + Q_t
read:     memory attention sees S_{t-1} + u * log1p(P_prior)      # u zero-initialized
observe:  decode frame t, C_t = MedSAM2 memory encoder(F_t, mask_t)
update:   R_t = softplus(R_net(C_t, normalize(C_t - S_{t-1}), proj F_t, M_t, p_t)) + a * (1 - p_t)
          K_t = P_prior / (P_prior + R_t)                          # per-pixel gain
          S_t = S_{t-1} + K_t * (C_t - S_{t-1})
          P_t = (1 - K_t) * P_prior
```

Relation to the references: the update is RKN's diagonal update with an
identity observation model (gain `covar / (covar + obs_covar)`, posterior
`(1 - gain) * covar`) and an identity transition; the variance is one value per
pixel shared across the 64 channels; the innovation enters the noise network
L2-normalized, as KalmanNet normalizes its difference features. Unlike
KalmanNet, the gain is not output directly by a network but computed from the
learned `Q` and `R`, which keeps `P` interpretable as uncertainty.

Checkpoint format `adseg_kalman_memory_v2`. Version 1 (used for the first
results below) computed `Q_t` only after decoding, so the memory attention read
`P_{t-1}` without the frame-gap uncertainty, used `|C_t - S|` unnormalized, and
fed the 512-pixel mask to `R_net` during training but the 128-pixel mask at
inference.

Training uses straight-through object gates (`SAM2Base.straight_through_object_gate`,
enabled only by `KalmanMemoryTrainer`). MedSAM2 rounds the object score at
three places: the output mask (set to -1024 when the score is <= 0), the
no-object pointer blend, and the no-object memory embedding. The forward pass
keeps these hard choices, so training and inference outputs are identical; in
the backward pass the pointer and memory gates use `sigmoid(score)`, and mask
losses reach the mask logits even when the gate is closed. The mask gate does
not pass gradient to the score, which would be scaled by about 1024.

Design intent, tied to the PolypGen measurements below:

| Problem measured in PolypGen | Mechanism |
|---|---|
| Absent runs of 16–21 frames exceed the 7-frame native window | anchor slot always present; absent frames get large `R` and barely overwrite `S` |
| Annotated frame gaps of 3–80 raw frames | `Q` grows with `dt` and appearance change, so after long gaps the next reliable frame is trusted more |
| Native predicts a polyp on 65% of empty frames | presence-dependent observation noise plus an absence objective in training |
| Clinical need for confidence | `P` is a per-pixel uncertainty map |

Initialization: the noise heads' last convolutions are zero-initialized with
biases giving `Q = 1`, `R = 0.1` (plus `10 * (1 - p)`) and `P_0 = 1`, so an
untrained module writes about 95% of a present frame and about 17% of an
absent one, and `u = 0` leaves the readout equal to `S`.

Retained spatial memory is 0.88 MiB per object (anchor, mean, variance,
position, previous image projection) regardless of video length.

## Data protocol

Training never reads `sequenceData/positive`.

| Role | Source |
|---|---|
| Positives for training | `data_C1`..`data_C6` single frames with non-empty masks (1,399 after exclusion) |
| Absences for training | `sequenceData/negativeOnly` (23 sequences, 4,275 frames) and out-of-view motion |
| Test | all 23 `sequenceData/positive` sequences (2,225 frames) |

`scripts/check_split_overlap.py` compares difference hashes of every training
and test frame. Twelve single frames are renamed copies of positive-sequence
frames (34 of seq15's 116 frames have a duplicate in `data_C2`); they are
listed in `datasets/splits/polypgen/single_frame_test_overlap.txt` and excluded.
No negative-sequence frame duplicates a test frame.

Training clips are 16-frame pseudo-videos (`datasets/pseudo_video.py`): one single frame
follows a mean-reverting camera path sampled in raw-frame time (about 3–4% of
the image per 3-frame gap and about 12% per 80-frame gap, matching measured
PolypGen centroid shifts), with photometric jitter and motion blur. With
probability 0.6 a stretch of 2–12 frames is replaced by consecutive
`negativeOnly` frames or the polyp is moved out of view; masks are warped with
the same transform, so labels are exact. The 12-frame maximum is longer than
MedSAM2's 7-frame memory window; test videos are still longer (15–250 frames,
absences up to 54 frames, frame gaps up to 400 in seq11 versus at most 80 in
training), so behavior on the longest absences remains an extrapolation.

Two difficulty presets (`--clip_difficulty`, default `hard`):

| | `easy` (first run) | `hard` |
|---|---|---|
| Frame gap | 1–40 | 1–80 |
| Camera spread (x, y, log-zoom, degrees) | 0.12, 0.12, 0.2, 8 | 0.18, 0.18, 0.35, 15; no zoom-out below 0.95 |
| Degradations | gain/offset/tint jitter, motion blur 20% | wider jitter, motion blur 35%, defocus 20%, noise 50%, JPEG 30%, moving specular highlights, vignetting/exposure |
| Occlusion | none | 40% of clips: displaced-tissue ellipse over part of the polyp for 1–4 frames, hidden pixels removed from the label |
| Absence | other-video splice or slide off the image (30%) | 40% polyp inpainted in the same scene; otherwise look away within the scene (50%) or other-video splice |

`hard` was calibrated against native MedSAM2 behavior on the real test videos.
The real empty-frame false positives are mostly polyp-like tissue or edge
remnants in the same scene, which the inpainted and look-away absences imitate.

| Native MedSAM2 | `easy` (40 clips) | `hard` (60 clips) | Real test videos |
|---|---:|---:|---:|
| Present-frame Dice | 0.894 | 0.588 | 0.537 |
| Empty frames predicted present | 0.44 | 0.53 | 0.65 |

The prompt box is sampled by MedSAM2's `SAM2Train` from the frame-0 mask
(`sample_box_points`: noise 10% of the box size, at most 20 px), so the
generator no longer draws its own box; the `easy` clip stream therefore differs
from the first run's.

Hyperparameters are fixed before testing and the final checkpoint is the last
step, so no test label influences model selection.

## Training stack

Training reuses MedSAM2's own training code; the project adds only the Kalman
pieces:

| Component | Source |
|---|---|
| Model | `SAM2Train` (`MedSAM2/training/model/sam2.py`) with `KalmanMemoryMixin`, as `KalmanSAM2Train` (`training/kalman_trainer.py`) |
| Kalman memory | `KalmanMemoryMixin` (`modeling/kalman_memory.py`), shared with `KalmanMemoryVideoPredictor` |
| Batch format | `BatchedVideoDatapoint` (`MedSAM2/training/utils/data_utils.py`) |
| Mask losses | `MultiStepMultiMasksAndIous` (`MedSAM2/training/loss_fns.py`) |
| Prompt box | `SAM2Train.prepare_prompt_inputs` / `sample_box_points` on the frame-0 mask |
| Teacher | the same model with `kalman_enabled = False` (native bank), without gradients |

The mixin hooks `_prepare_memory_conditioned_features` (predict + read) and
`_encode_memory_in_output` (update). The video predictor and `SAM2Train` both
call these on every frame, so training and inference run the same Kalman code.

Settings that keep training identical to inference:

- the model stays in eval mode with the video predictor's decoder overrides
  (stability-based multimask fallback, binarized prompt masks for the memory
  encoder), built through the same `build_video_predictor` path;
- one box prompt on frame 0 (`prob_to_use_pt_input_for_eval = 1`,
  `prob_to_use_box_input_for_eval = 1`) and no correction clicks
  (`frames_to_add_correction_pt` cleared; MedSAM2's correction routine fails
  with zero clicks);
- frame memories rounded to bf16, the video predictor's storage precision, in
  both paths.

## Training objective

MedSAM2 is frozen; only the Kalman update is trained. On propagated frames
`t >= 1` (`KalmanLoss`):

```text
present frames:  20 * focal(mask) + 1 * Dice(mask) + 1 * L1(predicted IoU, actual IoU)   # MultiStepMultiMasksAndIous
               + 1 * BCE(object score, 1)                   # fixed, MedSAM2 loss_class weight
empty frames:    w_absence  * BCE(object score, 0)          # --absence_weight, default 0.1
present frames:  w_distill  * normalized MSE(Kalman vs native-bank memory-conditioned features)
```

The mask terms use MedSAM2's weights from `sam2.1_hiera_tiny_finetune512.yaml`
and are averaged over frames; as in MedSAM2 they apply only where the object is
present, so the object score is the only absence signal. Unlike MedSAM2, they
supervise the output mask, i.e. the candidate the frozen IoU head selects
(`argmax` of the predicted IoUs), not the candidate that best matches the ground
truth: MedSAM2 also trains the IoU head to make those coincide, but here it is
frozen, and on 15 hard clips the two differed in 44% of frames with a polyp.
MedSAM2's own
object-score term (`loss_class`) is replaced by BCE averaged separately over
present and empty frames, so the `--absence_weight` ratio does not depend on how
many frames in a clip are empty.

When the object score of a present frame is <= 0, MedSAM2 sets its mask logits
to -1024, so the focal term for that frame is large (about 0.25 * 1024 per
polyp pixel). This is MedSAM2's behavior; the value is a constant in the
forward pass and its straight-through gradient is bounded, so logged
segmentation losses spike on such clips without destabilizing training.

The first recorded run (below) used an earlier objective: pixel BCE + Dice on
all frames, presence BCE 0.5 on present frames, and an NPO term (weight 1.0)
on empty frames relative to the native-bank teacher. Pixel BCE on empty frames
was itself a strong absence signal, so that run cannot isolate the effect of
the absence term. NPO and gradient ascent have since been removed.

## Verification

- `scripts/verify_kalman_parity.py --perturb` compares, with randomly perturbed
  Kalman layers, the training model against both inference predictors on 10
  frames of seq20 and seq5 (CPU): Kalman path versus `KalmanMemoryVideoPredictor`
  (mask IoU 1.0, object-score error <= 7.6e-6, gain error <= 6e-8) and teacher
  path versus the native MedSAM2 predictor (mask IoU 1.0, object-score error
  <= 8.6e-6), including bank eviction.
- Straight-through gates: forward outputs are bit-identical with the gates on
  or off; on a hard clip with 4 of 7 frames below the presence threshold, the
  mask loss on those frames reaches the Kalman parameters only with the gates
  on (gradient norm 0 versus 28).
- On MPS, MedSAM2 offloads frame memories with `non_blocking=True`; reading the
  copy immediately can return uninitialized values. The Kalman update now reads
  each candidate inside `_encode_memory_in_output`, before offloading; the
  predictor synchronizes before reading the offloaded prompt-frame memory and
  raises if the state becomes non-finite. Before the prompt-frame fix, one MPS
  test run produced NaN states on 1,348 of 2,012 frames (Dice 0.27); that run is
  discarded. CPU and CUDA are unaffected.

## Evaluation

`evaluation/temporal.py` writes `presence_stratified.csv`, separating propagated
frames by ground-truth presence and by frames since the polyp reappeared after
an absence of at least 1 or 7 frames. Pooled Dice scores 1 for an empty
prediction on an empty frame and 0 for any false positive there, so it mixes
segmentation quality with absence detection.

### Native MedSAM2 baseline (GT-box prompt, all 23 sequences)

Prompt: tight box of the first non-empty ground-truth mask; seq1 and seq7 have
no polyp and receive empty predictions.

| Group | Frames | Dice | Predicted-present rate |
|---|---:|---:|---:|
| All frames (incl. prompt and pre-prompt) | 2225 | 0.554 | |
| Propagated | 2012 | 0.507 | 0.865 |
| Ground truth present | 1689 | 0.537 | 0.906 |
| Ground truth absent | 323 | 0.350 | 0.650 |
| First frame after absence >= 1 | 51 | 0.361 | |
| First frame after absence >= 7 | 11 | 0.203 | |
| 1–4 frames after absence >= 7 | 52 | 0.445 | |

Output: `outputs/kalman_protocol/native_gtbox/`.

### First Kalman results (GT-box prompt, all 23 sequences, seed 0)

`untrained` is the initialization (`--steps 0`); `npo` is 1,000 optimizer steps
x 4 clips with the default objective. Step 250 and untrained ran on CPU, the
final checkpoint on MPS after the synchronization fix.

| Group | Native | Untrained | NPO step 250 | NPO final |
|---|---:|---:|---:|---:|
| Propagated Dice | 0.507 | 0.531 | 0.518 | 0.527 |
| Present-frame Dice | 0.537 | **0.574** | 0.553 | 0.551 |
| Present frames predicted present | 0.91 | 0.93 | 0.91 | 0.88 |
| Absent-frame Dice | 0.350 | 0.307 | 0.334 | **0.402** |
| Absent frames predicted present (FP) | 0.65 | 0.69 | 0.67 | **0.60** |
| First frame after absence >= 7 (n=11) | 0.203 | 0.251 | 0.249 | 0.177 |
| >= 10 frames after reappearance | 0.452 | **0.519** | 0.470 | 0.482 |

Per-sequence paired Wilcoxon tests over the 21 sequences containing a polyp:

| Comparison | Metric | Mean diff | Better / worse | p |
|---|---|---:|---:|---:|
| Untrained vs native | present-frame Dice | +0.019 | 14 / 7 | 0.070 |
| Untrained vs native | propagated Dice | +0.007 | 14 / 7 | 0.137 |
| NPO final vs native | propagated Dice | -0.002 | 12 / 8 | 0.332 |
| NPO final vs untrained | present-frame Dice | -0.032 | 7 / 14 | 0.022 |
| NPO final vs native | empty-frame FP rate (17 seqs) | -0.051 | 6 lower / 2 higher | 0.183 |
| NPO final vs native | present-frame detection rate | -0.017 | 2 higher / 8 lower | 0.074 |

Interpretation:

- The architecture alone (anchor slot plus presence-gated state, no training)
  improves present-frame Dice, mostly late after reappearance (seq5 +0.15,
  seq6 +0.08, seq18 +0.06); seq21 drops from 0.685 to 0.448. Not yet
  significant.
- NPO training trades sensitivity for specificity: false positives on empty
  frames fall (0.65 to 0.60; seq21 0.39 to 0.17, seq23 0.36 to 0.14) but fewer
  present frames are detected and present-frame Dice is significantly below the
  untrained module. Net effect versus native is nil.
- Training raised the learned gain on absent frames from about 0.17 to about
  0.90, undoing the intended presence gating; seq6 collapses after training
  (0.865 to 0.391, all empty frames predicted present).
- In 13 of 21 sequences the empty-frame false-positive rate is identical for all
  methods, so absence errors are mostly decided by the frozen decoder's object
  score, not by the memory.

## Required runs

1. Native baseline with GT-box prompts (done above) and with YOLO prompts from a
   detector not trained on PolypGen.
2. Kalman memory with the MedSAM2 objective on `hard` clips, with
   `--absence_weight` 0, 0.1, and 1.0 (the absence ablation).
3. Ablations with the same seed and steps: `--distill_weight 0`,
   no training absences (`--absence_probability 0`), and an untrained module
   (`--steps 0`, initialization only).
4. Native bank trimmed to 1, 2, and 4 frames, and a fixed-gain average (EMA).
5. One seed (0) per trained row on PolypGen, a proxy dataset; per-sequence and
   per-center tables; paired Wilcoxon tests over sequences.
6. Retained-memory size and FPS for every row.

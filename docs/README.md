# AdeSEG

AdeSEG studies **video-level adenoid hypertrophy assessment from
nasopharyngoscopy**. The intended clinical pipeline is:

```text
nasopharyngoscopy video
    -> adenoid and nasopharyngeal-airway segmentation
    -> valid and clinically stable frames
    -> per-frame obstruction measurement
    -> video-level aggregation
    -> hypertrophy grade
```

The exact obstruction-ratio formula and grade thresholds are deliberately not
hard-coded. They must come from the annotation and clinical protocol supplied
with the adenoid dataset.

## Research scope

### C1: video-level two-region assessment

The target task is joint segmentation of `adenoid` and
`nasopharynx_airway`, followed by a quantitative video-level obstruction
measurement. This differs from static-image classification or one-frame
measurement because the clinical score is derived after temporal propagation,
frame selection, and aggregation.

### C2: measurement-aware segmentation

The first required experiment is a diagnostic: determine whether good regional
Dice can still produce a poor obstruction measurement. The repository provides
`evaluation.evaluate_two_region_measurement()` for this comparison. It reports
per-region Dice/IoU, predicted ratio, ground-truth ratio, and absolute ratio
error for an explicitly selected ratio protocol.

A measurement-aware loss is **not implemented yet**. It should only be added
if the diagnostic demonstrates a real Dice--measurement mismatch and after the
adenoid protocol defines the clinical ratio.

### C3: Kalman spatial memory (experimental)

The repository contains an experimental replacement for MedSAM2's 7-frame
FIFO memory bank, evaluated on PolypGen (proxy). The bank is replaced by the
fixed prompt-frame memory plus one Kalman spatial state with a per-pixel
variance: each frame's memory is fused into the state with gain
`K = P⁻ / (P⁻ + R)`, where the measurement noise `R` grows when MedSAM2 calls
the object absent. `--skip_absent` (off by default) treats a frame called
absent as a missing measurement. The constant-gain EMA baseline
(`modeling/ema_memory.py`, `--fixed_gain PRESENT ABSENT`) keeps the same
single state and read path but uses a fixed gain, which isolates what the
Kalman update itself contributes.

Current findings (held-out C6, 8 sequences; `docs/EXPERIMENTS.md`):

| | Dice | Polyp-frame Dice | Empty-frame FP |
|---|---|---|---|
| native MedSAM2 | 0.617 | 0.599 | 0.486 |
| EMA (constant gain) | 0.603 | 0.623 | 0.587 |
| Kalman | 0.602 | 0.613 | 0.555 |
| Kalman + output presence gate | 0.628 | 0.598 | 0.388 |
| native + output presence gate | 0.628 | 0.573 | 0.332 |

- The 1-slot memory raises polyp-frame and high-motion Dice over native's bank
  (Kalman beats native on Dice in 7 of 8 sequences) but also raises
  empty-frame false positives; one collapsed sequence leaves its mean Dice
  slightly below native. On the longer dev videos it loses fewer polyp frames
  (archived: 0.40 → 0.32).
- The Kalman update rule does not beat the constant-gain EMA, including a
  pre-registered test with `--skip_absent` designed to favour it on
  reappearance. Its gain is nearly constant, so the filter behaves like an EMA.
- An output presence gate on MedSAM2's object score lowers false positives
  for every method.
- None of the C6 differences is significant with 8 sequences.

It is not yet a validated component of the adenoid clinical pipeline.

Native MedSAM2 remains the primary baseline. `EXPERIMENTS.md` records the
design, data protocol, verification, and baseline results.

## What is implemented now

- **MedSAM2 wrapper:** `modeling/medsam2.py` and the bundled upstream code under
  `MedSAM2/`.
- **Kalman spatial memory:** `modeling/kalman_memory.py` (checkpoint format
  `adseg_kalman_memory_v5`; compatible with existing v4 checkpoints).
- **EMA baseline:** `modeling/ema_memory.py`, a constant-gain memory with the
  same read path, used for the ablation.
- **Frozen-backbone training:** `training/kalman_trainer.py` and
  `scripts/train_kalman.py`. Both consume one dataset-neutral JSONL manifest
  with ordered video frames and indexed semantic masks. Only the Kalman update
  is trainable.
- **Inference:** `scripts/infer.py` runs native, Kalman (`--skip_absent`
  optional) or EMA (`--fixed_gain`) memory from that manifest, with a first
  annotated-frame box prompt. It writes masks and a per-frame diagnostics CSV
  (object score, presence, gain, camera motion).
- **PolypGen manifest:** `scripts/build_polypgen_manifest.py` writes
  `data/polypgen_sequence.jsonl` for protocol (a), with frames in numeric
  temporal order.
- **Evaluation:** `evaluation/temporal.py` (per-frame, presence-stratified and
  drift metrics; `--no_overlays` for metrics only), plus
  `scripts/analyze_reappearance.py` and `scripts/analyze_illumination.py`.
- **Measurement utilities:** `evaluation/measurement.py` and
  `evaluation/ratio.py` support downstream two-region measurements.

There is currently no annotated adenoid dataset adapter, clinical obstruction
ratio protocol, or complete adenoid video runner in this repository. The
adenoid folders provide scaffolding and measurement helpers; they do not make
the clinical task executable without the target data and protocol.

## Method interpretation and novelty boundary

Related work this memory builds on or must be compared with: RDE-VOS
(recurrent constant-size memory), LiVOS (gated recurrent memory), SAMURAI
(Kalman filtering of box motion, not of memory), EMA-SAM (confidence-weighted
moving-average memory for medical SAM2), SAM2Long and DAM4SAM (memory
selection), and TinySAM 2 (memory-token compression). Related Kalman work: KalmanNet and
RKN (learned Kalman gains), KEEP (Kalman-inspired feature propagation for
video), and SAM2Plus (Kalman filter on SAM2 box IoU). The candidate technical
contribution is the per-pixel Kalman update of MedSAM2's spatial memory with
presence-dependent measurement noise. On PolypGen its benefit over MedSAM2's
bank (better polyp-frame tracking, fewer lost frames on long videos) is matched
by a constant-gain moving average, so it comes from the single recursive
state rather than the adaptive gain. A gain that adds to that would need an
informative per-frame measurement noise, which no proxy signal has provided.
These are claims to test against the baselines above, not established results.

The project-level contribution remains the **video-level two-region adenoid
assessment formulation**: segmentation of adenoid and nasopharyngeal airway
over a video, followed by a protocol-defined obstruction measurement and
video/patient-level aggregation. The Kalman memory supports that formulation
and is evaluated on PolypGen until adenoid video data are available.

The current measurement-aware objective is not implemented. It must not be
claimed as a contribution until an adenoid annotation protocol defines the
ratio and an experiment shows that pixel Dice alone can give clinically
material ratio errors.

## Repository layout

- `adenoid/`: target-task measurement and grading helpers.
- `datasets/`: the manifest-based video dataset (`video.py`) and the PolypGen
  protocol (a) split.
- `modeling/`: MedSAM2 construction, native-pointer preparation, the Kalman
  memory (update module, shared mixin, video predictor) and the EMA baseline.
- `training/`: `KalmanSAM2Train` (MedSAM2's `SAM2Train` with the Kalman mixin),
  batch construction, and the loss built on MedSAM2's `MultiStepMultiMasksAndIous`.
- `evaluation/`: segmentation, temporal, ratio, and two-region measurement
  metrics.
- `scripts/`: manifest building, inference, training, and the reappearance and
  illumination analyses.
- `MedSAM2/`: bundled upstream implementation.

## Proxy datasets

### PolypGen

PolypGen validates video segmentation and temporal propagation. The positive
sequences are sparsely annotated (3–80 raw frames between annotations), 4–55%
of frames in several sequences contain no polyp, and seq16–seq23 all come from
center C6. With GT-box prompts on all 23 sequences, native MedSAM2 reaches Dice
0.537 on frames with a polyp and predicts a polyp on 65% of empty frames
(`EXPERIMENTS.md`). Protocol (a) (`datasets/splits/polypgen/protocol_a.json`)
develops on seq1–15 and tests frozen methods once on C6 (seq16–23); seq1 and
seq7 contain no polyp, so 13 dev sequences are propagated. Compare methods only
under the same checkpoint, prompts, evaluator, and protocol.

Frames must be in temporal order. PolypGen frame files carry numeric suffixes
of different lengths, so an alphabetical sort is wrong (frame 104 sorts before
69); `scripts/build_polypgen_manifest.py` sorts numerically. Results from a
manifest built before commit `c8df175` are invalid.

### CholecSeg8k

[CholecSeg8k](https://arxiv.org/abs/2012.12453) (CC BY-NC-SA 4.0, no access
request) was used as a second real-video dataset in archived runs; its
converter is not part of the current manifest-based workflow.

### Two-region measurement

`evaluation/ratio.py` and `evaluation/measurement.py` compute the candidate
adenoid ratios (`adenoid / airway`, `adenoid / (adenoid + airway)`) and report
per-region Dice/IoU next to ratio MAE, RMSE, Pearson and Spearman agreement.
The REFUGE2 proxy that exercised this path was removed with the old
dataset-specific scripts; it will be validated on adenoid data.

## Kalman spatial memory experiment

MedSAM2 stays frozen. Only the Kalman update has parameters. The dataset is a
JSONL manifest with `split`, `video_id`, `frame_index`, `image`, and `mask`
fields. The mask is indexed (`0` background; `1` for a binary target, or, for
example, `1` adenoid and `2` airway). Split by video or patient, never by frame.

Train and infer without detectors or learned fusion:

```bash
python3 scripts/train_kalman.py \
  --sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/adenoid/frames.jsonl --label_ids 1 2 \
  --output_dir outputs/kalman

python3 scripts/infer.py \
  --sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/adenoid/frames.jsonl --split test --label_ids 1 2 \
  --memory_backend kalman --kalman_checkpoint outputs/kalman/kalman_memory.pt \
  --output_dir outputs/kalman_test
```

Reproduce the PolypGen held-out comparison (run each method once on `--split
test`; use `--split dev` with seq1–15 for any calibration):

```bash
python3 scripts/build_polypgen_manifest.py   # data/polypgen_sequence.jsonl

COMMON="--sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/polypgen_sequence.jsonl --split test --label_ids 1"
python3 scripts/infer.py $COMMON --memory_backend native --output_dir runs/native
python3 scripts/infer.py $COMMON --memory_backend kalman --fixed_gain 0.835 0.266 --output_dir runs/ema
python3 scripts/infer.py $COMMON --memory_backend kalman \
  --kalman_checkpoint kalman_memory_untrained.pt --output_dir runs/kalman

python3 evaluation/temporal.py --output_mask_dir runs/kalman/masks \
  --sequences seq16 seq17 seq18 seq19 seq20 seq21 seq22 seq23 --no_overlays
```

The EMA gains 0.835 / 0.266 are the untrained Kalman update's mean gains on
present / absent dev frames. The untrained checkpoint is
`AdeSEG/outputs/Kalman/kalman_memory_untrained.pt` on Google Drive. The
checkpoint format is `adseg_kalman_memory_v5`. Results go to Google Drive
(`AdeSEG/outputs/Kalman`); `EXPERIMENTS.md` has the update equations, every
run, and the folder names.

## Reference papers and implementations

- [MedSAM2: Segment Anything in 3D Medical Images and Videos](https://arxiv.org/abs/2504.03600)
  and the [official MedSAM2 repository](https://github.com/bowang-lab/MedSAM2).
- [Recurrent Dynamic Embedding for Video Object Segmentation (RDE-VOS)](https://arxiv.org/abs/2205.03761)
  and its [official code](https://github.com/Limingxing00/RDE-VOS-CVPR2022).
- [LiVOS: Light Video Object Segmentation with Gated Linear Matching](https://arxiv.org/abs/2411.02818)
  and its [official code](https://github.com/uncbiag/LiVOS).
- [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf),
  [DAM4SAM](https://arxiv.org/abs/2509.13864),
  [EMA-SAM](https://arxiv.org/abs/2510.18213), and
  [TinySAM 2](https://arxiv.org/abs/2605.18013): SAM2 memory selection,
  moving-average memory, and memory compression baselines.
- [KalmanNet](https://arxiv.org/abs/2107.10043) and
  [Recurrent Kalman Networks](https://arxiv.org/abs/1905.07357): learned
  Kalman gains.
- [SAMURAI](https://arxiv.org/abs/2411.11922): Kalman filtering of SAM2 box
  motion for tracking.
- [PolypGen](https://arxiv.org/abs/2106.04463): the multi-center dataset.
- [CholecSeg8k](https://arxiv.org/abs/2012.12453): laparoscopic video
  segmentation dataset.

These references support the source ideas being adapted. They do not validate
the AdeSEG clinical task, its eventual obstruction ratio, or the Kalman memory
results; those require target-dataset experiments.

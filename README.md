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
FIFO memory bank, evaluated on PolypGen (proxy) and CholecSeg8k. The bank is
replaced by the fixed prompt-frame memory plus one Kalman spatial state with a
per-pixel variance: each frame's memory is fused into the state with gain
`K = P⁻ / (P⁻ + R)`, where the measurement noise `R` grows when MedSAM2 calls
the object absent. Optional, off-by-default variants treat absent frames as
missing measurements (`--skip_absent`) or reject frames by measurement
validation on the object score (`--score_gate`).

Current findings (`EXPERIMENTS.md`):

- Without a detector, the 1-slot memory reduces error accumulation on the long
  PolypGen dev videos (lost polyp frames 0.40 → 0.32, recovered episodes
  0.57 → 0.82) and helps most under fast camera motion; on the short C6 test
  videos it ties native MedSAM2.
- The untrained gain is nearly constant, so the filter behaves like an
  exponential moving average; a constant-gain ablation matches it.
- An oracle gain (write only correct frames) shows large headroom on PolypGen
  (dev Dice +0.11 over the moving average, half the lost frames) and none on
  CholecSeg8k. No hand-designed or pseudo-video-trained gain has captured that
  headroom without raising false positives.

It is not yet a validated component of the adenoid clinical pipeline.

Native MedSAM2 remains the primary baseline. `EXPERIMENTS.md` records the
design, data protocol, verification, and baseline results.

## What is implemented now

- **MedSAM2 wrapper:** `modeling/medsam2.py` and the bundled upstream code under
  `MedSAM2/`.
- **Kalman spatial memory:** `modeling/kalman_memory.py` (checkpoint format
  `adseg_kalman_memory_v4`; ablation and diagnostic options listed by
  `scripts/infer.py --help`). `modeling/native_pointers.py` reproduces
  MedSAM2's object-pointer selection (`scripts/verify_native_pointers.py`).
- **Detector observations and presence fusion (DOK-Mem):**
  `modeling/detector_observation.py`, `scripts/build_detector_dataset.py`,
  `scripts/train_detector.py`, `scripts/build_fusion_data.py`,
  `scripts/train_presence_fusion.py`. The presence fusion also runs without a
  detector.
- **Calibration on real dev videos:** `scripts/calibrate_innovation_gate.py`
  and `scripts/calibrate_reliability.py`.
- **Frozen-backbone training:** `training/kalman_trainer.py` and
  `scripts/train_kalman.py`, reusing MedSAM2's `SAM2Train`, loss, and batch
  format, on single-frame pseudo-videos from `datasets/pseudo_video.py`. Only
  the Kalman update is trainable.
- **Verification:** `scripts/verify_kalman_parity.py` compares the training and
  inference trajectories; `scripts/check_split_overlap.py` finds training
  frames that duplicate test frames.
- **Proxy evaluation:** `scripts/infer.py` for video inference (`native` or
  `kalman`; YOLO or GT-box prompts; `--split dev|test` for protocol (a)),
  `evaluation/temporal.py` for presence- and reappearance-stratified metrics,
  and `scripts/run_refuge2.py` for REFUGE2 image evaluation.
- **Second video dataset:** `scripts/prepare_cholecseg8k.py` converts one
  CholecSeg8k class into the same sequence layout.
- **Measurement utilities:** `evaluation/measurement.py`,
  `evaluation/ratio.py`, and `evaluation/refuge2.py`.

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
presence-dependent measurement noise. So far its benefit over MedSAM2's bank
is matched by a constant-gain moving average; a gain that adds to that is
still open (see the oracle-gain results). These are claims to test against the
baselines above, not established results.

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
- `datasets/`: common sample interface, PolypGen/REFUGE2 adapters, pseudo-video
  generation, and PolypGen split lists.
- `modeling/`: MedSAM2 construction, native-pointer preparation, and the Kalman
  memory (update module, shared mixin, video predictor).
- `training/`: `KalmanSAM2Train` (MedSAM2's `SAM2Train` with the Kalman mixin),
  batch construction, and the loss built on MedSAM2's `MultiStepMultiMasksAndIous`.
- `evaluation/`: segmentation, temporal, ratio, and two-region measurement
  metrics.
- `scripts/`: proxy inference, training, calibration, dataset conversion, and
  parity commands.
- `MedSAM2/`: bundled upstream implementation.

## Proxy datasets

### PolypGen

PolypGen validates video segmentation and temporal propagation. The positive
sequences are sparsely annotated (3–80 raw frames between annotations), 4–55%
of frames in several sequences contain no polyp, and seq16–seq23 all come from
center C6. With GT-box prompts on all 23 sequences, native MedSAM2 reaches Dice
0.537 on frames with a polyp and predicts a polyp on 65% of empty frames
(`EXPERIMENTS.md`). Protocol (a) (`datasets/splits/polypgen/protocol_a.json`)
develops on seq1–15 and tests frozen methods once on C6 (seq16–23). Compare
methods only under the same checkpoint, prompts, evaluator, and protocol.

### CholecSeg8k

[CholecSeg8k](https://arxiv.org/abs/2012.12453) (CC BY-NC-SA 4.0, no access
request) has 101 clips of 80 consecutive laparoscopic frames from 17 Cholec80
videos with dense masks. Place it at `data/CholecSeg8k` and convert one class:

```bash
python3 scripts/prepare_cholecseg8k.py --target gallbladder \
  --output data/CholecSeg8k_gallbladder
```

This writes `seqN/images_seqN`, `seqN/masks_seqN`, `sequences.csv`, and a
dev/test `split.json` that splits whole surgical videos. Gallbladder tracking is
easy for every method (native dev Dice 0.916).

### REFUGE2

REFUGE2 is used as a static engineering proxy for the future two-region
segmentation-to-measurement pipeline. The oracle protocol uses GT-derived disc
and cup boxes, so it is not comparable to fully automatic challenge systems.

Evaluation reports disc, cup, and derived disc-rim Dice/IoU. It keeps the
standard vertical cup-to-disc ratio (vCDR) and also derives two non-overlapping
area-ratio proxies from `cup` and `rim = disc - cup`:

```text
cup_rim_ratio = area(cup) / area(rim)
cup_fraction  = area(cup) / (area(cup) + area(rim))
```

These exercise the same software paths as the candidate future adenoid ratios
`adenoid / airway` and `adenoid / (adenoid + airway)`. REFUGE2 does not imply
an anatomical or clinical equivalence between optic-disc structures and the
adenoid/nasopharyngeal airway.

For each structural ratio, the evaluator reports MAE, RMSE, Pearson, and
Spearman agreement against ground truth. Per-image outputs also include ratio
absolute errors and allow analysis of segmentation Dice versus downstream
measurement error.

Run the full REFUGE2 oracle evaluation with:

```bash
python3 scripts/run_refuge2.py \
  --split val \
  --output_dir outputs/refuge2_val
```

The run writes `metrics_per_image.csv` and `summary.json`.

## Kalman spatial memory experiment

MedSAM2 stays frozen. Only the Kalman update has parameters; the untrained
checkpoint (`K` from fixed noise) is the main configuration, and training on
single-frame pseudo-videos is optional (`data_C1`..`data_C6` single frames minus
test duplicates, plus `sequenceData/negativeOnly`).

Protocol (a) on PolypGen, no detector:

```bash
# optional training of the update
python3 scripts/check_split_overlap.py
python3 scripts/train_kalman.py --device cuda --output_dir outputs/kalman
python3 scripts/verify_kalman_parity.py --kalman_checkpoint outputs/kalman/kalman_memory.pt

# dev runs: native baseline and Kalman memory
python3 scripts/infer.py --split dev --prompt_source gt_box -o outputs/native_dev/masks
python3 scripts/infer.py --split dev --prompt_source gt_box --memory_backend kalman \
  --kalman_checkpoint outputs/kalman/kalman_memory.pt -o outputs/kalman_dev/masks
python3 evaluation/temporal.py --output_mask_dir outputs/kalman_dev/masks

# once a design is frozen on dev: the same command with --split test
```

CholecSeg8k uses the same commands with
`-i data/CholecSeg8k_gallbladder --seq_nums <dev or test from split.json>` for
inference and `--data_root data/CholecSeg8k_gallbladder` for
`evaluation/temporal.py`.

`scripts/summarize_kalman_runs.py` collects runs into CSV tables. The
checkpoint format is `adseg_kalman_memory_v4`. Results go to Google Drive
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

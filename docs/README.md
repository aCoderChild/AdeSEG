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
`evaluation.evaluate_two_region_measurement()` for this comparison, run over a
whole manifest by `evaluation/two_region.py`. It reports per-region Dice/IoU,
predicted ratio, ground-truth ratio, and absolute ratio error for an explicitly
selected ratio protocol. On the REFUGE stand-in there is **no** mismatch so far:
prompting the outer and inner regions (`--nested_labels`) gives cup / rim Dice
0.917 / 0.924 and a ratio error of 0.019. An earlier figure (Dice 0.734 / 0.886,
ratio error 0.109) came from a flawed prompting design (see "Proxy datasets").

A measurement-aware loss is **not implemented yet**. It should only be added
if the diagnostic demonstrates a real Dice--measurement mismatch and after the
adenoid protocol defines the clinical ratio.

### C3: recurrent spatial memory (experimental)

The repository contains an experimental replacement for MedSAM2's 7-frame memory
bank, evaluated on PolypGen (proxy): the prompt frame's memory plus **one
recurrent state** that a memory update rewrites from each new frame memory. Two
update rules:

- **Modified RKN** (`modeling/kalman_memory.py`; under training): the official
  `KalmanFilter` of `external/RecursiveKalmanNet` on the appearance memory, with
  a data-association update (absent or unreliable frames only predict) and
  process noise, measurement noise, presence and reliability learned by official
  RKN `GRUNetwork`s. A YOLOv8n detector (`modeling/detector.py`,
  `checkpoints/polypgen_yolov8n.pt`, trained without any `sequenceData/positive`
  frame) corrects presence, and re-detects the polyp when the tracker's mask is
  judged wrong and a confident detection disagrees with it.
- **RDE-VOS** (`modeling/rde_memory.py`): the key branch of the official
  `MemCrompress` of `external/RDE-VOS-CVPR2022` (3D non-local + 3D ASPP +
  2x3x3 squeeze), read in the official `two-frames-compress` mode (prompt
  memory, RDE, last memory frame; written every 3 frames).

`modeling/recurrent_memory.py` plugs either update into MedSAM2; this part is
ours. The modified RKN is trained through frozen MedSAM2 on later frames' mask
loss, presence and write-reliability BCE, and the RKN's Gaussian NLL against the
carried ground-truth memory (inference uses past frames only); RDE-VOS uses its official objective (bootstrapped CE + KL distillation toward
native MedSAM2's bank).

Current findings (`docs/EXPERIMENTS.md`; C6 = seq16–23, 8 sequences):

| C6 | Dice | Polyp-frame Dice | Empty-frame FP | Lost polyp frames |
|---|---|---|---|---|
| native MedSAM2 | 0.617 | 0.599 | 0.486 | 0.283 |
| RDE-VOS (step 0 = untrained) | 0.623 | 0.638 | 0.533 | 0.237 |

- **Training did not help RDE-VOS.** On whole held-out dev videos every trained
  checkpoint scored below its untrained start, so selection kept step 0 and the
  C6 row is an untrained module. Training cut empty-frame false positives (0.89
  to 0.20) by losing polyp frames.
- **No difference from native MedSAM2 is significant** on C6 (RDE-VOS − native
  polyp-frame Dice +0.039, p = 0.38).
- The untrained RDE-VOS does well for a structural reason: its official mode
  always reads the prompt memory and the last memory frame, while its compressed
  slot is randomly initialised.
- The modified RKN has no results yet.
- C6 is not a pristine test set: it was seen in the first all-sequence protocol
  and used for several frozen comparisons during development.

It is not a validated component of the adenoid clinical pipeline.

Native MedSAM2 remains the primary baseline. `EXPERIMENTS.md` records the
design, data protocol, verification, and every result, marked as regenerated
by `scripts/analyze_runs.py` or archived. The earlier hand-set Kalman filter and
constant-gain EMA memories were removed from the code (last at commit
`fa3df57`); their results are archived there.

## What is implemented now

- **MedSAM2 wrapper:** `modeling/medsam2.py` and the bundled upstream code under
  `external/MedSAM2/`.
- **Recurrent memory:** `modeling/recurrent_memory.py` (MedSAM2 side, shared by
  both updates: memory state, `RecurrentMemory` mixin, video predictor,
  checkpoint save/load), `modeling/kalman_memory.py` (modified RKN update, from
  the official code in `external/RecursiveKalmanNet`) and
  `modeling/rde_memory.py` (RDE-VOS update, official code from
  `external/RDE-VOS-CVPR2022`, which needs `retrying`).
- **Frozen-backbone training:** `training/memory_trainer.py` and
  `scripts/train_memory.py --memory_update {rkn,rde}` train only the memory
  update on natural clips (16 frames, drawn video first) from a manifest's
  `train` and `val` splits, or from one split with `--val_videos` held out
  (PolypGen: `--train_split dev --val_videos seq13 seq14 seq15`). The frame-0
  prompt is a tight ground-truth box, as at inference. Modified RKN: the loss
  above (`--presence_weight 1 --reliability_weight 0.5 --nll_weight 0.1
  --write_iou 0.5`, detector `--detector`), Adam lr 5e-4 decayed to 5e-5, weight decay 1e-4 (the RKN notebook's). RDE-VOS:
  official objective, lr 1e-4, weight decay 1e-7, `--rde_mode two-frames-compress
  --mem_every 3`, distillation weight 10. Checkpoints are selected on whole
  held-out videos through the inference path (best polyp-frame Dice with
  empty-frame FP not above step 0), else step 0.

  ```bash
  for rule in rkn rde; do
    python3 scripts/train_memory.py --memory_update $rule \
      --sam2_cfg configs/sam2.1_hiera_t512.yaml \
      --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
      --manifest data/polypgen_sequence.jsonl --label_ids 1 \
      --train_split dev --val_videos seq13 seq14 seq15 \
      --protocol protocols/polypgen_recurrent_v1.json \
      --output_dir "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/polypgen/training/${rule}_trained"
  done
  ```
- **Inference:** `scripts/infer.py` runs native MedSAM2, or a trained memory
  update with `--memory_checkpoint`; `--redetect` adds detector re-detection to
  native MedSAM2 or an update without detector heads (RDE-VOS), from that manifest, with a first
  annotated-frame box prompt (`--nested_labels` prompts label k as the union of
  labels >= k). It writes masks, a per-frame diagnostics CSV per label (object
  score, presence, mean gain, camera motion), and retained memory and FPS in
  `setup.json`.
- **PolypGen manifest:** `scripts/build_polypgen_manifest.py` writes
  `data/polypgen_sequence.jsonl` for protocol (a), with frames in numeric
  temporal order.
- **Evaluation:** `evaluation/temporal.py` (per-frame, presence-stratified and
  drift metrics; `--no_overlays` for metrics only); `scripts/analyze_runs.py`
  rebuilds every PolypGen table (per-sequence and pooled metrics, lost episodes,
  high motion, paired Wilcoxon and bootstrap intervals) from saved
  per-frame CSVs; `scripts/analyze_reappearance.py` and
  `scripts/analyze_illumination.py` give the robustness analyses.
- **Two-region path:** `scripts/build_mask_manifest.py` (grey-coded masks to
  indexed masks + manifest) and `evaluation/two_region.py` (per-region Dice/IoU,
  ratio error, grade agreement). `tests/test_data_pipeline.py` guards frame
  order and mask decoding (`python3 -m pytest tests`).
- **Measurement utilities:** `evaluation/measurement.py` and
  `evaluation/ratio.py` support downstream two-region measurements.

There is currently no annotated adenoid dataset adapter, clinical obstruction
ratio protocol, or complete adenoid video runner in this repository. The
adenoid folders provide scaffolding and measurement helpers; they do not make
the clinical task executable without the target data and protocol.

## Method interpretation and novelty boundary

Related work this memory builds on or must be compared with: RDE-VOS
(recurrent constant-size memory, used here from its official code), LiVOS
(gated recurrent memory), EMA-SAM (confidence-weighted moving-average memory for
medical SAM2, the closest method), SAMURAI (Kalman filtering of box motion, not
of memory), SAM2Long and DAM4SAM (memory selection), and TinySAM 2
(memory-token compression). Related Kalman work: KalmanNet and Recursive
KalmanNet (learned Kalman gains; the modified RKN adapts Recursive KalmanNet's
structure), KEEP (Kalman-inspired feature propagation for video), and SAM2Plus
(Kalman filter on SAM2 box IoU). As run, RDE-VOS training did not improve on its
untrained state and the modified RKN is untested, so no claim about learned
memory updates is supported yet; it needs a training objective that does not
reward giving up on polyp frames, and a comparison with EMA-SAM.

The project-level contribution remains the **video-level two-region adenoid
assessment formulation**: segmentation of adenoid and nasopharyngeal airway
over a video, followed by a protocol-defined obstruction measurement and
video/patient-level aggregation. The recurrent memory supports that formulation
and is evaluated on PolypGen until adenoid video data are available.

The current measurement-aware objective is not implemented. It must not be
claimed as a contribution until an adenoid annotation protocol defines the
ratio and an experiment shows that pixel Dice alone can give clinically
material ratio errors.

## Repository layout

- `adenoid/`: target-task measurement and grading helpers.
- `datasets/`: the manifest-based video dataset (`video.py`) and the PolypGen
  protocol (a) split.
- `modeling/`: MedSAM2 construction, native-pointer preparation, the recurrent
  memory (MedSAM2 side) and the modified RKN and RDE-VOS updates.
- `training/`: `RecurrentMemorySAM2Train` (MedSAM2's `SAM2Train` with the
  recurrent memory), batch construction, the modified-RKN loss and the official
  RDE-VOS losses.
- `evaluation/`: segmentation, temporal, ratio, and two-region measurement
  metrics.
- `scripts/`: manifest building, inference, training, and the reappearance and
  illumination analyses.
- `external/`: bundled upstream code: `MedSAM2`, `RDE-VOS-CVPR2022`
  and `RecursiveKalmanNet` (`Latent_KalmanNet_TSP` is a reference copy, not
  imported).

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
converter (`scripts/prepare_cholecseg8k.py`) was removed and is in commit
`9bb801a`. Tracking rarely fails there, and all 1-slot variants tie.

### Two-region measurement (REFUGE as the two-label stand-in)

The target adenoid data are expected to follow Cai et al. 2024 (fiberoptic
nasopharyngoscopy, 3-class grey masks: 0 background, 128 nasopharyngeal
airway, 255 adenoid; graded by the A/N ratio, <50% / 50–75% / >75%). REFUGE
(`data/REFUGE`, fundus images) uses the same 0/128/255 grey coding (255
background, 128 optic-disc rim, 0 cup) with two adjacent regions, so it
exercises the same two-label path: mask decoding, two-label inference, per-region
Dice/IoU and the area ratio cup / (cup + rim), which has the form of an A/N ratio.
It is static images, so it does not test the video memory.

`scripts/build_mask_manifest.py` converts a grey-coded image/mask folder into
indexed masks (nearest-value mapping, so JPEG noise cannot create classes) and a
manifest; `evaluation/two_region.py` scores `infer.py` predictions per region and
on the ratio (MAE, RMSE, Pearson, Spearman; grade accuracy and Cohen's kappa
with `--grade_thresholds`; the median ratio per video for multi-frame videos).

```bash
python3 scripts/build_mask_manifest.py --root data/REFUGE --splits train val test \
  --value_map 255:0 128:1 0:2 --output_dir data/refuge     # adenoid: 0:0 128:1 255:2
python3 scripts/infer.py --sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt --manifest data/refuge/manifest.jsonl \
  --split val --label_ids 1 2 --nested_labels --output_dir runs/refuge_val
python3 evaluation/two_region.py --manifest data/refuge/manifest.jsonl --split val \
  --pred_dir runs/refuge_val --region_a cup:2 --region_b rim:1 --ratio_mode fraction_of_total
```

REFUGE val (400 images, native MedSAM2, ground-truth box prompts):

| Prompting | Cup Dice | Rim Dice | Ratio MAE |
|---|---|---|---|
| one pass per label (rim box = disc box) | 0.734 | 0.886 | 0.109 |
| `--nested_labels` (disc and cup; rim = disc − cup) | 0.917 | 0.924 | 0.019 |

Without `--nested_labels`, the rim's box is the whole disc's box and the rim
pass absorbs the cup: the predicted cup fraction was too low on all 400 images.
With it (the same structure as A/N, where nasopharynx = adenoid + airway), the
ratio error is small and REFUGE shows no Dice–measurement mismatch. Results:
`AdeSEG/outputs/refuge/val_nested/`.

## Recurrent memory experiment

MedSAM2 stays frozen. Only the memory update has parameters. The dataset is a
JSONL manifest with `split`, `video_id`, `frame_index`, `image`, and `mask`
fields. The mask is indexed (`0` background; `1` for a binary target, or, for
example, `1` airway and `2` adenoid, as in Cai et al.'s 128/255 coding). Split by
video or patient, never by frame.

```bash
python3 scripts/train_memory.py --memory_update rde \
  --sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/adenoid/frames.jsonl --label_ids 1 2 \
  --output_dir outputs/rde

python3 scripts/infer.py \
  --sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/adenoid/frames.jsonl --split test --label_ids 1 2 \
  --memory_checkpoint outputs/rde/memory_update.pt --output_dir outputs/rde_test
```

Reproduce the PolypGen C6 comparison (train as above on `--split dev`; never
calibrate on C6):

```bash
python3 scripts/build_polypgen_manifest.py   # data/polypgen_sequence.jsonl

COMMON="--sam2_cfg configs/sam2.1_hiera_t512.yaml \
  --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
  --manifest data/polypgen_sequence.jsonl --split test --label_ids 1"
python3 scripts/infer.py $COMMON --output_dir runs/native
for rule in rkn rde; do
  python3 scripts/infer.py $COMMON --output_dir runs/$rule \
    --memory_checkpoint outputs/polypgen/update_rules/${rule}_trained/memory_update.pt
done
for m in native rkn rde; do
  python3 evaluation/temporal.py --output_mask_dir runs/$m/masks \
    --sequences seq16 seq17 seq18 seq19 seq20 seq21 seq22 seq23 --no_overlays
done
python3 scripts/analyze_runs.py --run native=runs/native --run rkn=runs/rkn \
  --run rde=runs/rde --motion_run rkn --compare rkn:native rde:native rde:rkn
```

Masks are read with a label check (values must be 0 or the requested
`--label_ids`; grey multi-class JPEGs are rejected) and MedSAM2's hole filling is
off, so results do not depend on whether its compiled `_C` extension is
installed. Checkpoints written before the rename are called `kalman_memory.pt`;
they load the same way. Results go to Google Drive (`AdeSEG/outputs/`, indexed
by its `README.md`); `EXPERIMENTS.md` has every run and the folder names.

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
  [Recursive KalmanNet](https://arxiv.org/abs/2506.11639) (official code in
  `external/RecursiveKalmanNet`, the basis of the modified RKN): learned Kalman gains.
- [SAMURAI](https://arxiv.org/abs/2411.11922): Kalman filtering of SAM2 box
  motion for tracking.
- [PolypGen](https://arxiv.org/abs/2106.04463): the multi-center dataset.
- [CholecSeg8k](https://arxiv.org/abs/2012.12453): laparoscopic video
  segmentation dataset.

These references support the source ideas being adapted. They do not validate
the AdeSEG clinical task, its eventual obstruction ratio, or the recurrent
memory results; those require target-dataset experiments.

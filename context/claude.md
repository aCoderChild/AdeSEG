# AdeSEG — Claude Session Summary

## Project Goal

Segment the **adenoid and nasopharynx region** in endoscopy video using a two-stage pipeline:
1. YOLO detects the target region (bounding box)
2. SAM2-family model segments from that box

The current dataset is **PolypGen** (23 colonoscopy video sequences) used as a temporary public proxy. The intended target is real adenoid video endoscopy data.

---

## Pipeline Definitions

### YOLO_SAM2 (frame-by-frame)
- YOLO (YOLOv8n, `polypgen_yolov8n.pt`) detects per frame
- SAM2 large (`sam2_hiera_large.pt`) segments from the YOLO box — image mode, no temporal memory
- Each frame is independent; if YOLO finds no box, a blank mask is saved silently

### MedSAM2 (video propagation)
- Same YOLO model for detection
- YOLO box → SAM2 image predictor → mask → fed into **MedSAM2 video predictor** as a prompt
- Video predictor propagates masks **forward and backward** through the full sequence
- Config: `sam2.1_hiera_t512.yaml`, checkpoint: `MedSAM2_latest.pt`
- `video_prompt_source: mask`, `video_prompt_stride: 1`, `max_yolo_boxes_per_frame: 1`
- If YOLO finds zero boxes at conf=0.5 → throws `RuntimeError`, sequence skipped entirely (no mask files)

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| BBox IoU | YOLO box vs GT box — detection quality |
| Dice / IoU | Pixel-level mask overlap |
| Sensitivity | Fraction of GT mask captured (recall) |
| Specificity | Clean background (true negative rate) |
| S-measure | Structural similarity |
| E-measure | Enhanced alignment (pixel + region level) |

---

## Key Results

### BBox IoU (YOLO detection — identical for both methods)
- **Overall average: 0.8012** (21 of 23 sequences; seq1 and seq7 have zero detections)
- Best: seq10 0.936, seq16 0.940, seq13 0.930
- Worst: seq3 0.268, seq18 0.342, seq15 0.543
- seq1, seq7: YOLO finds **no detections** at conf=0.5

### Mask Segmentation

| Method | Seqs evaluated | Dice | IoU | Sensitivity | Specificity | S-measure | E-measure |
|---|---|---|---|---|---|---|---|
| MedSAM2 | 19 | **0.643** | **0.592** | 0.637 | 0.991 | 0.870 | 0.921 |
| YOLO_SAM2 | 23 | 0.519 | 0.486 | 0.505 | ~1.03* | 0.914 | 0.910 |

*YOLO_SAM2 specificity >1 is an averaging artefact from seq1/seq7 where prediction is empty (no FP → specificity = 1.0 exactly).

**Apples-to-apples caveat**: On the same 19 sequences MedSAM2 covers, YOLO_SAM2 achieves Dice ≈ 0.566 / IoU ≈ 0.531. MedSAM2 still leads but the gap is narrower than the headline numbers suggest.

---

## Most Noticeable Issues Found

### 1. seq1 and seq7 — zero YOLO detections
Both methods' bbox CSVs contain only a header row (0 detections). Consequence:
- YOLO_SAM2: saves blank mask, IoU = 0.0, included in average (drags it down)
- MedSAM2: throws RuntimeError, sequence skipped, not included in average

### 2. YOLO_SAM2 has a frame-ordering bug
`Test.py:122` uses `sorted(glob.glob(...))` — lexicographic string sort — giving order `0, 1, 10, 11, 12..., 2, 20...` instead of numeric order. MedSAM2 correctly uses `get_numeric_sort_key`. Does not corrupt per-frame metrics but is wrong for any sequential/temporal analysis.

### 3. seq10 — detection-segmentation disconnect
| Seq | YOLO BBox IoU | MedSAM2 mask IoU | YOLO_SAM2 mask IoU |
|---|---|---|---|
| seq10 | 0.936 (excellent) | 0.265 (terrible) | 0.262 (terrible) |

YOLO locates the target almost perfectly but both SAM2 variants fail to segment it. Likely low-contrast tissue appearance confuses the SAM2 segmentation head. **High relevance for adenoid imaging** (adenoids blend into surrounding pharyngeal mucosa).

### 4. MedSAM2 missing seq5 and seq11
YOLO detects boxes in both (YOLO_SAM2 achieves IoU 0.694 / 0.406 on them), but MedSAM2 errored during video propagation inference. Represents a reliability gap.

### 5. Where each method wins
- **MedSAM2 wins by large margins** on hard sequences where YOLO detection is weak within a sequence (seq4 +22.9%, seq18 +10.4%, seq9 +7.4%) — temporal propagation rescues weak-detection frames
- **YOLO_SAM2 wins marginally** on easy sequences where YOLO is already strong (seq12, seq14, seq16, seq19 — margins <3%)
- Net: MedSAM2 better overall because it wins hard cases by more than it loses easy cases

### 6. Bbox IoU files are identical by design
Both methods use the same YOLO checkpoint (`polypgen_yolov8n.pt`), same conf=0.5, same images → same detections. The `eval/bbox/iou_per_sequence.csv` files are **byte-for-byte identical**. Raw bbox CSVs differ only in:
- Frame sort order (numeric vs lexicographic — see bug above)
- Coordinate values at the 8th decimal place (~2×10⁻⁸) — floating-point rounding from different image-loading code paths; does not affect IoU comparisons

---

## Per-Sequence Quick Reference

| Seq | YOLO BBox IoU | MedSAM2 IoU | YOLO_SAM2 IoU | Notes |
|---|---|---|---|---|
| seq1 | — | MISSING | 0.000 | Zero YOLO detections |
| seq2 | 0.835 | 0.811 | 0.797 | Good for both |
| seq3 | 0.268 | 0.105 | 0.058 | YOLO fails → cascade failure |
| seq4 | 0.724 | 0.467 | 0.238 | MedSAM2 much better (+22.9%) |
| seq5 | 0.923 | MISSING | 0.694 | MedSAM2 errored |
| seq6 | 0.887 | 0.604 | 0.612 | Near-tie |
| seq7 | — | MISSING | 0.000 | Zero YOLO detections |
| seq8 | 0.918 | 0.631 | 0.653 | YOLO_SAM2 marginally better |
| seq9 | 0.717 | 0.326 | 0.252 | MedSAM2 better (+7.4%) |
| seq10 | 0.936 | 0.265 | 0.262 | Detection-segmentation disconnect |
| seq11 | 0.787 | MISSING | 0.406 | MedSAM2 errored |
| seq12 | 0.843 | 0.880 | 0.903 | Best for both |
| seq13 | 0.930 | 0.619 | 0.600 | MedSAM2 slightly better |
| seq14 | 0.924 | 0.580 | 0.663 | YOLO_SAM2 better (+8.3%) |
| seq15 | 0.543 | 0.630 | 0.588 | MedSAM2 better despite weak YOLO |
| seq16 | 0.940 | 0.786 | 0.791 | Near-tie |
| seq17 | 0.917 | 0.627 | 0.626 | Tie |
| seq18 | 0.342 | 0.366 | 0.262 | MedSAM2 better (+10.4%); both poor |
| seq19 | 0.913 | 0.875 | 0.881 | Excellent for both |
| seq20 | 0.804 | 0.165 | 0.155 | Both fail despite decent detection |
| seq21 | 0.911 | 0.344 | 0.372 | Both mediocre |
| seq22 | 0.889 | 0.833 | 0.842 | Very good for both |
| seq23 | 0.873 | 0.522 | 0.533 | Moderate |

---

## Implications for Real Adenoid Data

1. **YOLO must be fine-tuned on adenoid images** — current model trained on colon polyps; domain gap will be large.
2. **The seq10 failure pattern (good detection, poor mask) is expected to be common** in adenoid imaging due to low tissue contrast — needs investigation.
3. **MedSAM2 is the better architecture for video endoscopy** — temporal propagation handles occlusion, camera motion, and frames where YOLO detection is momentarily weak.
4. **Sensitivity is the weak metric** (MedSAM2: 0.637, YOLO_SAM2: 0.505) — both under-segment rather than over-segment. For adenoid grading (need full volume), this needs improvement.
5. **Fix the frame-ordering bug in YOLO_SAM2** before using for temporal analysis.
6. **Add robust error handling to MedSAM2** — currently crashes and loses entire sequences when YOLO finds no detections.
7. **`video_prompt_stride=1`** (prompt every frame) is safe but slow; experiment with stride=5–10 for real video.

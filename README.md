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

### C3: recurrent fusion memory

Recurrent Fusion Memory for MedSAM2 investigates whether MedSAM2's multi-frame
spatial memory can be compressed into one recurrent state while retaining its
native object-pointer history. Its main component is an RDE-VOS-inspired
two-frame spatial convolution. The optional LiVOS-inspired gate is retained as
an ablation, rather than a headline contribution.

Native MedSAM2 remains the primary baseline. The first held-out experiment did
not preserve the validation gain, so recurrent fusion is an experimental
memory--accuracy trade-off rather than an improved segmentation method.

## Repository layout

- `adenoid/`: target-task ratio measurement and grade conversion helpers.
- `datasets/`: common sample interface and adapters for PolypGen and REFUGE2.
- `modeling/`: MedSAM2 construction, native-pointer preparation, and RGM.
- `inference/`: image and video inference utilities.
- `evaluation/`: segmentation, temporal, ratio, and two-region measurement metrics.
- `scripts/`: runnable proxy-dataset scripts.
- `MedSAM2/`: bundled upstream implementation.

## Proxy datasets

### PolypGen

PolypGen validates video segmentation and temporal propagation. Native MedSAM2
on the fixed seq16--19 validation protocol achieved sequence-macro Dice 0.7753
and IoU 0.7134.

### REFUGE2

REFUGE2 verifies static two-region segmentation and structural measurement. The
oracle protocol uses GT-derived disc and cup boxes, so it is not comparable to
fully automatic challenge systems.

## Recurrent-fusion memory experiment

`modeling/rgm_memory.py` and `training/rgm_trainer.py` implement the current
recurrent-fusion experiment. It replaces only spatial mask memories with a recurrent state;
the anchor and MedSAM2's normal recent object-pointer history remain available
to memory attention. `scripts/train_rgm.py` freezes MedSAM2 in evaluation mode,
uses the same decoder and memory-encoding path as inference, updates state from
predicted masks, and applies Dice+BCE only to future predictions.
The optional gate receives decoder-predicted IoU plus pooled summaries of the
prior state, candidate, and their absolute difference. It is never trained
against ground-truth IoU. A fixed-gate ablation showed no material benefit
from learning this gate in the current protocol.

The initial update is fixed EMA alpha=0.1 by construction. The implementation
is retained for memory-efficiency and failure-analysis experiments; it is not
currently claimed as a segmentation improvement.

The bundled MedSAM2 code has one minimal RGM instrumentation change: compact
per-frame outputs retain the decoder's selected predicted-IoU value. Native
prediction and native recent-memory selection are unchanged.

## Current implementation boundary

There is no real adenoid dataset adapter, annotated ratio protocol, or
end-to-end adenoid segmentation entry point in the repository yet. The project
does not claim that the clinical video pipeline is implemented until those
inputs are available.

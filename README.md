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

### C3: temporal robustness study

MedSAM2 native memory and published temporal strategies may be compared as
baselines for nasopharyngoscopy. The contribution is the controlled comparison
on segmentation and downstream measurement, rather than a claim to a new
memory mechanism.

A literature-backed candidate is documented in
[`docs/constant_memory_direction.md`](docs/constant_memory_direction.md).
It separates anchor/recent references from a recurrent key-value history, but
it is not yet integrated into MedSAM2 because LiVOS readout and MedSAM2 memory
attention have incompatible interfaces.

## Repository layout

- `adenoid/`: target-task ratio measurement and grade conversion helpers.
- `datasets/`: common sample interface and adapters for PolypGen and REFUGE2.
- `modeling/`: MedSAM2 construction plus archived compact-memory ablation code.
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

## Archived compact-memory ablation

The single-state EMA, learned projection, and gate-only variants are retained
for reproducibility but are not active methods. They reduced retained spatial
state storage to 0.5 MiB, but suffered severe immediate validation degradation
against native MedSAM2. This is a negative finding: simple recurrent averaging
does not preserve the information used by MedSAM2's multi-frame memory.

`scripts/infer.py` therefore defaults to `--memory_backend native`. The
compact-memory backends remain available only for reproducing the ablation.

## Current implementation boundary

There is no real adenoid dataset adapter, annotated ratio protocol, or
end-to-end adenoid segmentation entry point in the repository yet. The project
does not claim that the clinical video pipeline is implemented until those
inputs are available.

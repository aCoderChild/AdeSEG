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

Recurrent Fusion Memory replaces MedSAM2's multi-frame spatial memory with one
recurrent state while retaining its native object-pointer history.

The current update combines two published mechanisms:

- a LiVOS-style data-dependent channel forget gate from the current image feature;
- the RDE-VOS spatio-temporal aggregation path: non-local extraction, residual
  3-D ASPP enhancement, then a `2 x 3 x 3` Conv3D squeeze.

The gate acts on the previous state before fusion, so the previous state is not
added again after the RDE fusion.

Native MedSAM2 remains the primary baseline. RGM is an experimental
memory--accuracy trade-off, not an assumed segmentation improvement.

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

`modeling/rgm_memory.py` and `training/rgm_trainer.py` implement the recurrent
memory experiment. MedSAM2 stays frozen; the recurrent state is updated from
predicted masks and training uses future-frame Dice+BCE.

For frame `t`, let `F_t` be the current image feature, `S_(t-1)` the recurrent
mask-memory state, and `C_t` the current MedSAM2 mask-memory candidate:

```text
g_t = mean(sigmoid(Conv1x1(F_t)), spatial_dims)
retained = g_t * S_(t-1)
S_t = RDE_Fusion([retained, C_t])
```

`g_t` is channel-wise. `RDE_Fusion` follows the public RDE-VOS `MemCrompress`
structure with non-local extraction, residual ASPP3D enhancement, and the final
`2 x 3 x 3` Conv3D squeeze.

This implementation uses checkpoint format `adseg_livos_rde_v1`. Older RGM
checkpoints are intentionally incompatible and require retraining.

The bundled MedSAM2 code retains the decoder's selected predicted-IoU value in
compact frame outputs for diagnostics. The LiVOS-style gate itself is driven by
image features, not predicted IoU.

## Current implementation boundary

There is no real adenoid dataset adapter, annotated ratio protocol, or
end-to-end adenoid segmentation entry point in the repository yet. The project
does not claim that the clinical video pipeline is implemented until those
inputs are available.

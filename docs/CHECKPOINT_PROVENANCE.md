# Backbone and checkpoint provenance — verified distinctions

There are **two different pretrained model families** in the research records.

## A. Compatibility-corrected author Lean-SAM2 (21 sequences)

- **Run:** `author_lean_automatic_mps_v1_dw8pj78c`
- **Saved artifact:** [scores/report.md](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view)
- **Model family:** Lean-SAM2 with SAM2.1 Hiera **B+** backbone
- **Checkpoint SHA-256 (from run manifest):**
  `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
- **Config:** `sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`
- **Lean submodule:** `8d1cf2618519d291905d22fe859b8302476d7f50`
- **Bypass checkpoint SHA-256:** `5e415e3d207f09ffe94216b1472a83518de5c54425ab55bdef8f331761af1c65`
- **Detector SHA-256:** `858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e`
- **Corrections:** `lean_indexed_spatial_rope_mps.py` (retained spatial positions)
  and `lean_mps_blocking_pointer_position.py` (blocking pointer transfer).
- **No learned channel-value refiner, trained teacher adapter, project token
  window, or additional memory-anchor modification.**
- **Result:** 21 eligible sequences, 1,683 post-prompt GT-present frames,
  patient-macro Dice **0.7420711278676757**, frame-weighted Dice
  **0.77249528554744**, and 188/336 labeled-empty false alarms.
- **Execution:** MPS, FP32, one automatic initial mask per sequence,
  no later detector redetection. The saved run was independently scored,
  but the videos were previously inspected and the seq23 viewport was
  retrospectively selected.

The exact historical runner is
`scripts/run_author_lean_automatic_v1.py` in the
[full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).
The current focused repository does not reproduce that complete runner's
legacy dependency tree; use the backup for strict run reproduction.

## B. MedSAM2-latest (medical Hiera-T/512 experiments)

- **Recorded filename:** `checkpoints/MedSAM2_latest.pt`
- **SHA-256:** `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`
- **Architecture/config family:** MedSAM2 Hiera **T**, 512-pixel input;
  a saved native-predictor study refers to
  `configs/sam2.1/sam2.1_hiera_t.yaml` within its selected author source.
  The bundled MedSAM2 package also has its own 512-pixel Tiny configurations.
  Their config strings cannot be swapped without validation.
- **Code:** `modeling/medsam2.py`, original vendored
  `external/MedSAM2/`, and hash-checked
  `modeling/medsam2_latest.py`.
- **Evidence:** [MedSAM2 selected high-res comparison](https://drive.google.com/file/d/1rIQRCOE9aHxfVpprNbyrk8dIzc2r5sub/view)
  and [teacher-preserving four-sequence analysis](https://drive.google.com/file/d/1xbxi2dWPbSalk1emae7LsDSSua7qNZ1_/view).
- **High-res four-target results:** equal-patient mean 0.5223 (selected)
  versus 0.5111 native, but seq8 and seq11 worsened; the predeclared
  advancement rule failed.
- **Teacher-preserving four-target results:** 0.5144 adapter versus
  0.5111 native, with mixed sequence changes; the advancement gate failed.
- **No verified 21-sequence mean Dice of 0.742 for MedSAM2-latest.**

The Drive reports name local implementation files such as
`modeling/highres_skip_ablation_v1.py` and
`scripts/run_medsam2_spatial_video_512_v1.py`. **Those files are not
present in the full GitHub experiment backup** inspected here. The teacher
loss is preserved, but a runnable, complete high-resolution teacher adapter
cannot be reconstructed from the repository alone.

## Verification boundaries

These SHA-256 values are recorded in saved manifests/reports. The actual
`.pt` weights are **not stored in this Git repository** and were not
accessible for direct byte hashing in this review. The optional
`verify_medsam2_latest` loader checks actual bytes before model creation.

Do not call one experiment's Dice an architecture-only gain over another
unless they share the backbone, checkpoint, initialization, frame set and
evaluation rule. A file named `latest` does not prove that it is the
newest available pretrained model release.

# Contribution 1 — Medical high-resolution, teacher-preserving mask refinement

This is the primary active method in the reorganized repository. The backbone
is the saved **MedSAM2_latest.pt**, MedSAM2 **Hiera-T at 512 pixels**. It is
not the SAM2.1 Hiera-B+ backbone from the 0.7421 Lean comparison.

## Hypothesis

Early detail-rich features and deep memory-conditioned features may correct
native MedSAM2's decoder high-resolution skip. A training-only native-teacher
KL term may help retain correct native predictions while learning from masks.

## Existing historical evidence (4 previously inspected target sequences)

| Method | Equal-patient present Dice | Empty false alarms | Outcome |
|---|---:|---:|---|
| Native MedSAM2 | **0.5111** | 149/180 | Reference |
| Learned high-resolution skip adapter | **0.5223** | 152/180 | Seq8/seq11 worsened; four-patient gate failed |
| Teacher-preserving adapter | **0.5144** | 148/180 | Seq8/seq11/seq14 worsened; four-patient gate failed |

These numbers are from **separate, historically saved candidate studies**.
No ranking of the two candidates under fully identical training conditions is
implied. The reported target sequences were seq8, seq11, seq14 and seq23.
The larger 21-video effectiveness study was **not run** for either candidate.
No significant or confirmed generalization improvement may be claimed.

Sources:
- [Selected high-resolution adapter report](https://drive.google.com/file/d/1rIQRCOE9aHxfVpprNbyrk8dIzc2r5sub/view)
- [Teacher-preserving adapter analysis](https://drive.google.com/file/d/1xbxi2dWPbSalk1emae7LsDSSua7qNZ1_/view)

## Saved model identities

- Frozen pretrained backbone: `checkpoints/MedSAM2_latest.pt`,
  SHA-256 `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`.
- Independently inspected *one separate saved teacher-resume checkpoint*:
  [initial_native_kl checkpoint](https://drive.google.com/file/d/18mDMPtPe4cMDz9cK-4fIMj8JQJ2kZuqs/view),
  SHA-256 `96d48afe993826436f3b799223592c6e1829f6970f25ff9ca2bc2e00692a19fd`.
  It contains 10 tensors for five layers and **12,736 parameters**.
  This file is not asserted to be the exact model that produced the
  four-patient 0.5144 result.

## Reconstruction status

The historical adapter source files
`modeling/highres_skip_ablation_v1.py`,
`training/highres_initial_clip_v1.py` and
`scripts/run_medsam2_spatial_video_512_v1.py`
are missing from the original complete GitHub experiment branch, though
their SHA256 identities and execution receipts survive in Drive.
The new `modeling/medical_highres/` package reproduces **parameter
geometry, API structure and source-verified checkpoint loading**. Its forward
equation and training protocol are an explicit reconstruction and must NOT
inherit the historical scores.

For architecture and limitations, see
[Medical adapter architecture](../../docs/MEDICAL_HIGHRES_ARCHITECTURE.md).

## New experiment pipeline

`training/train_medical_highres_adapter.py` can train this new adapter from
frozen, TRAIN-only decoder captures and the verified MedSAM2-latest backbone.
It uses teacher KL weight 0.1 by default and records distinct new hashes.
It is not the original four-frame video clip pipeline. To evaluate it, conduct
a separate video experiment with a frozen cohort, original prompt budget,
the same native baseline and an independent per-patient evaluation.

**Research contribution wording:** proposed high-resolution feature injection
with native-teacher regularization, together with a verified negative/mixed
four-patient exploratory ablation. Do not call it an established improvement.

## Clean interface

Use `adeseg.backbones.medsam2`, `adeseg.adapters.high_resolution_skip`, `adeseg.inference.medical_decoder_integration` and `adeseg.losses.teacher_kl`. The original implementation is absent; saved 4-sequence Dice figures cannot be transferred to the reconstructed source. Checkpoint SHA and architecture must be confirmed before inference.

Train interface: `python -m scripts.train_med --help`.

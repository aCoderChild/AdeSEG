# Experiment record

## Recurrent Fusion Memory protocol

Recurrent Fusion Memory retains one recurrent spatial mask-memory state and
MedSAM2's normal prompt plus recent object-pointer history. Its main component
is a two-frame 3-D convolution inspired by RDE-VOS. The optional scalar gate,
inspired by the role of gating in LiVOS, is retained as an ablation.

Training freezes MedSAM2, updates state from predicted masks, and uses Dice+BCE
only on future-frame masks.

## Superseded CPU smoke result

All rows use CPU, seed 0, identical seq16--19 YOLO prompt records, and the
same MedSAM2 checkpoint.

| Method | Sequence-macro Dice | Sequence-macro IoU | Temporal IoU |
|---|---:|---:|---:|
| Native MedSAM2 | 0.7655 | 0.7048 | 0.6019 |
| RGM initialization | 0.7815 | 0.7177 | 0.6294 |
| RGM after tiny seq2/3 training (100 steps) | 0.6310 | 0.5310 | 0.7204 |

These numbers were produced before train--inference parity was verified. That
trainer used different image resizing, a different pointer time normalization,
and a different memory-encoder input path. They are retained only as a record
of the earlier prototype and must not be used to assess RGM.

The valid protocol now freezes one first-detection YOLO prompt per sequence,
uses the native decoder's multimask policy, reuses MedSAM2's native memory
encoder, and verifies a fixed CPU clip against inference before training. The
next experiment is the staged curriculum on seq2--15 with seq16--19 frozen for
validation.

## Corrected recurrent-fusion CPU curriculum

All artifacts for these runs are in the Google Drive `outputs/rgm_curriculum_cpu_20260928`
folder. The fixed `seq2` four-frame parity run passed at tolerance 0.02: the
largest prediction-logit difference was 2.3e-5 and the largest recurrent-state
difference was 9.8e-5. Candidate-memory differences are limited to the
intentional bfloat16 compact-output storage used by inference.

The 0.5-confidence first-detection prompt protocol found no YOLO box for
`seq7`; it is excluded from RGM training rather than being replaced with a GT
prompt. Validation uses the same frozen records for every row.

| Method | Sequence-macro Dice | Sequence-macro IoU | Notes |
|---|---:|---:|---|
| Native MedSAM2 | 0.7655 | 0.7048 | reference |
| Recurrent fusion, 20-step gate smoke | 0.7808 | 0.7171 | seq2/3 only; gate remained about 0.1 |
| Recurrent fusion, Stage 2, 8-frame joint | **0.7881** | **0.7242** | selected checkpoint |
| Recurrent fusion, Stage 3, 16-frame joint | 0.7825 | 0.7178 | stopped: validation declined |

The Stage 2 improvement is validation evidence only: checkpoint selection used
seq16--19, and no held-out test result should be claimed. The gate remained
close to its EMA initialization, so these runs do not yet establish that
reliability-adaptive gating provides the gain. Recurrent fusion retained exactly 524,288
bytes (0.5 MiB) of spatial state per sequence. Native retained 14.9--24.8 MiB
of spatial state in the matched CPU runs; this is not a peak-memory comparison.

## Fixed-gate attribution and held-out test

Artifacts are in Google Drive under
`outputs/rgm_attribution_heldout_cpu_20260928`. The fixed-gate ablation used
the same training sequences, 8-frame clips, seed, frozen MedSAM2 checkpoint,
and 50 Conv3D optimization steps as the selected Stage-2 joint phase. It fixed
the update coefficient at 0.1 and trained only the temporal Conv3D fusion.

| Validation method (seq16--19) | Sequence-macro Dice | Sequence-macro IoU |
|---|---:|---:|
| Fixed gate 0.1 + learned Conv3D | 0.7878 | 0.7239 |
| Full recurrent fusion | 0.7881 | 0.7242 |

The learned gate changed the validation Dice by only +0.0003. These data do
not establish a useful reliability-gating effect; the validation result is
attributable to the learned Conv3D fusion under this protocol.

The selected Stage-2 checkpoint was then evaluated once on held-out
`seq20--23`. Native and recurrent fusion used identical frozen first-detection YOLO prompts,
the same MedSAM2 checkpoint, and seed 0.

| Held-out method (seq20--23) | Sequence-macro Dice | Sequence-macro IoU | Retained spatial memory | Mean CPU FPS |
|---|---:|---:|---:|---:|
| Native MedSAM2 | 0.4489 | 0.4190 | 17.63 MiB | 5.50 |
| Stage-2 recurrent fusion | 0.4243 | 0.3952 | 0.50 MiB | 6.50 |

Recurrent fusion reduced retained spatial-state storage by 97.2%, but lost 0.0246 Dice and
0.0238 IoU on held-out sequences. It was lower on `seq21` and `seq22`, tied
on the failed `seq23` prompt case, and only marginally higher on `seq20`.
Therefore, this one-seed experiment does not support a claim that RGM improves
segmentation over native MedSAM2. The retained-storage measurement is not a
peak-memory measurement, and the four-sequence held-out result remains too
small for a robust performance claim.

## Held-out failure analysis

The paired overlays and numeric distance analysis are in
`outputs/rgm_attribution_heldout_cpu_20260928/failure_analysis`. RFM denotes
Recurrent Fusion Memory in this table.

| Sequence | Native Dice | RFM Dice | RFM - Native |
|---|---:|---:|---:|
| seq20 | 0.3589 | 0.3598 | +0.0009 |
| seq21 | 0.7349 | 0.7073 | -0.0276 |
| seq22 | 0.6838 | 0.6121 | -0.0717 |
| seq23 | 0.0179 | 0.0179 | +0.0000 |

| Frames after prompt | Native Dice | RFM Dice | RFM - Native |
|---|---:|---:|---:|
| 1--5 | 0.2691 | 0.1409 | -0.1282 |
| 6--10 | 0.4662 | 0.3686 | -0.0976 |
| 11--20 | 0.4984 | 0.3356 | -0.1628 |
| >20 | 0.3242 | 0.3841 | +0.0599 |

RFM does not exhibit a monotonic long-term failure: it loses most in the
early and middle windows, then is higher in the aggregate >20 window. Paired
overlays show that `seq22` loses the target under an early change in its
apparent position, scale, and illumination, while `seq21` has a large middle
segment failure before later recovery. These observations are compatible with
appearance-change sensitivity, but this four-sequence diagnostic does not
separate scale, motion, occlusion, and illumination as independent causes.
`seq23` is a shared failed initialization and cannot identify a memory effect.

## Three-seed validation check

Artifacts are in Google Drive under
`outputs/recurrent_fusion_multiseed_cpu_20260928`. Seeds 1 and 2 repeated the
existing seed-0 protocol exactly: 50 gate-only steps, then 50 joint
gate-plus-Conv3D steps, with the same frozen `seq2--15` prompts and
`seq16--19` validation prompts. Held-out sequences were not used.

| Seed | Sequence-macro Dice | Sequence-macro IoU |
|---|---:|---:|
| 0 | 0.788086 | 0.724225 |
| 1 | 0.788085 | 0.724224 |
| 2 | 0.788089 | 0.724230 |
| Mean ± sample SD | 0.788086 ± 0.000002 | 0.724226 ± 0.000003 |

The near-zero variation shows that this CPU protocol is effectively
deterministic under these seeds. It verifies reproducibility of the validation
number, not independent generalization: the architecture starts from an
identity Conv3D update, clips and prompt records are fixed, and the held-out
result remains lower than native MedSAM2.

# Experiment record

## RGM-MedSAM2 protocol

RGM-MedSAM2 retains one recurrent spatial mask-memory state and MedSAM2's
normal prompt plus recent object-pointer history. Its candidate state is
produced by a two-frame 3-D convolution inspired by RDE-VOS. A scalar gate,
inspired by the role of gating in LiVOS, receives decoder-predicted IoU and
global summaries of the preceding state, candidate state, and their absolute
difference. Ground-truth IoU is never supplied to the gate.

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

## Corrected RGM CPU curriculum

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
| RGM, 20-step gate smoke | 0.7808 | 0.7171 | seq2/3 only; gate remained about 0.1 |
| RGM Stage 2, 8-frame joint | **0.7881** | **0.7242** | selected checkpoint |
| RGM Stage 3, 16-frame joint | 0.7825 | 0.7178 | stopped: validation declined |

The Stage 2 improvement is validation evidence only: checkpoint selection used
seq16--19, and no held-out test result should be claimed. The gate remained
close to its EMA initialization, so these runs do not yet establish that
reliability-adaptive gating provides the gain. RGM retained exactly 524,288
bytes (0.5 MiB) of spatial state per sequence. Native retained 14.9--24.8 MiB
of spatial state in the matched CPU runs; this is not a peak-memory comparison.

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

## CPU smoke result

All rows use CPU, seed 0, identical seq16--19 YOLO prompt records, and the
same MedSAM2 checkpoint.

| Method | Sequence-macro Dice | Sequence-macro IoU | Temporal IoU |
|---|---:|---:|---:|
| Native MedSAM2 | 0.7655 | 0.7048 | 0.6019 |
| RGM initialization | 0.7815 | 0.7177 | 0.6294 |
| RGM after tiny seq2/3 training (100 steps) | 0.6310 | 0.5310 | 0.7204 |

At initialization, the compressor returns the candidate and the gate is 0.1,
so the update is exactly EMA alpha=0.1. Retained spatial state is 524,288 bytes
(0.5 MiB) per evaluated sequence. The tiny run has nonzero fusion gradients,
changed fusion parameters, and no MedSAM2 parameter gradients, but its
validation drop means it is not evidence for an effective learned gate.

The next valid experiment is full training on seq2--15 with validation fixed at
seq16--19. The learned RGM checkpoint must improve over its fixed-EMA
initialization before any learned-fusion claim is made.

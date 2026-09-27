# Experiment record

## Archived compact-memory ablation

The compact-memory experiments are retained as a negative result, not as the
active method.

| Method | Sequence-macro Dice | Sequence-macro IoU |
|---|---:|---:|
| Native MedSAM2 | 0.7753 | 0.7134 |
| Fixed EMA, alpha=0.1 | 0.1543 | 0.1532 |
| Adaptive projection | 0.1498 | 0.1486 |
| Gate-only curriculum | 0.1518 | 0.1509 |

All rows used the same seq16--19 prompt records and MedSAM2 checkpoint. The
gate-only implementation was verified to reproduce fixed EMA alpha=0.1 exactly
before training. Its validation result did not exceed fixed EMA.

The compact state retained 0.5 MiB of spatial-state tensors in the seq16 smoke
run. This measures retained spatial state only; it is not a total peak-memory
claim.

## Active next diagnostic

For each adenoid frame with predicted and ground-truth adenoid/airway masks:

1. score regional Dice and IoU;
2. compute the protocol-defined obstruction ratio for both masks;
3. inspect whether ratio error remains high when regional Dice is acceptable;
4. aggregate valid, clinically stable frame measurements only after the
   protocol defines that criterion.

Only if this diagnostic identifies a genuine mismatch should a
measurement-aware training objective be introduced.

## Next temporal direction

The next candidate is not another EMA variant. It is a bounded-memory study
motivated by LiVOS, RDE-VOS, XMem, and PNS+. See
[`docs/constant_memory_direction.md`](docs/constant_memory_direction.md) for
the verified equations, source repositories, and the MedSAM2 integration
boundary.

# Results Analysis: Reliability-Gated MedSAM2, Prompt Strides 1/5/10

## 1. Scope and result provenance

This report analyzes the three completed full-dataset runs produced by
`experiments/reliability_gated_memory_experiment.py`:

| Run | Google Drive result folder | Prompt source | Gate threshold |
|---|---|---|---:|
| Stride 1 | `AdeSEG/outputs/reliability_gated_stride1` | Frame-specific GT box | 0.35 |
| Stride 5 | `AdeSEG/outputs/reliability_gated_stride5` | Frame-specific GT box | 0.35 |
| Stride 10 | `AdeSEG/outputs/reliability_gated_stride10` | Frame-specific GT box | 0.35 |

All conclusions below were calculated from the saved `experiment_notes.json`,
candidate logs and masks, gated logs and masks, `metrics.csv`, per-sequence
evaluation CSVs, and global summaries. The three runs used the same 23
sequences, checkpoint, prompt type, unlimited prompt count, CPU inference, and
gate configuration. Only `video_prompt_stride` changed.

The saved outputs contain 2,225 predictions per run and report zero missing
predictions. The dataset has 1,710 foreground-positive frames and 515 empty-GT
frames. `seq1` (36 frames) and `seq7` (48 frames) are completely empty.

> **Interpretation boundary:** these are oracle-prompt experiments because each
> eligible positive frame receives a box derived from its own GT annotation.
> They measure segmentation/propagation behavior under controlled prompting,
> not fully automatic detection-to-segmentation performance.

## 2. Experiment flow

```text
                           ONLY VARIABLE CHANGED
                    +-------------------------------+
                    | prompt stride = 1, 5, or 10  |
                    +---------------+---------------+
                                    |
RGB frames + frame-specific GT boxes|  eligible frame i when i % stride == 0
                    |               v
                    +------> MedSAM2 candidate generation
                                    |
                                    v
                         raw candidate mask C_t
                                    |
                 +------------------+------------------+
                 | reliability(C_t, previous state, I_t)|
                 +------------------+------------------+
                                    |
                      reliability >= 0.35 ?
                         /                     \
                       yes                      no
                        |                        |
                 output current          hold prior mask
                                           (max 3 times;
                                        then force current)
                         \                     /
                          +---------+---------+
                                    v
                            gated output G_t
                                    |
                                    v
                    Dice / IoU / temporal metrics
```

The reliability gate is downstream of MedSAM2. It does not alter SAM2's
internal memory; it selects either the current binary candidate or an unwarped
previous binary output.

## 3. Executive result

| Metric | Stride 1 | Stride 5 | Stride 10 | Best |
|---|---:|---:|---:|---|
| Mean Dice ↑ | **0.6919** | 0.6419 | 0.6255 | Stride 1 |
| Mean IoU ↑ | **0.6499** | 0.5948 | 0.5769 | Stride 1 |
| Temporal IoU ↑ | 0.6750 | 0.6934 | **0.7051** | Stride 10 |
| Centroid shift, px ↓ | 72.54 | 69.25 | **66.23** | Stride 10 |
| Area change ↓ | 0.01840 | 0.01918 | **0.01638** | Stride 10 |
| Mean reliability ↑ | **0.5619** | 0.5543 | 0.5535 | Stride 1 |
| Accepted updates | 1,934 | 1,894 | 1,885 | — |
| Rejected updates | 291 | 331 | 340 | — |
| Rejection rate | **13.08%** | 14.88% | 15.28% | Stride 1 |
| Prompt seeds | 1,710 | 349 | **174** | Stride 10 uses fewest |
| Total elapsed time | 15m 57s | 12m 45s | **9m 53s** | Stride 10 |

### Main tradeoff

```text
Spatial accuracy (Dice)                 Temporal persistence (temporal IoU)

stride 1   0.6919  ####################  0.6750  ##################
stride 5   0.6419  ##################    0.6934  ###################
stride 10  0.6255  #################     0.7051  ####################
                     best <----------     ----------> best
```

Increasing stride from 1 to 10 reduces Dice by 0.0664 (9.60% relative) and IoU
by 0.0730 (11.23% relative), while increasing temporal IoU by 0.0301 (4.46%
relative). Centroid movement falls by 8.70% and area change falls by 10.94%.

This is not evidence that stride 10 segments better. It indicates that sparse
prompting plus output holding makes predictions change less, even when the held
mask is spatially wrong.

## 4. Prompt-density and runtime tradeoff

| Quantity | Stride 1 | Stride 5 | Stride 10 |
|---|---:|---:|---:|
| Positive GT frames | 1,710 | 1,710 | 1,710 |
| Actual prompt seeds | 1,710 | 349 | 174 |
| Seed reduction vs stride 1 | — | 79.6% | 89.8% |
| Candidate stage | 14m 56s | 11m 43s | 8m 52s |
| Complete run | 15m 57s | 12m 45s | 9m 53s |
| Complete runtime reduction | — | 20.1% | 38.0% |

Prompt count falls almost in inverse proportion to stride, but total time does
not. Video encoding, propagation, mask serialization, Drive synchronization,
gating, and evaluation still process all 2,225 frames. Consequently, stride 10
removes 89.8% of seeds but saves only 38.0% wall time.

```text
Prompts                 Runtime                    Gated Dice
1:  1710  ---------->   15m57s  ---------->       0.6919
5:   349  --79.6%---->  12m45s  --20.1%---->      0.6419
10:  174  --89.8%---->   9m53s  --38.0%---->      0.6255
```

Stride 5 is the practical compromise if prompt cost is important: it saves
1,361 prompts and about 3m12s relative to stride 1, but loses 0.0500 Dice.
Stride 10 saves only another 2m52s and 175 prompts relative to stride 5 while
losing another 0.0165 Dice globally.

## 5. Raw candidate versus gated output

The experiment intentionally has no separate baseline run. However, its saved
`_candidate` masks are the direct input to the gate. Re-evaluating those masks
against the same GT provides a diagnostic of exactly what the gate changed.

| Stride | Candidate Dice | Gated Dice | Gate Δ Dice | Candidate IoU | Gated IoU | Gate Δ IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.7824** | 0.6919 | **-0.0905** | **0.7390** | 0.6499 | **-0.0891** |
| 5 | **0.7253** | 0.6419 | **-0.0834** | **0.6780** | 0.5948 | **-0.0832** |
| 10 | **0.7060** | 0.6255 | **-0.0805** | **0.6571** | 0.5769 | **-0.0801** |

| Stride | Candidate temporal IoU | Gated temporal IoU | Gate Δ temporal IoU |
|---:|---:|---:|---:|
| 1 | 0.5759 | **0.6750** | +0.0992 |
| 5 | 0.5888 | **0.6934** | +0.1045 |
| 10 | 0.6009 | **0.7051** | +0.1042 |

The gate consistently exchanges spatial correctness for temporal persistence.
This is the most important result in the study: **the current gate does not
improve segmentation accuracy at any tested stride.**

### Positive and empty frames

| Stride | Candidate Dice, positive GT | Gated Dice, positive GT | Δ | Candidate empty-frame correctness | Gated empty-frame correctness | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.9063** | 0.8395 | -0.0668 | **0.3709** | 0.2019 | -0.1689 |
| 5 | **0.8198** | 0.7674 | -0.0524 | **0.4117** | 0.2252 | -0.1864 |
| 10 | **0.7900** | 0.7408 | -0.0492 | **0.4272** | 0.2427 | -0.1845 |

For an empty GT frame, Dice is 1 only when the prediction is also empty and 0
when any foreground remains. Thus “empty-frame correctness” above is also the
fraction of empty frames predicted empty. The mean predicted foreground area on
empty frames increases after gating:

| Stride | Candidate area on empty GT | Gated area on empty GT |
|---:|---:|---:|
| 1 | 0.0428 | **0.0636** |
| 5 | 0.0427 | **0.0728** |
| 10 | 0.0403 | **0.0507** |

The gate therefore worsens false-positive persistence, especially when a polyp
leaves the frame.

## 6. Why rejected updates are usually harmful

| Stride | Rejected frames | Rejected candidate Dice | Final gated Dice | Improved | Equal | Worsened |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 291 | **0.7271** | 0.0354 | 6 | 64 | **221** |
| 5 | 331 | **0.6403** | 0.0797 | 37 | 66 | **228** |
| 10 | 340 | **0.6036** | 0.0767 | 44 | 72 | **224** |

Across all strides, most rejected candidates were more accurate than the mask
that replaced them. At stride 1, 221 of 291 rejections worsen Dice and only six
improve it.

The failure mechanism follows directly from the implemented state machine:

```text
t-1: polyp visible              t: polyp disappears

GT       [polyp]                [empty]
candidate[polyp]                [empty]       <- correct new observation
state    [polyp]  ------------> [polyp]
reliability                     very low      <- temporal/area disagreement
gate decision                   reject empty candidate
final                           [old polyp]    <- false positive, but temporally stable
```

Current-empty versus previous-nonempty masks receive a 0.25 multiplier. The
empty candidate is therefore likely to fall below 0.35 precisely when it is
correct. In addition, a forced blank output does not clear the stored non-empty
state, so stale foreground can return on subsequent rejected frames.

Rejected-frame composition confirms that this is not only an empty-frame
problem:

| Stride | Rejected positive-GT frames | Rejected empty-GT frames |
|---:|---:|---:|
| 1 | 140 | 151 |
| 5 | 186 | 145 |
| 10 | 194 | 146 |

The held prior also fails on positive frames when motion, scale change, or
boundary evolution makes the unwarped old mask stale.

## 7. Reliability calibration audit

### Reliability versus candidate correctness

| Stride | Positive frames Pearson / Spearman | Empty frames Pearson / Spearman |
|---:|---:|---:|
| 1 | 0.153 / 0.204 | **-0.760 / -0.795** |
| 5 | 0.369 / 0.286 | **-0.793 / -0.810** |
| 10 | 0.407 / 0.292 | **-0.793 / -0.830** |

Reliability is only weak-to-moderately associated with candidate Dice on
positive frames. On empty frames it is strongly *negatively* associated with
correctness: correct blank masks tend to receive the lowest reliability.

### Reliability bins

| Stride | Reliability bin | Frames | Candidate Dice | Gated Dice | Rejection rate |
|---:|---:|---:|---:|---:|---:|
| 1 | [0.0, 0.2) | 104 | **0.9987** | 0.1439 | 85.6% |
| 1 | [0.2, 0.35) | 318 | **0.7177** | 0.3644 | 63.5% |
| 1 | [0.35, 0.5) | 333 | 0.6493 | 0.6493 | 0% |
| 1 | [0.5, 0.7) | 676 | 0.7882 | 0.7882 | 0% |
| 1 | [0.7, 1.0] | 794 | 0.8308 | 0.8308 | 0% |
| 5 | [0.0, 0.2) | 147 | **0.7959** | 0.1728 | 84.4% |
| 5 | [0.2, 0.35) | 330 | **0.6819** | 0.3973 | 62.7% |
| 5 | [0.35, 0.5) | 310 | 0.5884 | 0.5884 | 0% |
| 5 | [0.5, 0.7) | 610 | 0.7561 | 0.7561 | 0% |
| 5 | [0.7, 1.0] | 828 | 0.7587 | 0.7587 | 0% |
| 10 | [0.0, 0.2) | 150 | **0.7533** | 0.1523 | 83.3% |
| 10 | [0.2, 0.35) | 362 | **0.6457** | 0.3998 | 59.4% |
| 10 | [0.35, 0.5) | 293 | 0.5551 | 0.5551 | 0% |
| 10 | [0.5, 0.7) | 562 | 0.7431 | 0.7431 | 0% |
| 10 | [0.7, 1.0] | 858 | 0.7504 | 0.7504 | 0% |

The most striking calibration failure is stride 1's lowest bin: raw candidates
have 0.9987 mean Dice, yet the gate changes many of them into outputs with
0.1439 Dice. Low reliability here means “different from the previous state,”
not “wrong.”

This behavior is expected from the formula. `mask_confidence` is hardcoded to
0.5, so the score has no model-confidence signal. Temporal IoU and area
consistency dominate decisions, making the gate a change detector rather than
a calibrated correctness estimator.

## 8. Accuracy as distance from a scheduled prompt

Offset zero is a stride-eligible positive frame and therefore has a GT-derived
prompt. Other offsets rely on video propagation. The values below use only
positive-GT frames.

### Stride 5

| Frame offset (`index % 5`) | Positive frames | Gated Dice | Gated IoU |
|---:|---:|---:|---:|
| 0, prompted | 349 | **0.8217** | **0.7696** |
| 1 | 338 | 0.7685 | 0.7020 |
| 2 | 349 | 0.7454 | 0.6800 |
| 3 | 336 | 0.7605 | 0.6961 |
| 4 | 338 | 0.7400 | 0.6815 |

### Stride 10

| Offset | Positive frames | Gated Dice | Gated IoU |
|---:|---:|---:|---:|
| 0, prompted | 174 | **0.8261** | **0.7704** |
| 1 | 173 | 0.7636 | 0.6975 |
| 2 | 178 | 0.7353 | 0.6708 |
| 3 | 168 | 0.7587 | 0.6911 |
| 4 | 169 | 0.7188 | 0.6608 |
| 5 | 175 | **0.7026** | **0.6351** |
| 6 | 165 | 0.7386 | 0.6742 |
| 7 | 171 | 0.7187 | 0.6541 |
| 8 | 168 | 0.7207 | 0.6587 |
| 9 | 169 | 0.7232 | 0.6624 |

Prompted offsets are clearly strongest. The non-monotonic recovery after
offset 5 is compatible with bidirectional video propagation and sequence
content; offset modulo stride is not a pure causal “frames since last prompt”
measure. Nevertheless, the approximately 0.10 Dice gap between prompted stride
10 frames and offsets 4–9 shows the cost of sparse conditioning.

## 9. Dataset-level aggregation effects

| Aggregation | Stride 1 | Stride 5 | Stride 10 |
|---|---:|---:|---:|
| Frame-weighted Dice, all frames | 0.6919 | 0.6419 | 0.6255 |
| Frame-weighted Dice, excluding all-empty seq1/seq7 | 0.6798 | 0.6279 | 0.6108 |
| Sequence-macro mean, all sequences | 0.7062 | 0.6660 | 0.6545 |
| Sequence-macro mean, excluding seq1/seq7 | 0.6782 | 0.6342 | 0.6216 |
| Sequence median, excluding seq1/seq7 | 0.7189 | 0.6870 | 0.6706 |

`seq1` and `seq7` contain no foreground, receive no prompts, and are predicted
perfectly blank at all strides. Their 84 frames contribute Dice 1.0 and inflate
the headline means. This does not invalidate the reported global metric, but a
foreground-task comparison should always include the positive-frame and
all-empty-excluded views.

## 10. Per-sequence results

The table reports gated Dice. “Seeds” are shown as stride 1 / stride 5 /
stride 10.

| Seq | Frames | Positive | Seeds 1/5/10 | Dice S1 | Dice S5 | Dice S10 | Best |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 36 | 0 | 0/0/0 | 1.0000 | 1.0000 | 1.0000 | Tie (all empty) |
| 2 | 63 | 60 | 60/12/7 | **0.8833** | 0.8371 | 0.8325 | S1 |
| 3 | 15 | 15 | 15/3/2 | **0.9426** | 0.8812 | 0.8872 | S1 |
| 4 | 48 | 46 | 46/10/5 | **0.8548** | 0.7626 | 0.7574 | S1 |
| 5 | 250 | 199 | 199/39/20 | **0.7189** | 0.7041 | 0.7022 | S1 |
| 6 | 91 | 62 | 62/13/6 | **0.6130** | 0.5819 | 0.5498 | S1 |
| 7 | 48 | 0 | 0/0/0 | 1.0000 | 1.0000 | 1.0000 | Tie (all empty) |
| 8 | 73 | 54 | 54/11/6 | **0.6510** | 0.5664 | 0.5119 | S1 |
| 9 | 51 | 42 | 42/8/4 | **0.7393** | 0.7247 | 0.6988 | S1 |
| 10 | 25 | 7 | 7/1/0 | 0.3128 | 0.3802 | **0.7200*** | Artifact |
| 11 | 228 | 136 | 136/28/14 | **0.5399** | 0.4654 | 0.4406 | S1 |
| 12 | 250 | 250 | 250/50/25 | **0.8678** | 0.8624 | 0.8581 | S1 |
| 13 | 250 | 199 | 199/39/19 | **0.6407** | 0.5421 | 0.5539 | S1 |
| 14 | 249 | 200 | 200/41/21 | **0.6083** | 0.4916 | 0.4579 | S1 |
| 15 | 116 | 116 | 116/24/12 | 0.7961 | **0.8130** | 0.8082 | S5 |
| 16 | 40 | 33 | 33/6/3 | **0.7941** | 0.7656 | 0.7384 | S1 |
| 17 | 63 | 44 | 44/10/6 | **0.6249** | 0.4944 | 0.4592 | S1 |
| 18 | 63 | 56 | 56/11/6 | **0.7837** | 0.6870 | 0.6444 | S1 |
| 19 | 56 | 55 | 55/12/6 | **0.8802** | 0.8462 | 0.8215 | S1 |
| 20 | 52 | 31 | 31/6/2 | **0.4129** | 0.3581 | 0.2051 | S1 |
| 21 | 56 | 25 | 25/6/2 | 0.3856 | **0.4892** | 0.4316 | S5 |
| 22 | 46 | 44 | 44/10/5 | **0.7521** | 0.7103 | 0.6706 | S1 |
| 23 | 56 | 36 | 36/9/3 | **0.4407** | 0.3552 | 0.3046 | S1 |

\* `seq10` stride 10 receives zero prompts because none of its seven positive
frames lies on an index divisible by 10. The output is blank on all 25 frames:
18 empty frames score 1 and seven positive frames score 0, producing the
misleading mean `18/25 = 0.72`. Its positive-frame Dice is 0. This is a metric
artifact, not a successful stride-10 segmentation.

### Positive-frame Dice for diagnostically important sequences

| Sequence | S1 | S5 | S10 | Interpretation |
|---:|---:|---:|---:|---|
| 10 | **0.9742** | 0.9294 | **0.0000** | Scheduled stride misses every positive frame at S10. |
| 12 | **0.8678** | 0.8624 | 0.8581 | Most robust sequence; foreground exists in every frame. |
| 15 | 0.7961 | **0.8130** | 0.8082 | Sparse prompts do not hurt and slightly help. |
| 17 | **0.8947** | 0.6398 | 0.5893 | Large propagation/gating degradation. |
| 20 | **0.6603** | 0.4071 | 0.2473 | Hardest positive segmentation under sparse prompting. |
| 21 | 0.7037 | **0.7358** | 0.7267 | Sparse conditioning improves this sequence. |
| 23 | **0.6578** | 0.5248 | 0.4461 | Strong stride sensitivity. |

Outside the two all-empty sequences, stride 1 beats stride 5 on 18 of 21
sequences and beats stride 10 on 18 of 21. Stride 5's genuine wins are `seq15`
and `seq21`; its apparent `seq10` win must be read alongside positive-only
performance. Stride 10's largest apparent gain is the same `seq10` artifact.

## 11. Paired sequence-level statistics

The paired analysis excludes all-empty `seq1` and `seq7`, leaving 21 sequence
pairs.

| Comparison | Mean paired Dice Δ | Median Δ | Bootstrap 95% CI | Wins/Losses | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| S5 − S1 | -0.0440 | -0.0462 | [-0.0670, -0.0186] | 3 / 18 | 0.00375 |
| S10 − S1 | -0.0566 | -0.0632 | [-0.0999, -0.0006] | 3 / 18 | 0.00137 |
| S10 − S5 | -0.0126 | -0.0259 | [-0.0420, 0.0297] | 3 / 18 | 0.00328 |

The first two comparisons consistently favor stride 1. For S10 versus S5, the
Wilcoxon result reflects the consistent direction (18 losses), while the
bootstrap mean CI crosses zero because the magnitude is uncertain and is
strongly influenced by the positive `seq10` aggregate artifact. This is another
reason to report positive-only and sequence-level results instead of relying on
one global mean.

## 12. Which stride should be selected?

| Objective | Recommended stride | Evidence |
|---|---|---|
| Highest segmentation accuracy | **1** | Best Dice/IoU globally, on positive frames, and on 18/21 nonempty sequence comparisons. |
| Best prompt/accuracy compromise | **5** | 79.6% fewer prompts with a 0.0500 Dice loss; materially better than S10 on most sequences. |
| Lowest runtime / fewest prompts | **10** | 38.0% faster and 89.8% fewer seeds than S1. |
| Smoothest-looking masks | **10** | Highest temporal IoU and lowest centroid shift/area change. |
| Scientific default for current code | **1**, with gate disabled or repaired | Dense prompts are strongest, but the present gate still reduces candidate Dice. |

Stride 10 should not be selected on temporal metrics alone. Its stability is
partly created by holding stale masks and by producing blanks when the prompt
schedule misses short positive intervals.

## 13. Prioritized corrective experiments

| Priority | Experiment | Exact code/config target | Expected result | Risk |
|---:|---|---|---|---|
| 1 | Clear state after an accepted/forced blank | `apply_reliability_gate` state transition in `scripts/utils/reliability_gate.py` | Prevent stale foreground from reappearing after disappearance. | May make recovery slower if blank was erroneous. |
| 2 | Treat empty transitions separately | Empty-current penalty and threshold logic | Improve empty-frame correctness; stop rejecting correct disappearances. | Requires validation to avoid flicker. |
| 3 | Evaluate pass-through candidate as a diagnostic in reports | Evaluation stage only; no second neural run | Makes gate gain/loss visible in every experiment. | None beyond extra evaluation time. |
| 4 | Tune threshold on validation data, including `tau=0` | `RELIABILITY_THRESHOLD` / threshold ablation | `tau=0` should reproduce candidates and is the critical control. | Test-set tuning would overfit. |
| 5 | Replace constant 0.5 confidence | Candidate logits/image-predictor IoU/object score plumbing | Reliability may correlate with actual mask quality. | Larger interface and storage change. |
| 6 | Motion-align prior before comparison/holding | Temporal score and held-mask path | Reduce stale-mask errors during camera/polyp motion. | Added compute and motion-estimation failures. |
| 7 | Make scheduling foreground-aware for research ablation | Prompt scheduler | Avoid zero-prompt positive cases such as `seq10` S10. | Not deployable if based on GT; label clearly as oracle. |
| 8 | Validate on YOLO prompts | `--prompt-source yolo` | Measures realistic detector + propagation errors. | Confounds segmentation with detector misses. |

The first experiment alone is insufficient because a correct first blank may be
rejected before it can clear the state. The empty-transition decision and state
clearing should be tested together.

### Minimum next ablation matrix

```text
                         threshold
                    0.00   0.20   0.35   0.50
                 +------+------+------+------+
stride 1         | pass |      |current|      |
stride 5         | pass |      |current|      |
stride 10        | pass |      |current|      |
                 +------+------+------+------+

For every cell report:
  all-frame Dice/IoU
  positive-frame Dice/IoU
  empty-frame correctness
  temporal IoU
  rejected-candidate gain/loss
```

Using `tau=0` is not reintroducing a separate baseline inference run: it reuses
the same saved candidates and provides the necessary pass-through control.

## 14. Limitations of this analysis

| Limitation | Consequence |
|---|---|
| GT-derived boxes | Results are oracle-assisted and overstate automatic-system performance. |
| One checkpoint and one dataset split | Model and dataset generalization are unverified. |
| Test set used for comparison | Do not tune the final gate threshold directly on these 23 sequences. |
| Binary saved candidates | Logit uncertainty and model confidence cannot be reconstructed. |
| Frame-weighted global metrics | Long sequences dominate; all-empty frames can inflate Dice. |
| Temporal IoU rewards persistence | A frozen wrong mask can score well temporally. |
| Centroid shift is in pixels | Cross-resolution interpretation would require diagonal normalization. |
| CPU timing with Drive output | Timing includes filesystem/synchronization effects and is not a pure model benchmark. |

## 15. Final conclusions

1. **Stride 1 is the accuracy winner.** It achieves Dice 0.6919 and IoU 0.6499,
   and wins 18 of 21 nonempty sequence comparisons against both sparse strides.
2. **Stride 5 is the strongest efficiency compromise.** It removes 79.6% of
   prompt seeds and saves 20.1% total time for a 0.0500 absolute Dice cost.
3. **Stride 10 is fastest and smoothest, not most accurate.** Its temporal gains
   coexist with lower spatial accuracy and a zero-prompt failure on `seq10`.
4. **The present reliability gate is harmful to Dice at all three strides.** It
   lowers raw-candidate Dice by 0.0805–0.0905 while increasing temporal IoU by
   about 0.10.
5. **Reliability is miscalibrated for object disappearance.** Correct blank
   candidates receive very low scores, so the gate preserves stale foreground.
6. **The next work should repair and validate the gate before optimizing its
   threshold.** Empty transitions, state clearing, real confidence, and a
   pass-through `tau=0` control are the highest-value changes.

The scientifically defensible headline is therefore:

> Denser prompting improves MedSAM2 spatial accuracy on this dataset. Sparse
> prompting improves runtime and apparent temporal smoothness. The current
> post-hoc reliability gate amplifies smoothness by retaining prior masks, but
> reduces segmentation accuracy—especially at object disappearance—and should
> be treated as an unsuccessful gate design pending correction.

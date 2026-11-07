# Regional memory-token selection: Run170

**Status:** engineering passed; diagnostic continuation criterion failed; full propagation not launched

One fixed configuration; PolypGen is exploratory because it has already informed development.
SAM2 weights and native memory-frame selection are fixed. Prompt and latest memory stay full;
older frames retain 25% of tokens per layer. No new confidence threshold or training.

**Scope:** 73 fixed-native-history replay targets on seq2/seq3. Interventions never
write memory. These are immediate diagnostic effects; no full-propagation or
full-cohort efficacy result was produced.

**Verification:** 32 focused tests passed; native replay was bit-identical across
all targets. An offline audit verified 365 mask hashes/metrics, 1,460 layer reads,
56 source files and the source archive/completion hashes.

| Condition | Present Dice | IoU | Boundary F | False alarms / empty frames |
|---|---:|---:|---:|---:|
| native | 0.881663 | 0.805582 | 0.796671 | N/A (0 empty targets) |
| usage | 0.877979 | 0.800930 | 0.789535 | N/A (0 empty targets) |
| foreground | 0.879072 | 0.801960 | 0.791782 | N/A (0 empty targets) |
| role_blind | 0.877983 | 0.800953 | 0.790175 | N/A (0 empty targets) |
| regional | 0.878046 | 0.801048 | 0.790090 | N/A (0 empty targets) |

| Sequence | Native | Attention only | Foreground anchors | Role-blind anchors | Regional anchors |
|---|---:|---:|---:|---:|---:|
| seq2 | 0.888730 | 0.881637 | 0.883938 | 0.881643 | 0.881664 |
| seq3 | 0.874597 | 0.874320 | 0.874205 | 0.874322 | 0.874428 |

Regional minus native: -0.003618; 95% sequence-bootstrap CI [−0.007066, −0.000169]; wins/ties/losses 0/0/2.

Regional minus foreground: -0.001026; 95% sequence-bootstrap CI [−0.002274, +0.000222]; wins/ties/losses 1/0/1.

Regional minus role_blind: +0.000063; 95% sequence-bootstrap CI [+0.000021, +0.000106]; wins/ties/losses 2/0/0.

The regional rule did not improve over native and was below foreground-anchor
selection on average. Its small advantage over role-blind anchors does not meet
the intended improvement. Stop this configuration under the frozen protocol.
Two sequences provide limited scope, and their bootstrap intervals do not
establish population-level significance. False-alarm effects are **unassessed**
because there are no GT-empty scored targets.

Artifacts: `/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/170_regional_token_selection_v1/attempt_01`. Mounted Drive presence does not independently prove remote upload.

[Method](REGION_TOKENS.md) · [Frozen protocol](../protocols/regional_token_selection_v1.json)

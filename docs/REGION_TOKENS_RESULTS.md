# Run171: memory-token contribution diagnostic

**Completed and verified. The current selector failed its frozen ranking criterion; no temporal follow-up was launched.**

Cross-dataset follow-up: Run172 replicates this diagnostic on DAVIS 2017
validation with unchanged selector settings. Its multi-object engineering gate
passed after 70 focused tests; the 30-video cohort is running. No complete
Run172 score is available yet.
Progress and final artifacts are stored in Drive under
`outputs/experiment/172_davis_token_contribution_v1`.

21 PolypGen sequences (seq1/seq7 excluded); 114 fixed targets: 84 GT-present and 30 GT-empty. All pretrained SAM2 weights stay frozen. Each intervention replays the same native history without memory encoding or writeback. These are sampled immediate effects, not full-propagation results.

## Aggregate results

Present metrics: mean within sequence, then equal mean over 21 sequences. False-alarm counts use the 30 sampled GT-empty targets.

| Condition | Removed tokens per older frame/layer | Present Dice | Δ native | IoU | Boundary F | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| Native SAM2 | 0 | 0.714797 | +0.000000 | 0.654425 | 0.622498 | 13/30 |
| Remove rejected A | 128 | 0.714752 | -0.000044 | 0.654355 | 0.622440 | 13/30 |
| Remove rejected B | 128 | 0.714801 | +0.000004 | 0.654437 | 0.622560 | 13/30 |
| Remove rejected-enriched | 256 | 0.714756 | -0.000041 | 0.654366 | 0.622563 | 13/30 |
| Remove retained-enriched | 256 | 0.718508 | +0.003711 | 0.659833 | 0.625411 | 14/30 |
| Remove random | 256 | 0.715647 | +0.000850 | 0.655728 | 0.625193 | 13/30 |

256 tokens = 16 disjoint 4×4-token blocks, or 6.25% of each older frame. Prompt, latest frame and pointers remain full. Rejected A/B are disjoint halves of the joint rejected removal.

## Prespecified ranking tests

Positive differences would support removing rejected groups more safely. Both tests failed that direction and the majority-win requirement.

| Contrast | Mean Dice difference | 95% sequence-bootstrap CI | Holm p | Wins / ties / losses |
|---|---:|---|---:|---:|
| rejected minus retained | -0.003752 | [-0.011455, +0.000871] | 0.1454 | 5 / 1 / 15 |
| rejected minus random | -0.000891 | [-0.002126, -0.000161] | 0.0179 | 5 / 1 / 15 |

## Per-sequence present Dice

Four sampled GT-present frames per sequence. These are not whole-video averages.

| Sequence | Native | Remove rejected | Remove retained | Remove random |
|---|---:|---:|---:|---:|
| seq2 | 0.938914 | 0.938908 | 0.939120 | 0.938902 |
| seq3 | 0.790571 | 0.790762 | 0.801109 | 0.803027 |
| seq4 | 0.841129 | 0.840716 | 0.842849 | 0.842292 |
| seq5 | 0.719722 | 0.719714 | 0.720057 | 0.719653 |
| seq6 | 0.809739 | 0.809712 | 0.811534 | 0.810111 |
| seq8 | 0.232717 | 0.232716 | 0.232742 | 0.232713 |
| seq9 | 0.869895 | 0.869805 | 0.872234 | 0.870917 |
| seq10 | 0.973690 | 0.973702 | 0.973828 | 0.973741 |
| seq11 | 0.334931 | 0.334915 | 0.328340 | 0.335125 |
| seq12 | 0.963256 | 0.963259 | 0.962978 | 0.963299 |
| seq13 | 0.774142 | 0.774081 | 0.772587 | 0.773778 |
| seq14 | 0.816943 | 0.817030 | 0.816581 | 0.817191 |
| seq15 | 0.950556 | 0.950542 | 0.950683 | 0.950791 |
| seq16 | 0.980905 | 0.980936 | 0.982311 | 0.982067 |
| seq17 | 0.787330 | 0.786739 | 0.862000 | 0.788208 |
| seq18 | 0.578593 | 0.578624 | 0.579063 | 0.578551 |
| seq19 | 0.736648 | 0.736648 | 0.736731 | 0.736649 |
| seq20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| seq21 | 0.913085 | 0.913112 | 0.913175 | 0.913169 |
| seq22 | 0.938875 | 0.938879 | 0.939287 | 0.938934 |
| seq23 | 0.059088 | 0.059065 | 0.051455 | 0.059462 |

## What the groups did

Counts below describe the 84 present targets; frames are correlated and are not independent statistical units.

| Removal | Dice improved | Dice worsened | Exact Dice tie |
|---|---:|---:|---:|
| Remove rejected-enriched | 32 | 36 | 16 |
| Remove retained-enriched | 48 | 24 | 12 |
| Remove random | 46 | 24 | 14 |

Pooled removed-group descriptors across memory-frame/layer sets (not equal-sequence means). Attention share refers to the source frame’s cached **first read**, normalized within that frame, not current full-bank attention.

| Groups | Selector-retained fraction | First-read attention share | Soft predicted foreground occupancy | Stored-vector norm |
|---|---:|---:|---:|---:|
| rejected | 0.00% | 2.28% | 3.38% | 7.105 |
| retained | 92.47% | 23.16% | 29.19% | 6.673 |
| random | 25.04% | 6.28% | 8.96% | 6.884 |

## Interpretation

- Removing rejected groups was almost neutral versus native: -0.000041 Dice; secondary 95% CI [-0.000116, +0.000018]. It performed worse than matched random removal in the prespecified comparison.
- Removing retained groups gave +0.003711 Dice, but its secondary CI [-0.000893, +0.011353] includes zero. Seq17 contributes almost the entire mean gain; without seq17, the gain is +0.000163. False alarms also rise from 13/30 to 14/30. This is not a reliable new pruning method.
- Higher cached attention and predicted-foreground occupancy did not identify groups whose removal was consistently more damaging. The current relevance criterion does not establish token usefulness.
- Rejected-group nonadditivity, U(AB)−U(A)−U(B): +9.928e-07 Dice; 95% CI [-3.356e-06, +5.213e-06]. There is no detected average interaction at this budget; this does not prove absence of redundancy.
- Native object scores were nonpositive on 12/84 present targets. None of the three main removals changed the present/absent decision on any present target. All four sampled present frames in seq20 remain at zero Dice.
- An exact Dice tie is not proof a group was unused: rejected removal has one tied-Dice target with a different mask and four with different final logits.

Post hoc examples explain influence, not validation: at seq17/manifest frame 49, retained-group removal raises Dice 0.590977 → 0.866208 and precision 0.424651 → 0.796304 while recall falls slightly (0.971484 → 0.949565), consistent with reduced leakage. At seq3/frame 3, random and retained removal both improve Dice to approximately 0.902 from 0.856. These examples must not be used to invert the selector or choose a new policy after seeing results.

[Illustrative influential cases](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/171_token_contribution_v1/attempt_02/influential_cases.png>) (green: GT; magenta: prediction).

## Verification and limits

- 57 focused tests passed before launch. Seq2/seq3 independent reference parity covered 75 native outputs, with exact native replay on all 73 post-prompt frames.
- All 21 sequence receipts completed: 2,033 native frames including prompts; 684 saved mask/metric records and 456 paired target/layer group plans independently checked. Original weights match across sequences; 60 frozen source files and the checkpoint/manifest hashes match. Primary statistics were recomputed.
- Replay checks protect input clones, tracked bank/cache, pointer inputs and all four layers. Sparse retrieval retains original cell coordinates. No treatment writes memory.
- Effects belong to joint token groups under fixed native histories, not individual vectors or semantically pure regions. Deletion also changes attention normalization.
- Controls match count, local block geometry and frame age; global spatial arrangement is not matched. First-read selector caches were not persisted; intervention block IDs/hashes, source recipe and input hashes are archived.
- One control seed and four present targets per sequence; PolypGen has informed development. No patient independence, held-out confirmation, DAVIS result or full-propagation accuracy improvement is claimed.

**Decision:** stop this selector under the committed rule. Do not launch a temporal pulse or tune/invert the ranking from these outcomes. The diagnostic shows that some group removals can help individual predictions, but this selector does not reliably identify which ones.

Artifacts: `/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/171_token_contribution_v1/attempt_02`. Canonical raw data: `results.json`, `sequences/*/*/frames.jsonl`; verification: `independent_completion_audit.json`, `completion_review.json`. Cloud visibility of the folder and selected artifacts was separately observed; this is not a remote checksum of every file.

[Method](REGION_TOKENS.md) · [Frozen protocol](../protocols/token_contribution_v1.json)

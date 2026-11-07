# Regional memory-token utilisation

**Hypothesis:** predicted object/context information can identify useful memory
vectors and improve frozen SAM2 segmentation without selecting good/bad frames.
The current selector has not validated that hypothesis.

## Selector under test

The implementation reuses pinned [Efficient-SAM2](https://arxiv.org/abs/2602.08224v2)
attention/top-k/sparse retrieval and [Lean-SAM2](https://arxiv.org/abs/2607.19811v1)
prompt anchors, maximum cosine matching and multiplicative score fusion.
It adapts their components, not their complete published systems.

1. Keep native SAM2 frame selection and all original weights frozen.
2. Measure each stored frame’s attention importance on its first read.
3. Obtain foreground/context appearance anchors from the prompt. Pool the exact
   stored frame’s native predicted mask probabilities onto its 64×64 memory grid.
4. Form regional affinity `p*sf + (1-p)*sb`, where `p` is soft foreground occupancy
   and `sf/sb` are maximum anchor similarities. Fuse with first-read importance
   using the author’s operation to rank tokens.

Roles describe predicted positions/context, not pure vector content. All vectors
can contain information from surrounding regions. There is no new confidence
rule, frame-quality gate, decoder training or foundation-weight update.

## Current experiment: Run171

This is a contribution diagnostic, not a new deployed pruning method. Native
trajectories keep every token. At fixed targets, private replays compare:

| Replay | Intervention on each older spatial memory frame, in all four layers |
|---|---|
| Native | All tokens |
| Rejected-enriched | Remove 16 blocks least represented in the selector’s top 25% |
| Retained-enriched | Remove 16 blocks most represented in that top 25% |
| Random | Remove 16 uniformly sampled blocks; one fixed seed |
| Rejected A/B | Remove separate eight-block halves; compare their exact union |

Each block is 4×4 tokens: main removals delete 256/4,096 tokens per older frame.
Prompt, latest frame and object pointers stay full. Original spatial/temporal
coordinates are preserved through the existing sparse K/V retrieval. Random
controls match count, block size and frame age, not global arrangement.
Replays cannot encode or write memory. Native stored-mask lineage stays fixed.

The current frame supplies queries; native memory contains the prompt and up to
six recent **past processed manifest frames**. These need not be consecutive
source-video frames.

## Evaluation and decision

- PolypGen: 21 sequences, excluding seq1/seq7; 114 targets frozen before inference
  (four present targets per sequence and up to two empty targets).
- Primary: present Dice within sequence, then equal mean across sequences.
  Boundary F and sampled-empty false alarms are secondary.
- Utility `U(G) = Dice_native - Dice_without_G`: positive means removing the group
  hurts this prediction. Zero does not establish dispensability.
- Compare rejected removal against retained and random removal using sequence
  bootstrap intervals, Holm-adjusted paired tests and majority sequence wins.
- Engineering passed; the complete ranking criterion failed. Stop the current
  selector. No temporal pulse or full-propagation efficacy run follows.

All results are exploratory: this cohort has already informed development.
Keep the distinction between group deletion effects, intrinsic vector content
and a deployable predictor of harmful tokens.

[Frozen Run171 protocol](../protocols/token_contribution_v1.json) ·
[Verified results and analysis](REGION_TOKENS_RESULTS.md) ·
[Prior-experiment audit](../protocols/PRIOR_EXPERIMENTS.md)

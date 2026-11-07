# Current method: regional memory-token selection

**Hypothesis:** regional context can improve which memory tokens SAM2 reads,
raising segmentation Dice while keeping native memory-frame selection fixed.

Run170 is a training-free experiment with all pretrained SAM2 weights frozen.
The prompt and latest memory frames remain full. Each older native-selected frame
retains exactly 1,024 of its 4,096 spatial tokens in each of four attention layers.
Object pointers and original spatial/temporal coordinates are preserved. Selection
only affects reading; the stored bank remains intact.

For each layer, the first read of a stored frame measures its tokens' attention
importance. Prompt-derived foreground and background feature anchors provide
appearance references. The stored frame's own predicted mask supplies soft
foreground occupancy `p`. With maximum anchor similarities `sf` and `sb`, the
regional affinity is `p*sf + (1-p)*sb`. This is a new project combination to test;
it does not establish semantic purity or known token harmfulness.

The implementation reuses pinned author code:

- [Efficient-SAM2](https://arxiv.org/abs/2602.08224v2): attention computation,
  fixed top-k selection, sparse K/V gather and original-coordinate RoPE.
- [Lean-SAM2](https://arxiv.org/abs/2607.19811v1): diverse prompt anchors,
  maximum cosine matching and multiplicative attention/anchor score fusion.

Both papers' complete systems include other components. This experiment isolates
their memory-selection components on native SAM2 and explicitly labels its new
regional combination as an adaptation.

| Condition | Selection score |
|---|---|
| Native | All tokens |
| Attention only | First-read attention importance |
| Foreground anchors | Author fusion of importance and `sf` |
| Role-blind anchors | Same anchors and fusion, using `max(sf,sb)` |
| Regional anchors | Same anchors and fusion, using `p*sf+(1-p)*sb` |

All sparse conditions have the same frame IDs and retention budget. There is no
additional confidence threshold, frame-quality decision or residual correction.
Fixed top-k is a discrete token-selection rule with a prespecified budget.

First, seq2/seq3 replay must establish native bit parity, exact budgets, unchanged
bank/cache, pointers and four-layer coverage. The fixed-native diagnostic must
then show a positive mean Dice change over native and no loss against the two
anchor controls. Only then run full propagation on the 21 eligible PolypGen
sequences, excluding seq1/seq7. Results remain exploratory because this cohort
has already informed development.

The full screen requires at least +0.020 Dice versus native and +0.010 versus
each anchor control, positive sequence-bootstrap lower bounds, majority sequence
wins and bounded false-alarm increases. Failure stops this candidate.

**Current status:** engineering passed; the seq2/seq3 diagnostic failed its
continuation criterion. This configuration is stopped; full propagation was not
launched. See the measured results below.

[Frozen protocol](../protocols/regional_token_selection_v1.json) ·
[Current results](REGION_TOKENS_RESULTS.md)

Before changing the model or launching another experiment, review the
[prior-experiment audit](../protocols/PRIOR_EXPERIMENTS.md). Identify the closest
predecessor, the concrete mechanism that changes and the observed failure it
addresses. Previous pruning, regional weighting, matching, merging and residual
reader experiments must inform that decision.

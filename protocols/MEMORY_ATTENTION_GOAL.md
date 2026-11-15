> **Superseded goal/work plan:** Follow [ACTIVE_RESEARCH_GOAL.md](ACTIVE_RESEARCH_GOAL.md), updated and resumed on 2026-11-11. The material below preserves the previous scope and decisions.

> **Active amendment — 2026-11-11:** The user authorised testing feature-based
> grouping, gradient-based attribution and alternative matching before further
> ToMe efficacy work. Follow [the staged identification study](TOME_REGION_IDENTIFICATION_STUDY.md).
> Its aim is a more reliable gain than Run155. The historical stop decisions
> below remain valid; new diagnostics do not reopen those failed runs.

# Research goal: region-constrained ToMe in SAM2 memory

## Objective

Test whether training-free, region-constrained merging of SAM2 spatial-memory
tokens can produce a meaningful, reproducible increase in present-frame Dice
on PolypGen. Compare native SAM2, generic local ToMe, and region-constrained
local ToMe at matched per-frame token budgets. The hypothesis is not established.

## Frozen method

- Reuse pinned official ToMe bipartite matching and weighted merging.
- Within each stored frame, merge locally inside 4×4 windows and predicted-mask
  region groups: foreground interior, mixed boundary, and background.
- Never merge across memory frames. Keep SAM2's frame schedule, memory writes,
  decoder, pretrained weights, and object pointers unchanged.
- Match generic local ToMe to the regional arm's exact merge counts per target,
  layer, frame, and window. This isolates the region constraint from the token
  budget and frame choice.
- Do not choose good/bad frames or tune a confidence gate.

The current position-consistent version merges aligned memory and position
inputs before native key/value projection, then applies spatial RoPE at each
cluster's representative coordinate. It preserves the native temporal slot
per frame, but averages within-frame features and therefore approximates native
attention; it is not lossless token preservation.

## Evidence and current decision — 2026-11-10

| Stage | Evidence | Decision |
|---|---|---|
| Run155 PolypGen, 21 sequences | Mean Dice: 0.7347 regional, 0.7125 generic, 0.7102 native. Paired intervals include zero; Holm *p*=0.8813 vs native and 0.5579 vs generic. Excluding seq8, regional gain vs native is +0.0020. | Exploratory signal; not a reliable gain. |
| Run164 DAVIS 2017 validation | Equal-video J&F: 0.9086 regional, 0.9087 generic, 0.9133 native; regional vs native Δ=−0.0047, 95% CI [−0.0124, −0.0003]. | Exploratory transfer result below native. |
| Position-consistent candidate, PolypGen | Real seq2/seq3 engineering, replay, matched-budget, pointer, four-layer, and no-merge parity gates passed. The persistent-ID gate passed on DAVIS bike-packing frames 0–2 for IDs 1 and 2; this is an engineering fixture, not full multi-object accuracy evaluation. Frozen efficacy run stopped after 19/21 matched sequences: interim Dice +0.0302 vs native, but +0.0063 without seq8; false alarms were 193/307 vs 192/307 native. The 14/21-win criterion became unreachable and the false-alarm condition had failed. | Stop under the frozen rule. No seq22/23 matched results, no final full-cohort CI or Holm test, and no claim of a meaningful gain. |

The position-consistent temporal regression test passes, and all 9 focused
ToMe tests pass. A real seq2 gate record shows conditioning frame 3 at `t_pos=0`
and frames 56–61 at slots 1–6, merged independently. This verifies temporal
slot alignment, not retention of every within-frame token's evidence.

Run155's exact main method module was recovered and hash-matched to its setup;
a source replay matched all 2,012 regional targets. A later archive scan found
12 identical snapshots of `prediction_conditioned_affinity.py`, supporting a
stable pre-run helper version. The setup did not hash the imported role-
assignment helper or complete runtime dependency closure, so these snapshots
cannot prove the original process's imports. Exact runtime helper provenance
and per-frame region-map reconstruction remain unverified. See the source audit
under Drive `outputs/experiment/155_region_constrained_tome_polypgen_v1/`.

## Frozen decision rule

Meaningful improvement requires all predeclared conditions: at least +0.01
sequence-mean Dice versus native, a positive mean difference versus matched
generic ToMe, at least 14/21 sequence wins versus native, positive 95% lower
bounds and Holm-adjusted *p*<0.05 for both planned contrasts, and no increase in
pooled false alarms on ground-truth-empty frames. The current position-consistent
candidate failed this screen and is stopped. Do not tune or rerun it on the same
PolypGen or DAVIS validation cohorts to seek a positive result.

## Next action

No ToMe efficacy process is active. The present candidate does not qualify for
held-out confirmation. Keep the results exploratory. Any new ToMe candidate
requires a distinct, evidence-backed mechanism and a newly frozen protocol on
untouched data before inference; otherwise, do not launch another efficacy run.
Do not claim publishability without robust independent confirmation.

Dataset audit: Run155 used all 21 prompt-eligible PolypGen videos; DAVIS 2017
validation has already been evaluated. CholecSeg8k is present locally, but prior
SAM2-family Kalman/EMA memory studies and region-token protocols used or planned
it, and its masks are semantic classes rather than persistent instances. It is
not independent confirmation. No suitable untouched video dataset is currently
staged for a new confirmatory run.

Detailed results and frame-level analyses: [`docs/REGION_TOKENS_RESULTS.md`](../docs/REGION_TOKENS_RESULTS.md).
Method summary: [`docs/REGION_TOKENS.md`](../docs/REGION_TOKENS.md).
Current position-consistent protocol: [`protocols/position_consistent_region_tome_polypgen_v2.json`](position_consistent_region_tome_polypgen_v2.json).

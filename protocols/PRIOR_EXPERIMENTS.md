# Internal experiment audit — 2026-11-07

## 2026-11-08 addendum: MemSAM-derived soft-mask writer

This audit was recorded before the frozen training/validation run in
`memsam_residual_memory_writer_v2`. The Run170 audit below remains unchanged;
this addendum records the later candidate review.

- **Closest completed evidence:** Run174's iterative regional reader was
  negative on its exploratory seq16–23 evaluation. The fixed-history deletion
  diagnostic had mean ΔDice +0.000455 with a 95% sequence-bootstrap interval
  [−0.000071, +0.001384]. Neither tested this memory-write insertion point.
- **Direct literature overlap:** MemSAM (CVPR 2024) uses a predicted-mask-
  conditioned Memory Reinforcement module before memory update: it downsamples
  the probability mask, concatenates it with image features, applies Conv3×3
  and Conv1×1, then gates a feature by the mask and inserts it into working
  memory. Its official code is pinned in the frozen protocol. Therefore
  mask-conditioned memory writing is established prior art; the present
  candidate is an adaptation to SAM2's native memory-encoder input, not a claim
  to invent that broad idea.
- **Related but distinct:** RDE-VOS (CVPR 2022) trains a mask-consistency
  self-correction objective so embeddings encoded from perturbed masks approach
  embeddings encoded from ground-truth masks. The current writer does not use
  this objective or replace SAM2's memory bank.
- **Native overlap and risk:** SAM2 already converts its predicted mask into a
  probability map and supplies the transformed mask to its MemoryEncoder.
  The added soft-mask cue may therefore be redundant or may amplify mask errors.
- **Specific testable difference:** the candidate adds a zero-initialized
  Conv3×3/Conv1×1 residual conditioned on the predicted soft mask to the feature
  entering the unchanged SAM2 MemoryEncoder. The matched writer has identical
  capacity and training but receives a constant 0.5 cue. All original SAM2
  weights, the native memory encoder, reads, pointers, decoder, frame schedule
  and token counts remain fixed.
- **Gate evidence:** attempt04 passed exact zero-writer parity on seq2/seq3,
  four-layer attention checks, finite real-clip writer gradients, exact
  save/resume, and a synthetic two-ID isolation fixture. This does not establish
  natural multi-object efficacy or a Dice gain. Receipt:
  `memsam_residual_writer_gate_results_2026-11-08.json`.
- **Decision and limits:** proceed only with the already-frozen v2
  train/validation schedule and its native plus capacity-matched control.
  PolypGen sequences informed earlier development, so the result is
  exploratory; it cannot establish external or patient-independent
  generalization. The frozen validation release rule must pass before seq16–23
  test inference. Do not change the method based on validation or test
  outcomes.

Prepared **after Run170**, from saved protocols, implementation code and Drive
results. This records the predecessors that must be reviewed before another
model change or experiment. It does not retrospectively preregister earlier work.

The broad hypothesis—regional token use may improve segmentation—has been tested
through several mechanisms. A new repository, seed, budget or region label alone
does not establish a new mechanism or justify another run.

## Evidence and comparability

Numbers below are matched differences within each recorded experiment. Do not
rank raw scores across checkpoint families or cohorts:

- Runs24–28: historical SAM2.1 Base+ setup, checkpoint hash beginning `a2345aed`.
- Runs109–139 listed below: MedSAM2-based setup, with the stated matched controls;
  key completed CV runs use checkpoint hash beginning `c92743b9`.
- Runs155/158: official SAM2.1 Tiny, 21 eligible sequences; native Dice 0.710224.
- Runs167–169: official Tiny, seq16–23; native Dice 0.633500.
- Run170: official Tiny, **fixed-native replay on seq2/seq3**, native Dice 0.881663.

This is an artifact/implementation audit, not a new inference reproduction of all
historical runs. Retry folders and engineering gates are not separate efficacy
experiments. Incomplete or superseded implementations must not be pooled with
completed corrected runs. Source/result hashes are recorded in the Drive audit.

## What has already been tested

| Run(s) | Mechanism already implemented | Recorded scope and outcome |
|---|---|---|
| 17 | Remove tokens by predicted foreground/interior, background or boundary; matched spatial controls | 21-sequence immediate exclusions. No region passed the all-definition robustness rule. Background-control geometry is a limitation; attention/exclusion results do not establish a deployable keep/drop policy. |
| 24 | Keep polyp, boundary and nearby context; discard far background and use the saved token budget for longer history | 21 sequences: regional-minus-native **+0.019348**, CI [−0.023085, +0.072815]. History length also changes, so this is not an isolated token-selection result. |
| 25 | Apply the same sparse longer-history policy to SAMURAI and SAM2Long | 21 sequences per author: **−0.013944** versus SAMURAI and **−0.006129** versus SAM2Long. |
| 26 | Keep native frame schedule; drop older boundary tokens while preserving interior/background | 21-sequence propagation: **−0.026513** versus native. Seq13 contributes a large loss. |
| 28–31 | Fixed-native replay, single pruning pulse and native-writeback replacement | Run28 immediate regional-minus-native **+0.000097**, Holm p=1. Run31's whole-frame writeback replacement restores subsequent native equality in the selected seq13 case. These expose feedback sensitivity, without isolating one causal tensor or proving a deployable remedy. |
| 74/81 | Change regional attention mass, including query-conditioned regional weighting | Historical eight-sequence pilots: no meaningful Dice gain. These already test suppressing/amplifying regions inside attention. |
| 85/87/89 | Scale values, then scale both keys and values by region | Historical eight-sequence pilots: changes near zero. K/V modulation is already implemented; it must not be proposed as an untested direction. |
| 109 | CFBI nearest-neighbour regional matching as a feature correction | Completed 21-sequence CV: **+0.002520** versus native; meaningful-gain screen failed. |
| 117 | Regional Q/K adaptation versus matched ordinary Q/K adaptation | Completed 21-sequence CV: **+0.007586** versus native, only **+0.000465** versus Q/K-only. It changes original Q/K weights and therefore differs from the current frozen-SAM2 constraint. |
| 119 | Mask-guided SimVOS TokenLearner regional prototypes | Completed 21-sequence CV: **+0.008575** versus native, **−0.000961** versus global prototypes; screens failed and false alarms increased. |
| 120 | RegionContrast supervision of Q/K representations | Completed 21-sequence CV: **+0.007847** versus native, **−0.000792** versus Q/K-only. Retains dense memory and updates Q/K; it is not token removal. |
| 123 | Learn soft query roles and bias query-to-memory role compatibility | Corrected/recovery 21-sequence CV: **+0.002991** versus native, **−0.000011** versus global control; screen failed. |
| 127 | Prediction-conditioned soft regional affinity during a second dense memory read | Completed corrected 21-sequence CV: **+0.002970** versus native, **+0.002801** versus globalized roles; uncertainty includes zero and screen failed. Do not substitute superseded gradient-bug attempts for this record. |
| 139 | Region-conditioned low-rank transforms trained with native histories | Completed 21-sequence CV: **+0.003556** versus native, **+0.000204** versus global control; meaningful-gain screen failed. |
| 150/163 | Learned regional memory prompts, followed by case diagnosis | Intentionally stopped after 14 regional validation sequences because the frozen win requirement became unreachable. No completed full-CV mean. Replay on seq8/seq14 associated failures with collapsed prediction-derived region maps. Auxiliary prompt-norm statistics with incorrect grouping are invalid. |
| 155 | Merge similar tokens only within the same predicted region/local window | 21-sequence propagation: **+0.024483** versus native, CI [−0.001366, +0.071965], wins 10/21. Removing seq8 from the arithmetic leaves **+0.002031**. Screen failed; source hash has since been recovered, but the efficacy result remains fragile. |
| 158 | Use continuous foreground/background occupancy together with feature similarity for ToMe matching | 21-sequence propagation: **−0.003773** versus native, **−0.003622** versus generic merging; screen failed. Soft occupancy plus feature similarity is already tested at this insertion point. |
| 167/168 | Cutie-derived object/context summaries and an additive readout correction; then fixed-history replay | Eight-sequence propagation: **+0.005054** versus native. Fixed-native replay: **+0.000898**. Neither establishes meaningful improvement. |
| 169 | Add foreground/context contrastive supervision to that same reader | Eight-sequence propagation: **+0.003964** versus native, **−0.001090** versus the same reader without contrastive loss. One chronological training pass; convergence was not established. |
| 170 | First-read attention/top-k with fixed prompt foreground/context anchors and soft role-conditioned affinity | 73 fixed-native targets: **−0.003618** versus native, **−0.001026** versus foreground anchors, **+0.000063** versus the same anchors ignoring roles. Engineering passed; full propagation was not launched. No GT-empty safety denominator. |

## Run170's actual distinction

The reviewed earlier project protocols do not show the exact combination of
fixed prompt foreground/context anchors, causal first-recollection top-k masks,
and `p*sf+(1-p)*sb` affinity. This is a limited search finding, not a novelty claim.

Its components overlap strongly with earlier work: region-conditioned use of
memory, predicted-mask roles, feature matching and selective spatial retrieval.
It is an incremental variation within that family. Earlier completed pruning
and compression experiments remain relevant evidence when judging its motivation.

Likewise, a directory or table row labelled Efficient-SAM2 SMR with unverified/NV
status is not a completed negative baseline. Run170 verifies an adaptation of
author selection components; it does not reproduce the complete Efficient-SAM2 or
Lean-SAM2 system.

Run170 used the same two sequences for engineering verification and its scientific
continuation screen. All 73 scored targets were GT-present. Its negative screen
is limited to that subset; it does not settle false-alarm effects or performance
across all 21 sequences. A future justified mechanism study should distinguish
parity verification from a diagnostic with prespecified, representative failure
cases. This is a design lesson for future work, not a reason to rewrite Run170's
completed stopping rule or extend it after seeing its result.

## Run155 source provenance follow-up

Run155's full-run setup records method-module SHA-256
`780426d00e9b6acad4ca4f90cdd6e5c4c0d10b19473aac05edc1c37daab05ffc`.
The exact 15,199-byte source was recovered from unreachable Git blob
`4c3c75b2b3d518b8a49c343361a189e7b28dc6f3` and its SHA-256 matches the setup
receipt. A preserved copy and recovery manifest are under Drive
`outputs/experiment/155_region_constrained_tome_polypgen_v1/source_audit_20261109/recovered_source/`.
This resolves the source-version gap; it does not change the Run155 result:
its confidence interval includes zero, the adjusted tests are not significant,
and its mean gain is strongly sensitive to seq8. Run158's current soft-mask
module matches its recorded hash. Recorded result receipt hashes checked for
Runs127/169/170 match their result files.

## Required review before the next implementation or launch

1. Identify the closest previous run IDs, their saved protocols, code versions,
   outcomes and limitations. Recover archived source when current code differs.
2. State the concrete mechanism that changes: inputs, representation, selection
   operation or insertion point. Explain which observed failure it addresses.
3. Use existing saved cases to justify that change. Distinguish token relevance,
   token harmfulness, spatial region labels and errors propagated through writes.
4. Explain why the result is not already answered by the matched controls. A
   region claim needs evidence beyond generic adaptation or extra anchor coverage.
5. Verify compatible checkpoint, prompt, frame schedule, cohort and metric before
   comparing scores. Separate engineering checks, fixed-history effects and full
   propagation. Tiny diagnostic subsets cannot settle full-cohort performance.
6. Freeze one justified protocol before inference, with integrity checks, a
   diagnostic, matched controls and a stopping rule. Seeds, budget changes or
   repository swaps alone do not justify restarting a stopped method.

Current action: review the evidence and source gap. This audit authorizes no new
inference and changes no completed scientific protocol or model implementation.

Drive audit: `outputs/experiment/170_regional_token_selection_v1/prior_experiment_audit/`.

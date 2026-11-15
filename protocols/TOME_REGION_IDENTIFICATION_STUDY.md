# ToMe region-identification study — 2026-11-11

## Objective

Develop and test token-level memory utilisation in frozen SAM2, aiming for a
more reliable PolypGen Dice gain than Run155. Test the hypothesis; do not require
positive results. Run155's 0.7347 is an exploratory reference, not a universal
baseline or proof of significance. Keep all historical failed screens intact.

## Ordered work

1. Audit previous implementations and pin upstream sources. Establish unchanged
   ToMe regression tests before implementing new grouping. Never mix Run155's
   post-RoPE merger with the later position-consistent merger in one comparison.
2. Diagnose region identification on fixed native trajectories, initially seq2
   and seq3. Sample frames by a fixed schedule, not observed improvement. Capture
   spatial memory features before position addition, predicted probabilities,
   source frame/object IDs, and gradients of a documented decoder target.
   Ground truth is evaluation-only after forward, never a grouping input.
3. Compare predicted-mask roles with feature-graph partitions (official TokenCut)
   and gradient attribution (pytorch-grad-cam). Do not equate low activation with
   background. GETAM is a literature reference, not a reproduced implementation.
   Report foreground/background overlap, object coverage, ambiguity, temporal
   stability, gradient-target sensitivity, runtime, and feature-layer choice.
   Evaluate prompt-conditioned orientation separately from the raw partition;
   never choose partition sign by agreement with evaluation ground truth.
4. Only after region diagnostics, test merge matching independently: pinned
   ToMe bipartite matching versus an upstream clustering implementation after
   source review. kNN alone is a neighbourhood graph, not a complete merger.
   Object pointers are an optional diagnostic, not automatic cluster centres:
   their embedding space must be validated against spatial memory features.
5. Gate each selected integration on no-merge parity, aligned K/V assignments,
   mass, positions, temporal frame isolation, unchanged pointers and pretrained
   weights, all four attention layers, persistent IDs and fresh-process replay.
   Include empty masks, tiny objects, all-background and degenerate features.
6. Freeze a separate exploratory efficacy protocol before inference. Compare
   native, Run155-compatible regional ToMe, generic ToMe and at most one selected
   candidate with equal local token budgets. Use complete matched sequences
   seq2–23 except seq7 (21 total; seq1 excluded). Reuse archived outputs only
   after checkpoint, prompt, frame list, preprocessing and metric compatibility
   checks. No training/evaluation selection on test outcomes.
7. Report paired equal-sequence Dice, Boundary F, false alarms, reappearance,
   runtime and memory, confidence intervals, sequence wins and leave-seq8-out
   sensitivity. Require improvement over both native and Run155 under the same
   setup; a larger raw mean alone is insufficient. Freeze numerical practical
   improvement and stopping criteria before step 6, not after seeing results.
8. Confirm a surviving candidate on untouched annotated videos under a frozen
   protocol. Existing PolypGen and DAVIS validation remain exploratory.

## Design constraints

No good/bad frame selection, confidence gating, pretrained weight changes or
unnecessary reruns. Native dense current-frame query grid stays intact. Test
membership and token usefulness separately: useful background is possible.
TokenCut's native graph threshold and eigenvector partition are explicitly
part of the diagnostic; they are not a new token-pruning/confidence policy.
No max-pooling or cross-frame averaging is silently introduced.

## Sources and current implementation

- ToMe: https://github.com/facebookresearch/ToMe — existing pinned implementation.
- TokenCut: https://github.com/YangtaoWANG95/TokenCut — revision
  `fed52cd5b60891baefd8ec7110dafa73be816ee1`, MIT; `object_discovery.ncut` reused.
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam — source review pending.
- GETAM: https://arxiv.org/abs/2112.02841 — gradient-based label-refinement reference.
- CFBI: https://arxiv.org/abs/2003.08333 — target/background matching reference;
  inspect existing CFBI work before proposing a duplicate.

`modeling/token_partition_diagnostic.py` is offline-only. It supplies the dummy
CLS entry discarded by upstream TokenCut, preserving all spatial tokens. It
returns a component and eigenvector, not verified foreground/background labels.
Porting the partition to SAM2 features does not reproduce TokenCut's published
DINO results. Full-resolution cost and feature-space suitability remain untested.

## Tracking

Store receipts, extracted arrays and plots under the Drive-mounted directory
`outputs/experiment/tome_region_identification_v1/`. Record source hashes,
commands, completion/errors, frame/object coverage and timestamps. Do not treat
this mount as proof of cloud synchronization. No full efficacy process may be
reported as running before its PID and advancing output are checked.

## Implemented diagnostic runner — 2026-11-11

- `scripts/run_token_identification_diagnostic.py`: frozen native replay,
  upstream TokenCut, upstream Grad-CAM channel aggregation, raw gradient norms,
  prompt-prototype partition orientation and evaluation-only source GT.
- `scripts/queue_token_identification_diagnostics.py`: sequential bounded gate,
  exact comparison with the prior verified artifacts, then wider diagnostics.
- `scripts/summarize_token_identification_diagnostic.py`: verified artifact
  reporting; prompt-source reads excluded from headline metrics.
- 19 focused tests pass; four real-frame decoder replays pass; six source–target
  reads (72 arrays) repeat bit-identically in a fresh process.
- Upstream Grad-CAM pin: `9194156183c097b8bb2bd8f945a59c644c2733d9`.
  Additional dependency: `protocols/requirements_token_diagnostics.txt`.
- `external/sam2_diagnostic_reference` is a clean worktree at SAM2 revision
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`. The user's comments in the original
  checkout remain untouched. The native model is neither trained nor patched.

Run with `.venv/bin/python`, preserving that path rather than resolving its
symlink to the base interpreter. Pass a new Drive-mounted output directory:

```sh
.venv/bin/python scripts/queue_token_identification_diagnostics.py --output-dir '<Drive experiment>/new_series' --reference '<Drive experiment>/real_frame_attempt03'
```

The amended wider protocol is `token_identification_diagnostic_wider_v2.json`.
It resolves frame availability before inference and records unreachable offsets.
Version 1 failed its final coverage check because seq3 has only 15 manifest
frames. The 42 completed source–target reads were revalidated offline under
the explicit availability amendment; the original failure is preserved.
This is not a full efficacy runner or a validated multi-object method.
Pointer-based grouping, temporal stability analysis and full-video Dice
comparison remain conditional follow-up work. The subsequent ATC matching
diagnostic is recorded below.

## Subsequent matching diagnostic — completed

The earlier shortened-video six-target figures are superseded for canonical
native comparisons. Exact canonical-timing replay, persistent-ID isolation and
native-cache gates passed. The broader diagnostic completed 318 fixed targets
across 21 sequences; regional ToMe Dice 0.647851 versus native 0.643445,
regional ATC 0.645320 and generic ATC 0.644563. Both new ATC candidates failed
the frozen continuation screen. No full-propagation study was launched.
All methods overlapped GT in 37/51 fixed return windows. See
[results](../docs/REGION_TOKENS_RESULTS.md) for the full table and limitations.

A post-run audit validated 1,590 condition records and 5,088 attention calls;
40 focused tests passed after stricter analysis validation. Original results
are unchanged. The next source-reviewed engineering candidate is PiToMe,
not another ATC sweep. Its protection stage requires a lower removal budget
than the half-group budget used above. No efficacy claim follows from its
source-only probe. Drive review: `candidate_source_review01/review.md`.

## Completed PiToMe screen — 2026-11-11

All 318 targets across 21 sequences completed. Native archive replay was exact; parameters and banks remained unchanged. Plan/results/summary/frozen-source hashes were checked. The post-completion arithmetic audit and supplemental strata/region-scope reports are saved on Drive. Neither protected-token candidate passes the frozen screen, so stop both fixed PiToMe variants; no full propagation and no margin/budget sweep.

| Method | Dice | Δ native |
|---|---:|---:|
| native | 0.643445 | +0.000000 |
| tome_regional | 0.643731 | +0.000286 |
| tome_global | 0.645229 | +0.001784 |
| pitome_regional | 0.643426 | -0.000019 |
| pitome_global | 0.643049 | -0.000396 |

False alarms 36/66 and return-window overlap 37/51 in every condition. Regional PiToMe versus native: CI [−0.000354,+0.000263], Holm p=1.0, wins 8/21. These are fixed-history immediate effects, not full-video performance.

Next action: finish evidence synthesis across the completed identification and matching diagnostics before proposing another implementation. In particular, distinguish predicted-mask role identification failures, matching/aggregation effects, and propagation dynamics. Do not launch another heuristic or a full-video run without a new mechanism justified by that synthesis and a frozen comparison. The broader improvement objective remains unmet.

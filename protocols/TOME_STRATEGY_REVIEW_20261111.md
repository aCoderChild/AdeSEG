# Token utilisation: source review and controlled tests

| Reference | Relevant mechanism | Relationship to this study |
|---|---|---|
| [ToMe, ICLR 2023](https://arxiv.org/abs/2210.09461) / [code](https://github.com/facebookresearch/ToMe) | Bipartite similarity matching, weighted aggregation and source tracking | Existing baseline; classification-oriented original work does not prove SAM2 Dice improvement. |
| [ATC, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7441_ECCV_2024_paper.php) / [code](https://github.com/JoakimHaurum/ATC) | Bottom-up hierarchical token clustering; includes detection/segmentation experiments | Selected alternative matching backend. Reuse author average-linkage function and weighted merge, at a fixed budget. SAM2 memory adaptation is new experimental glue, not a reproduction of their benchmark scores. |
| [ALGM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Norouzi_ALGM_Adaptive_Local-then-Global_Token_Merging_for_Efficient_Semantic_Segmentation_with_CVPR_2024_paper.html) / [code](https://github.com/tue-mps/algm-segmenter) | Local then global adaptive merging in plain-ViT semantic segmentation | Source reviewed. Local implementation has a CUDA-specific allocation and adaptive thresholding. Not integrated in this fixed-budget study; a direct CPU/SAM2 transplant is inappropriate. |
| [TokenCut](https://arxiv.org/abs/2209.00383) / [code](https://github.com/YangtaoWANG95/TokenCut) | Feature graph partitioning for object discovery/segmentation | Implemented upstream partition diagnostic; its foreground identity must be anchored to the prompted target. We do not use target GT to choose partition sign. |
| [GETAM](https://arxiv.org/abs/2112.02841) | Gradient-weighted Transformer attention plus refinement for segmentation pseudo-labels | Literature reference for attribution; current code uses upstream Grad-CAM, not a claimed GETAM reproduction. Low gradient is not automatically background. |
| [StructSAM, 2026 preprint](https://arxiv.org/abs/2603.07307) | Spatial finite-difference/Sobel feature energy, protected structures and encoder merge–unmerge | Distinct from backpropagated Grad-CAM. Relevant to structure preservation, but operates in the image encoder and reports an efficiency/accuracy tradeoff. No verified official implementation was identified in this review; not implemented from scratch or claimed reproduced. |
| [CFBI, ECCV 2020](https://arxiv.org/abs/2003.08333) | Foreground/background feature matching in VOS | Already adapted in project Run109; its full-cohort improvement did not meet the meaningful-gain screen. Avoid proposing the same feature correction as new. |

## Archive checks

Run158 already tested continuous occupancy plus feature similarity during ToMe
matching and failed its screen. Run155 and the later position-consistent method
also failed their reliability criteria. The new comparison varies matching
algorithm and regional constraint separately, rather than repeating soft-role
weighting or changing frame selection.

## Implemented four-arm diagnostic

1. ToMe + native predicted-region constraints.
2. ToMe ignoring region identity, with the same local token counts.
3. ATC + those same region constraints.
4. ATC ignoring region identity, with the same local token counts.

Insertion is the recovered Run155-style post-projection/post-RoPE K/V read.
Queries remain dense. The same native memory history is used in every arm;
interventions are not written back. Native decoder presence handling remains.
This tests immediate segmentation effects; it does not establish full-video
propagation benefit. Canonical protocol: `author_matching_diagnostic_v2.json`; broader completed study: `author_matching_causal_cohort_v2.json`.

ATC source pin: `227c94b584df323b94dd79e8ddbb88b1587eeb8e`, file
`SSLBackbone_setup/token_reduc/merge.py`. Upstream source is unchanged. Use the
author sum/mass weighted aggregation, avoiding raw `mean` on zero-initialized
destinations. Use source maps to validate cluster membership instead of relying
on an unmerge closure that assumes values have the metric's channel width.
Tests cover matched budgets, K/V alignment, pointer preservation and conserved
mass with different metric/value widths. These checks apply to this adapter,
not every usage of the complete upstream repository.

## Verified experiment defects

- Wider identification v1 requested seq3 offsets 16 and 32 even though its
  manifest has 15 frames. Coverage planning now runs before inference. An
  explicit v2 amendment records unavailable offsets; old failure evidence is
  preserved. The 42 completed reads were revalidated without new inference.
- First matching launch resized the 1024 mask to video resolution. The native
  predictor exports the selected 256 mask directly to video resolution. Native
  parity caught this before any treatment result was saved. Corrected replay
  uses the author's selected low-resolution output and unchanged native resize.

Outputs and receipts are on Drive under `tome_region_identification_v1/`.
No paper cited here guarantees a Dice gain for this SAM2 memory adaptation.

## Scope of ATC linkage conclusions

The current adaptation freezes **average linkage**, reusing the official
SSL-backbone helper. The official ATC README also documents single linkage for
its ViT-Adapter detection/segmentation training example (README lines57–63 at
the pinned revision), and complete linkage for DeiT. Consequently this study
cannot establish a family-wide negative result for ATC or reproduce its
trained dense-prediction architecture. No linkage change is permitted during
the current frozen experiment; any further test needs a distinct, documented
rationale rather than changing linkage to obtain a positive cohort result.

## After the completed screen

Both ATC average-linkage candidates failed the frozen broader screen. Stop
these variants. A new source review considers [PiToMe](https://arxiv.org/abs/2405.16148)
and its [official implementation](https://github.com/hchautran/PiToMe), pinned
at `550b5deed94aadfeac28bfbe381d87e672044a40` on Drive. Its explicitly protected
set has N−2r tokens, so it is empty at 50% removal. A lower-budget, matched-control
engineering study is required to actually test protection. No SAM2 inference or
positive segmentation result exists for this candidate. Review and source probe:
Drive `tome_region_identification_v1/candidate_source_review01/`.

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

## Task-aware source applicability review

Official ECM, ToMA and HAC-ToMe repositories are pinned and reviewed on Drive at `task_aware_source_review01/`. ECM author-only synthetic probe confirms original retained keys, in-place value updates and a default-target-count limitation for small groups; no SAM2 transplant was run. HAC self-attention correction and learned correction are not direct fits for the current training-free dense-query cross-attention setup. ToMA diffusion merge/unmerge is also not a drop-in adaptation. No candidate promoted. Next freeze a bounded existing-merge distortion diagnostic (K/V norms and attention outputs), keeping fidelity measurements distinct from segmentation improvement. See the linked paper/code review and receipts in that Drive folder.

## Candidate considered: longer history under a fixed token budget

Representing more timestamps with fewer spatial tokens per frame is a coherent
temporal-coverage hypothesis. In the default fully populated one-prompt read,
SAM2 contributes one conditioning map plus six previous 64×64 maps: 28,672
spatial tokens, with object pointers separate. For example, one full conditioning
map plus twelve half-density past maps has the same spatial-token count. This
would change the native frame schedule and require an explicit way to encode
ages beyond the seven pretrained temporal slots. Reusing those slots without a
validated rule could make distinct frame ages ambiguous.

Changing token counts across only the same native frames cannot redistribute
the full native budget: each map already contributes at most its original
4,096 cells. A fixed-budget longer-history method therefore needs additional
timestamps, another feature source, or a smaller total budget. Uniform temporal
coverage could avoid content-based good/bad frame choice, but it still changes
which frames enter attention. Keep this candidate outside the frozen
same-schedule ToMe diagnostic. Revisit only after its attention-distortion
results and an explicit temporal-position design; no implementation or
inference has been authorized by this note.

## Trajectory audit addendum — 2026-11-12

The completed wider-history run is now available and its saved seq6/seq8
trajectories have been audited. In both sequences, native, generic ToMe and
regional ToMe prediction hashes match on all seven present targets with at most
six available history maps. After compression begins, seq6 degrades strongly
with ToMe while seq8 recovers more returns and incurs more empty-frame false
alarms. Generic and regional results are nearly identical. This localizes the
trade-off to the wider-history intervention, but temporal coverage, per-map
spatial density and reuse of the oldest learned age slot are coupled.

Strategy review: ToMe mass-weighted mean is the only full-video aggregation
tested; channel-wise max is untested. ATC and PiToMe failed their frozen
diagnostic screens. TokenCut and Grad-CAM were identification probes only;
Grad-CAM did not reliably label foreground. k-means and pointer-centred
grouping remain untested. The current local ToMe window cannot retain fewer
than 2,048 tokens per map in one pass, so a strongly age-weighted split at a
12-map horizon is infeasible without changing the merge operator. A ten-map
follow-up is now specified as a four-arm 2×2: uniform versus recent-dense
allocation across the same ten maps, crossed with generic versus regional ToMe
matching. All arms keep the total spatial K/V budget, local matching,
weighted-mean aggregation, frame IDs and SAM2 behavior fixed. This isolates
the allocation effect and checks whether region-aware matching interacts with
it. It is exploratory on this already analyzed cohort; implementation and the
engineering gate are pending, and no inference has been run. Do not add
max-pooling, CAM, k-means, pointer-centred or ATC/PiToMe variants to this
comparison. Untouched-data confirmation remains necessary. The frozen design
is in `protocols/ten_map_allocation_tome_polypgen_v1.json`. Full analysis and
source hashes are on Drive:
`wider_history_fixed_budget_v1/full_attempt01/trajectory_strategy_review_2026-11-12/`.

This addendum supersedes the older instruction in this file to inspect seq6/seq8
before choosing a follow-up. No model code or efficacy protocol changed during
the audit.

### Ten-map follow-up status — 2026-11-12

The frozen ten-map allocation protocol and runner passed the real-video
seq2/seq3 engineering gate (73 targets; 292 attention-layer reads). Checks
covered native parity through six history maps, identical source-frame IDs,
matched quotas within each allocation, fixed token/mass budgets, frozen SAM2
parameters, temporal-slot restoration, and source hashes matching the snapshot.
The full 21-sequence exploratory run is active on Drive at
`outputs/experiment/ten_map_allocation_tome_polypgen_v1/full_attempt01/`.
An independent process replay also reproduced all four arms on seq3 exactly:
14/14 targets per arm, 56 total, including prediction/memory hashes, metrics
and attention traces. Receipt:
`outputs/experiment/ten_map_allocation_tome_polypgen_v1/independent_replay_all_arms_attempt01/replay_receipt.json`.
There is no Dice efficacy result yet. The full run was paused for this replay;
the `uniform_regional/seq2` wall-time includes the pause and is excluded from
runtime comparison. Gate receipts are at
`outputs/experiment/ten_map_allocation_tome_polypgen_v1/engineering_gate_attempt01/`.

## Completed ten-map decision — 2026-11-12

The frozen 21-sequence four-arm comparison is complete and failed its screen. Native Dice0.710224; uniform generic/regional0.706793/0.706820; recent-dense generic/regional0.705597/0.705649. All five equal-sequence means were independently recalculated. No regional-specific gain or allocation improvement is established; no more variants of these frozen candidates are authorized by this result.

Next mechanism contrast must compare the same cached native bank under native-six, dense-ten and compressed-ten reads, using existing replay and official ToMe code. Dense-ten is an intentionally over-budget diagnostic control, not an efficiency competitor. Score only after forward; no writes. Compressed-ten minus dense-ten isolates the immediate compression intervention on this bank; dense-ten minus native-six isolates the broader-history read intervention (including age-slot handling). Comparing either with propagated cohort output does not independently identify propagation causality because their banks differ. A paired rollout with a shared starting bank would be required for that claim. Keep the original FP32 identity failure and its numerical forensics separate from this segmentation-output contrast.

## Additional official implementation screening — 2026-11-12

[vid-TLDR (CVPR2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.html) is training-free video-token merging, but its official repository evaluates video-text retrieval and QA, not SAM2 segmentation. Cloned unmodified official code to Drive and pinned revision/source in `task_aware_source_review01/vid_tldr_applicability_probe.json`. Exact author `get_objective_score` on cross-attention [1,1,4,16] returns [1,4,1] query scores, not16 memory-token scores; uniform attention yields nonfinite normalization. It uses mean-based clipping. No direct transplant or efficacy run is justified. Omitting its clipping or changing the scoring axis would be a new adaptation requiring separate justification, not an author reproduction.

[MCTF official code](https://github.com/mlvlab/MCTF) includes a30-epoch fine-tuning recipe; its classification results do not support a promised frozen-SAM2 Dice gain. Existing ECM/ToMA/HAC applicability caveats remain. Prior selection/protection failures must not be renamed and relaunched as new methods. The spatial-history diagnosis motivates preventing damaging read perturbations, but no new repair is yet validated.

### 2026-11-13 source correction and new direct prior

The local Efficient-SAM2 author checkout does contain the SMR mask consumer: `build_model/window_bypass.py::WB_RoPEAttention_ca.forward` filters spatial K/V tokens and indexes the retained RoPE positions from the original grid. The earlier audit missed this because it searched only under `sam2/`; see `protocols/EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md`. The official method is hard attention-derived top-k pruning for efficiency, so it conflicts with the no-gating constraint and has not been adapted or run here.

Puspitasari et al., “Accelerating SAM2 With Efficient Memory Attention Module via Spatiotemporal Token Pruning,” TPAMI 2026, is a direct memory-attention token-pruning reference ([DOI](https://doi.org/10.1109/TPAMI.2026.3723878)). The abstract reports up to 87% token compression and 1.7× speedup with negligible performance disturbance on its benchmarks. This is evidence that memory tokens have unequal utility for efficient VOS, not evidence of a PolypGen Dice improvement or a Seq5 fix. No author repository was located in the source search; no reproduction is claimed.

## DecAF source review — 2026-11-13

Pinned official DecAF revision `bf385b9a7fc73201dd79bd967cc0154a8888c0bc` and inspected paper/code. Its contrastive maps arise from separate language-model prompts and feed new SAM2 point prompts plus thresholded tracklet selection; this is not a native memory-token method and does not fit the frozen prompt/no-gating goal. Six exact-author normalization fixtures passed (no model inference). Do not transplant the full pipeline or claim a SAM2 memory gain. Source hashes, exact function and applicability table: Drive `tome_region_identification_v1/decaf_source_review_20261113/`. Conceptual relevance only; no new efficacy candidate.

# PolypGen token-utilisation results

Updated: 2026-11-13. **No meaningful, reliable improvement is established yet.**

Entries below the latest completed run include historical, time-stamped interim observations; references to a run being active describe the checkpoint when recorded, not its current status.

## Current candidate: cross-frame Video-ToMe value sharing

| Check | Result |
|---|---|
| Upstream | Official Video-ToMe repo, pinned at `267fb4915c0dcb55f87f555fd72dbb6df4a618b7`; unmodified `tome/merge.py` SHA256 `b70f6d48…22af57` |
| Method | Match projected memory keys in each local 4×4 tile across the six selected past maps; average corresponding projected values and unmerge to all original slots |
| Controls | Exact native/no-op; same 25% merge budget performed independently within each past frame |
| Frozen SAM2 components | Weights, Q/K, positions, memory writes, six-map schedule, pointers and mask decoder |
| Efficiency claim | None. Dense token count is restored by unmerge; this tests cross-frame value sharing |
| Full-propagation engineering gate | Two fresh v4 runs passed on seq6 and seq8: 138 target frames, 414 condition-frame outputs each; exact no-op/native parity and byte-identical result SHA256 `b2ab812f…a85802e` |
| Focused verification | 17 tests passed, including real attention-layer, audit and cohort-statistics checks |
| Full-cohort protocol | Frozen as `spatiotemporal_tome_value_exchange_cohort_durable_v2.json`; 21 sequences, 2,012 targets, 6,036 predictions planned |
| Current status | Durable v2 completed all 21 sequences; the predeclared improvement screen failed |

Final result (2026-11-13): all 21 eligible sequences completed, comprising
2,012 post-prompt targets and 6,036 condition-frame outputs. The completion
receipt matches the results SHA256. The audit passed on all 6,036 saved rows and
hash-bound attention receipts. Candidate masks were not independently rescored.

| Condition | Mean present-frame Dice | Boundary F | GT-empty false alarms | Missed returns |
|---|---:|---:|---:|---:|
| Native/no-op | 0.710224 | 0.616994 | 194/323 | 11/51 |
| Within-frame control | 0.711331 | 0.618016 | 194/323 | 11/51 |
| Cross-frame ToMe | 0.710586 | 0.617847 | 194/323 | 11/51 |

Cross-frame − native: +0.000362 Dice (95% sequence bootstrap CI
[+0.000064, +0.000721], Holm p=0.0876; 14 wins, 1 tie, 6 losses).
Cross-frame − control: −0.000745 (95% CI [−0.002554, +0.000235],
Holm p=0.3703). The predeclared meaningful-improvement screen failed: the
native and control gain thresholds were not reached. Seq20 Dice was 0 in all
three conditions. Seq6 and seq8 informed development, and sequence-level
independence is unverified; these results are exploratory. No independent
external-data confirmation is claimed. Audit receipt: Drive
`cohort_durable_v2/attempt01/audits/durable_reproduction_21seq.json`.

The saved frame-wise YOLO-box comparison on the same 2,012 frames gives
mean present-frame Dice 0.718842 and 17/323 false alarms. Cross-frame ToMe
is −0.008256 Dice lower; this is descriptive because historical detector,
checkpoint and prompt-cadence provenance is incomplete. Candidate condition
CPU runtimes were 2,720.6 s native, 4,272.8 s control and 3,197.2 s
cross-frame; YOLO runtime is unavailable. Receipt: Drive
`cohort_durable_v2/attempt01/audits/full_yolo_saved_output_comparison/spatiotemporal_yolo_matched_full.md`.


The earlier `cohort_v1/attempt01` stopped at seq14/local target 144/248 and
lost in-memory raw audits. Its eleven complete triplets below remain descriptive;
they are not outputs of durable v2. Recovery was gated by exact real-frame
interruption/replay checks and a two-ID isolation fixture using synthetic parts
of one real mask. Receipts: Drive `durable_main_real_gate_v1/comparison.json` and
`spatiotemporal_multiobject_gate_v2/attempt01/results.json`. This fixture does
not establish performance on anatomical multi-object videos.

| Interrupted v1 subset (11 sequences; equal-sequence means) | Dice | Boundary F | GT-empty false alarms | Missed return events |
|---|---:|---:|---:|---:|
| Native/no-op | 0.736563 | 0.654737 | 145/218 | 4/27 |
| Within-frame control | 0.736592 | 0.654932 | 145/218 | 4/27 |
| Cross-frame ToMe | 0.736640 | 0.655206 | 145/218 | 4/27 |

Cross-frame ToMe is +0.000078 Dice vs native and +0.000048 vs control. Native/no-op
matched exactly on 1,277/1,277 completed targets. This incomplete subset does not
establish a meaningful gain. Receipt: Drive `interim_checkpoint_11seq_20261113.json`.

On these eleven saved trajectories, native and cross-frame predictions have the
same empty/nonempty status on all 1,277 targets: both are empty on 120/1,059
GT-present targets and nonempty on 145/218 GT-empty targets. This is a frame-level
diagnostic of mask occupancy inferred from saved metrics, not proof that the masks
or individual token contributions are identical. Receipt: Drive
`cohort_v1/attempt01/audits/cross_frame_occupancy_diagnostic_11seq.json`.

Recorded CPU condition runtimes total native 1712.0 s, within-frame
2674.4 s (1.562× native), and cross-frame 2049.4 s
(1.197× native). Execution order was sequential. The earlier
3,105-record integrity audit and the additional 726-record seq13 audit passed.
Receipts: Drive `audits/completed_10seq_integrity_runtime.json` and
`audits/seq13_completed_triplet_audit.json`.

On these same eleven sequences, saved YOLO-box masks average 0.687914 present-frame
Dice; cross-frame ToMe wins 4/11 sequences. Historical YOLO inference provenance
is incomplete, so this is a descriptive saved-output comparison. The full
21-sequence YOLO reference is 0.718842 on the matched present-frame denominator.

The prepared YOLO audit was repaired to retain Dice in temporal rows, distinguish
partial versus full reference counts, and disclose that candidate metrics are
reaggregated from score rows while YOLO masks are independently rescored. Focused
failure reproduction and report checks passed. Receipt: Drive
`audits/yolo_audit_score_row_fix/`. Frozen inference sources were unchanged.
The verified post-completion command and input paths are recorded in Drive
`cohort_durable_v2/postrun_yolo_audit_command.json`; full-mask rescoring awaits cohort completion.

| Sequence | Native/no-op | Within-frame | Cross-frame | Cross−native | Cross−control |
|---|---:|---:|---:|---:|---:|
| seq2 | 0.888730 | 0.888720 | 0.888870 | +0.000139 | +0.000150 |
| seq3 | 0.874597 | 0.874768 | 0.874669 | +0.000073 | −0.000099 |
| seq4 | 0.814211 | 0.813921 | 0.814390 | +0.000179 | +0.000469 |
| seq5 | 0.541987 | 0.541793 | 0.542047 | +0.000061 | +0.000255 |
| seq6 | 0.864721 | 0.864632 | 0.864724 | +0.000003 | +0.000092 |
| seq8 | 0.279446 | 0.279454 | 0.279219 | −0.000227 | −0.000235 |
| seq9 | 0.906740 | 0.907716 | 0.907528 | +0.000788 | −0.000188 |
| seq10 | 0.936679 | 0.936717 | 0.936666 | −0.000013 | −0.000051 |
| seq11 | 0.280751 | 0.280690 | 0.280848 | +0.000097 | +0.000158 |
| seq12 | 0.944438 | 0.944317 | 0.944025 | -0.000413 | -0.000292 |
| seq13 | 0.769889 | 0.769785 | 0.770053 | +0.000165 | +0.000268 |

Seq12 saved-output audit passed all 747 condition-frame rows and exact native parity
on 249 targets. Cross-frame Dice improves on 90 frames, ties on 7 and worsens on
152; the largest decrease is 0.047795. This is a small mixed intervention effect,
with no identified causal token. Receipt: Drive
`audits/seq12_completed_triplet_audit.json`.

Seq11 is difficult in this cohort: all three methods produce false alarms on
90/90 GT-empty targets. Each recovers overlap on all 9 sampled
return events, so this is not a clean disappearance-recovery success. Cross-frame
ToMe changes present Dice by only +0.000097 vs native and Boundary F by +0.000365;
false alarms are unchanged. This sequence pulls all three interim means down.

The seq6/seq8 engineering runs had only tiny mixed effects:

| Sequence | Native Dice | Within-frame control | Cross-frame sharing | Cross-frame − native |
|---|---:|---:|---:|---:|
| seq6 | 0.864721 | 0.864632 | 0.864724 | +0.000003 |
| seq8 | 0.279446 | 0.279454 | 0.279219 | −0.000226 |

These two outcome-informed development sequences do not establish efficacy.
The full cohort is exploratory because seq6/8 informed the method design; it is
not an untouched confirmation set. A previous distinct 21-sequence **within-frame**
V-pooling study scored 0.710224 native, 0.710984 treatment and 0.711333 matched
control; its predefined screen failed. The current study tests cross-frame sharing,
not a repeat of that intervention.

Protocol, source/input hashes, progress and output artifacts are in Drive at
`outputs/experiment/spatiotemporal_tome_value_exchange_v1/cohort_v1/`.

This study stops if its frozen meaningful-improvement screen fails. No tuning on
these 21 sequences follows a failed screen.

## Original hypothesis: poorly predicted frames can contain useful groups

Reanalysis of the completed source-local diagnostic provides **local mechanism support**. Each intervention removed one 256-token spatial tile from one source frame while preserving the remaining native history, pointer inputs, positions, and all four attention layers. The two examples below use manifest frame indices; each source frame was GT-present.

| Sequence, source → target | Source's own Dice | Later native Dice | Remove most helpful group: Δ later Dice | Remove most harmful group: Δ later Dice |
|---|---:|---:|---:|---:|
| seq17, 17 → 19 | 0.003777 | 0.903168 | −0.005281 (tile14) | +0.010921 (tile10) |
| seq22, 41 → 45 | 0.000000 | 0.942278 | −0.000136 (tile01) | +0.000116 (tile06) |

Positive removal Δ means removal improved segmentation. The helpful/harmful extrema were chosen retrospectively using target GT; they are not an inference-time selection rule. These conditional effects include attention renormalization and other-memory interactions, and concern groups rather than individual token semantics.

**Verification and coverage:** 59 sampled target records and 1,180 arm receipts verified; 42 source/target PNG Dice values independently reproduced against original-hash GT. The examples' four source/target prediction hashes and scores match native Run143 exactly. There are 38 GT-present source/target pairs across 21 sequences. Of the six lowest-source-Dice pairs, four had zero measured visible-Dice effect; only seq17 and seq22 had mixed nonzero effects. This coverage does not establish how common the mechanism is.

**Practical limitation:** the original equal-sequence immediate effect was +0.000455 for uniform tile removal (95% CI [−0.000071, +0.001384]); even the best-of-16 oracle averaged +0.002184. This does not justify reviving the stopped coarse-tile pruning family as a meaningful full-video method. The unresolved contribution is a GT-free rule that predicts useful subsets and improves full propagation.

Report, all-pair CSV, scientific chart and verification receipts: Drive `within_frame_token_utility_v1/original_hypothesis_reanalysis_20261113/`. Reusable analyzer: [`analyze_poor_source_token_utility.py`](../scripts/analyze_poor_source_token_utility.py). No new model inference was run.

## Calibration follow-up: do not restart efficacy

The isolated synthetic check reproduced exactly in two processes. With identical tokens and no pointers, ordinary mass-aware ToMe has zero attention-output error; calibration produces relative squared error 0.085786 and halves output energy. Both use the same actual merge assignments. These are **not Dice scores**. The paper's scalar equations match the adapter, but its normalization/variance assumptions have not been established for this post-projection SAM2 integration. This rejects a universal fidelity guarantee, not the paper or token utilisation generally.

Keep the calibration full run stopped for its separate matched-assignment defect. Do not launch another full cohort from this synthetic evidence. Source, four-fixture table, implementation hashes and exact replay receipt: Drive `stat_calibrated_tome_polypgen_v1/source_applicability_20261113/report.md`.


**Shared-input measurement implementation (2026-11-13):** the existing native-returning attention observer now optionally evaluates calibration on the exact same merged tensors as ordinary ToMe (one matcher call). It checks merged-input hashes, unchanged pointers and finite outputs, and returns the native result. Fifteen focused tests passed, including the identical-token counterexample and rejection of the original invalid controls. Real-feature replay subsequently completed (see the result below); this observer does not repair full-run assignment replay. Source snapshots and test receipt: Drive `stat_calibrated_tome_polypgen_v1/shared_merge_observer_engineering_20261113/`.

**Real-feature applicability check complete (2026-11-13):** two independent native-returning replays exactly match (results SHA256 `b70676cb73d6403f1150a7d3cade9a981885a8fb980e6808d227398535de49fd`). On six seq2/seq3 targets × four layers, calibration worsens all48 same-merge attention comparisons. Mean relative squared error: regional0.005137→0.217725; generic0.007004→0.222524. These are not Dice scores. Native masks/presence, Q/K/V, pointers, banks and weights stay unchanged; no GT or shadow propagation. Sixteen focused tests passed; verifier negative checks reject missing shared inputs and incorrect completion hashes. Stop this calibration integration under the frozen diagnostic rule; no full efficacy or tuning follows. Protocol, source snapshots, per-read records and report: Drive `stat_calibrated_tome_polypgen_v1/shared_merge_real_feature_diagnostic_v1/`.

## Completed candidate: pointer-conditioned ToMe matching

The ToMe add-on changes how spatial memory keys are paired and mass-averaged before frozen SAM2 memory attention. It leaves pretrained SAM2 weights, attention projections, memory writes, object-presence handling, frame schedule, and decoder unchanged. Generic ToMe uses the same merge counts as the regional treatment.

**Engineering:** focused tests and the two-process seq2/seq3 real-frame gate passed. The gate checked exact native/no-op parity, pointer alignment, unchanged SAM2 weights and source banks, pointer preservation, temporal slots, and matched generic merge budgets.

**Full PolypGen cohort:** 21 eligible sequences completed. Mean equal-sequence present-frame Dice: native 0.710224; same-frame-pointer ToMe 0.710213 (Δ −0.000011; 95% sequence-bootstrap CI [−0.007212, +0.006357]; Holm p=0.9105; wins 7/21); generic ToMe 0.710073. The predeclared improvement screen failed. False alarms were 194/323 for native and 188/323 for both ToMe variants. This is no evidence of a meaningful Dice gain or a pointer-specific benefit. Outputs and independent audit: Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/`.

## Seq5 failure diagnosis and prompt-pointer candidate

Seq5 has a long shared object-loss period. All four full-video arms completed 249 post-prompt frames with the same 85/198 GT-present zero-Dice frames (frame 92; 157–182; 184–241), one undetected return out of four, and one false alarm in 51 GT-empty frames.

| Seq5 condition | Present-frame Dice | Δ vs native | IoU | Boundary F |
|---|---:|---:|---:|---:|
| Native SAM2 | 0.541987 | — | 0.517093 | 0.471983 |
| Fixed initial-prompt-pointer ToMe | 0.541171 | −0.000816 | 0.515692 | 0.471307 |
| Same-frame-pointer ToMe | 0.541059 | −0.000928 | 0.515497 | 0.470613 |
| Generic matched-budget ToMe | 0.541216 | −0.000771 | 0.515780 | 0.471766 |

Each ToMe arm has the same 85 ties and 68–71 losses versus native among present frames; all differences are very small overall. On GT-present frames 93–155, every ToMe arm is worse than native on average (fixed pointer −0.00588, same-frame pointer −0.00730, generic −0.00600). During the extended loss, the object score is nonpositive on all 84 GT-present frames 157–182 and 184–241; all methods share the same zero-Dice frames. The long zero-Dice intervals and reappearance metrics do not change. Fixed prompt-pointer matching therefore does **not** solve Seq5's loss/reacquisition problem, and is slightly below both native and its matched generic control. No significance claim is appropriate for this single development sequence.

The run verified that spatial keys were halved per stored memory frame in all four attention layers, while object-pointer keys and total spatial-key mass were preserved; all three ToMe arms used identical per-frame attention budgets. All four arms used identical evaluation frames and the same saved native reference; the saved summary means were independently recomputed from frame rows (audit passed). The candidate is not promoted to a new efficacy study. Existing native raw-mask probes also show why simply disabling object-presence suppression is unsafe: at frame 184 the unsuppressed raw mask had no GT overlap and 10.9× GT area, while the native decision correctly suppressed it.

Protocol, code, outputs, and analysis are in Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/prompt_anchor_seq5_diagnostic/`. Seq5 is development data; this diagnostic is not external confirmation.

## Latest full-cohort study: dependency-aware V-only ToMe

The frozen 21-sequence run is complete and its predefined continuation screen **failed**. Exact result hash: `03e7a559508e5b6e3f5eb5216bd9455c89fce1e9776d881d2f81ec206e7985f3`. The saved-record audit passed for all 63 sequence summaries and 42 treatment frame files; it is an arithmetic/provenance audit, not an independent inference replay. Final external source pins passed.

| Condition | Equal-sequence present-frame Dice | Boundary F | IoU | False alarms / GT-empty targets | Undetected returns / all returns |
|---|---:|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | 0.616994 | 0.651707 | 194/323 | 11/51 |
| Dependency-aware V ToMe | 0.710984 | 0.618188 | 0.652606 | 194/323 | 11/51 |
| Feature-only V control | 0.711333 | 0.618011 | 0.653027 | 194/323 | 11/51 |

- Dependency-aware V minus native: **+0.000759 Dice** (+0.076 percentage points), 95% sequence-bootstrap CI [+0.000015, +0.001939], paired Wilcoxon p=0.179, Holm-adjusted p=0.537; 12 wins, 1 tie, 8 losses.
- Dependency-aware V minus feature-only V control: **-0.000349 Dice** (-0.035 pp), 95% CI [-0.001087, +0.000052], paired Wilcoxon p=0.455, Holm p=0.911; 11 wins, 1 tie, 9 losses.
- The mean is positive versus native but tiny; it is negative versus the matched control. False alarms and missed returns are unchanged. The screen required ≥0.01 Dice vs native, ≥0.002 vs control, statistical support, and at least 14 sequence wins; these criteria were not met. No test sequences were opened after the failed screen.
- The largest value-vs-native gain is seq14 (+0.010523); without seq14 the mean gain is only about +0.00027. The seq5 treatment differences are near zero and do not resolve its sustained object loss.

Protocol/results: Drive `dependency_value_intervention_v1/full_protocol.json` and `full_attempt01/results.json`; integrity receipt `full_attempt01/audits/final_arithmetic.json`.

## Seq5: shared prolonged object loss

Native, dependency-aware V and its control all have the same 85 present-frame zero-Dice frames (92, 157–182, 184–241). On seq5, Dice is 0.541987 / 0.541849 / 0.541792; all arms have 1/51 false alarms and 1 of 4 returns undetected. The previous saved bank showed nonpositive presence scores in all six preceding memory frames during the long miss. This supports a shared SAM2 presence/reacquisition failure rather than a meaningful ToMe-specific degradation or recovery. :codex-annotation{index="1"}

A read-only native replay now probes frames 157, 184 and 242 for unsuppressed decoder logits and per-source attention; all 249 post-prompt frames matched the saved native output exactly. On the decoder's 256×256 logits grid, frame 157 raw Dice is 0.574; the −0.883 presence score suppresses it. At frame 184 raw Dice is 0.000 and the predicted area is 10.9× GT; the −1.916 score correctly suppresses this bad mask. At frame 242 raw 256-grid Dice is 0.943 and the positive score retains it. Thus removing the presence decision globally would trade a partly useful mask for a severe false positive. Attempt01 caught a dict/tuple observer bug before capture and is preserved; corrected attempt02 completed with unchanged weights and restored temporal slots. Attempt03 completed with exact parity at all249 frames. Object-query attention is concentrated on GT-object spatial cells in some memory sources, but frames178–182 have negative presence scores and were encoded as no-object, so spatial attention is not proof of object-labeled values. At the recovered frame242, object queries focus on the latest source’s GT-object cells more than background queries do in layer1. This single post-hoc pattern is descriptive, not causal. Do not bypass presence logic or promote a new token rule from seq5. Full analysis: Drive `dependency_value_intervention_v1/seq5_posthoc_analysis/seq5_raw_probe03_analysis.md`.

**Decision rule:** do not implement a proposed Seq5 recovery module unless the probe shows that memory contains useful polyp evidence that the decoder/reader fails to use. Attention maps are descriptive, not causal proof. If raw decoder and memory evidence both lack the object, stop the token-only Seq5 recovery claim.

## Evaluation scope

- 21 eligible sequences: seq2–23 excluding seq7; seq1 and seq7 excluded.
- Primary metric: per-sequence mean Dice over post-prompt GT-present frames, then equal mean over sequences. 1,689 GT-present and 323 GT-empty frames.
- SAM2.1 Hiera-Tiny, 1024×1024, frozen pretrained weights, native memory writes, frame schedule and decoder. First labeled frame prompts; subsequent frames use the native propagation protocol.
- Patient-level independence is unverified. PolypGen is exploratory. The framewise YOLO score is not a matched comparison until checkpoint, detector/prompt, resolution and evaluation policy are verified.

## Prior experiments

Older methods, exploratory results, and scope limits remain below. They are not the current candidate or confirmation evidence.

## MPM full-video comparison — complete; screen failed

All 21 eligible sequences; equal-sequence mean Dice on post-prompt GT-present frames. `value` = MPM V-only; `joint` = ToMe V-only control (not joint K/V). Native Q/K and physical token budget/history retained. Logical counts match on identical banks; propagated banks can differ.

| Method | Mean Dice | Boundary F | False alarms | Detected returns | Treatment runtime (CPU s) |
|---|---:|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | 0.616994 | 194/323 | 40/51 | cached reference |
| MPM V-only | 0.710453 | 0.618126 | 194/323 | 40/51 | 4829.2 |
| Matched-count ToMe V-only | 0.712917 | 0.620523 | 192/323 | 40/51 | 4849.9 |

| Contrast | Δ Dice | 95% sequence-bootstrap CI | Holm p | Win/tie/loss | Δ without seq8 |
|---|---:|---|---:|---|---:|
| MPM − native | +0.000229 | [-0.000085, +0.000638] | 1.000 | 12/1/8 | +0.000240 |
| MPM − ToMe | -0.002464 | [-0.006879, +0.000193] | 1.000 | 10/1/10 | -0.000490 |
| ToMe − native | +0.002693 | [+0.000018, +0.007125] | 1.000 | 10/1/10 | +0.000730 |

| Sequence | Native | MPM | ToMe control |
|---|---:|---:|---:|
| seq2 | 0.888730 | 0.888649 | 0.888633 |
| seq3 | 0.874597 | 0.877298 | 0.874888 |
| seq4 | 0.814211 | 0.814350 | 0.814015 |
| seq5 | 0.541987 | 0.542112 | 0.542135 |
| seq6 | 0.864721 | 0.864722 | 0.865558 |
| seq8 | 0.279446 | 0.279448 | 0.321396 |
| seq9 | 0.906740 | 0.907264 | 0.908030 |
| seq10 | 0.936679 | 0.936714 | 0.936768 |
| seq11 | 0.280751 | 0.280567 | 0.280509 |
| seq12 | 0.944438 | 0.943936 | 0.943193 |
| seq13 | 0.769889 | 0.769988 | 0.769589 |
| seq14 | 0.839942 | 0.840418 | 0.850247 |
| seq15 | 0.904580 | 0.904771 | 0.904936 |
| seq16 | 0.975692 | 0.975645 | 0.975595 |
| seq17 | 0.817131 | 0.816908 | 0.820540 |
| seq18 | 0.639817 | 0.639711 | 0.638363 |
| seq19 | 0.845400 | 0.845425 | 0.845375 |
| seq20 | 0.000000 | 0.000000 | 0.000000 |
| seq21 | 0.767995 | 0.770718 | 0.770856 |
| seq22 | 0.915598 | 0.915051 | 0.914753 |
| seq23 | 0.106367 | 0.105824 | 0.105885 |

MPM gain is only +0.0229 percentage points and does not exceed the control; the predefined meaningful-gain/statistical screen failed. ToMe gains +0.2693 percentage points, with about 74% of its net gain from seq8; no Holm-significant improvement. No reappearance gain. Stop this candidate without retuning. Engineering integrity passed, which does not establish efficacy.

Audit: all 63 summaries and planned comparison recomputed exactly; 42 treatment runs frame-aligned, unchanged parameter hashes/restored temporal slots; all 60 source pins match. Full-cohort inference has not been independently replayed. Artifacts: Drive `outputs/experiment/mpm_value_read_polypgen_v1/full_attempt01/{results.json,completion.json,audits/full_integrity.json}`.

Saved-trajectory diagnosis (no inference): both treatments change 1,677/2,012 prediction hashes, but recover **0 native zero-Dice frames** and introduce 0 new zero-Dice frames. MPM mainly makes small extent changes: seq21/frame32 Dice 0.2236→0.2859; seq11/frame95 0.0766→0.0479. ToMe improves seq8/frame56 0.3387→0.9642 (recall 0.2142→0.9639), but worsens seq14/frame165 0.6622→0.4510 (recall 0.6505→0.3691). This supports an extent/coverage effect on already detected objects; it does not show recovery from complete object loss or identify causal tokens. Receipt: Drive `full_attempt01/audits/saved_trajectory_diagnosis.json`.

### Completed diagnostics

These establish mechanisms or engineering integrity, **not Dice improvement**. Receipts below are under Drive `tome_region_identification_v1/post_value_source_review_20261112/`.

| Diagnostic | Verified finding | Receipt directory |
|---|---|---|
| Pair averaging | Two-process exact replay; pair identity/SDPA errors ≤1.22e−14. Lower MPM distortion did not yield higher cohort Dice. | `value_selectivity_diagnostic_v1/` |
| Native attention allocation | Four targets × four layers: conditioning14.35–22.35%, six-map history34.17–43.06%, pointers42.44–43.48%. Allocation does not establish utility. | `evidence_competition_v1/` |
| Dependency-aware grouping | Changes1694–1917/6144 window assignments per target; held-out read error worsens4/4. Not a Dice result or DTMFormer reproduction. | `dependency_matching_v1/` |
| Actual dependency intervention | Four-target and fullseq3 propagation gates each replay exactly in two processes; native/noop, source banks, weights, keys and pointer/conditioning values preserved. Synthetic batch isolation passed; multi-object video integration unverified. | `dependency_value_intervention_v1/` |
| Native Dice deficit | 194/1689 present targets have zero overlap:50.91% of equal-sequence deficit; partial overlap49.09%. | `dice_deficit_audit/` |
| Empty/disjoint classification | 187 empty predictions;7 disjoint. All187 cached empty masks have nonpositive presence logits and−1024 values; ten bank hashes verified. This does not reveal unsuppressed-mask quality. | `dice_deficit_audit/` |
| Completed MPM/ToMe on suppressed frames | Presence logit increases108/145, decreases79/42; median changes+0.001390/+0.009181. Neither recovers any of187 frames. Descriptive propagated outcomes, not same-bank attribution. | `dice_deficit_audit/` |
| CMSA-Net edge case | Exact author functions give separability1 for empty/full masks with a zero prototype. Not a real-data frequency or complete-model failure claim. | `cmsa_source/` |

Largest empty-mask counts: seq5 **85/198**, seq8 **32/53** (+2 disjoint), seq20 **30/30**, seq23 **25/35** present targets. Native median presence logit on187 empty frames:−1.587554. Native presence behavior remains unchanged.

Official FSSAM, G2TM, Representation Shift and DTMFormer source was inspected/pinned; SAM2Dual paper reviewed. Their complete methods are not verified drop-ins under the current constraints. The active dependency-aware method is an explicitly documented adaptation. Source decisions: `segmentation_source_triage.md`, `dtmformer_pinned_source/review.md`.

## Native-schedule value consolidation — completed, screen failed

Training-free official ToMe merge/unmerge on values, original keys retained; native six-map schedule. Joint K/V consolidation is the matched control. All 21 sequences completed; 56 source pins and all 42 treatment summaries passed final integrity checks.

| Condition | Mean Dice | IoU | Boundary F | False alarms | Returns detected | CPU wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | 0.651707 | 0.616994 | 194/323 | 40/51 | 2150.9 (archived) |
| V-only consolidation | 0.711331 | 0.653026 | 0.618013 | 194/323 | 40/51 | 3777.9 |
| Joint K/V control | 0.710318 | 0.651745 | 0.617876 | 192/323 | 40/51 | 4173.6 |

| Contrast | Mean Δ Dice | 95% sequence-bootstrap CI | Holm p | Wins / ties / losses | Without seq8 |
|---|---:|---|---:|---|---:|
| value_minus_native | +0.001106 | [+0.000020, +0.002980] | 1.0000 | 9 / 1 / 11 | +0.001161 |
| value_minus_joint | +0.001013 | [-0.000191, +0.002870] | 1.0000 | 12 / 1 / 8 | +0.001078 |
| joint_minus_native | +0.000094 | [-0.000806, +0.000817] | 0.9404 | 13 / 1 / 7 | +0.000084 |

| Sequence | Native | V-only | Joint K/V | Δ V-only − native |
|---|---:|---:|---:|---:|
| seq2 | 0.888730 | 0.888719 | 0.881731 | -0.000011 |
| seq3 | 0.874597 | 0.874768 | 0.874736 | +0.000171 |
| seq4 | 0.814211 | 0.813921 | 0.813837 | -0.000290 |
| seq5 | 0.541987 | 0.541792 | 0.540254 | -0.000195 |
| seq6 | 0.864721 | 0.864632 | 0.864293 | -0.000089 |
| seq8 | 0.279446 | 0.279454 | 0.279740 | +0.000008 |
| seq9 | 0.906740 | 0.907716 | 0.907672 | +0.000976 |
| seq10 | 0.936679 | 0.936717 | 0.936735 | +0.000037 |
| seq11 | 0.280751 | 0.280691 | 0.280935 | -0.000060 |
| seq12 | 0.944438 | 0.944318 | 0.944101 | -0.000119 |
| seq13 | 0.769889 | 0.769785 | 0.770218 | -0.000104 |
| seq14 | 0.839942 | 0.857668 | 0.841948 | +0.017726 |
| seq15 | 0.904580 | 0.904802 | 0.904654 | +0.000222 |
| seq16 | 0.975692 | 0.975637 | 0.975878 | -0.000055 |
| seq17 | 0.817131 | 0.817806 | 0.818369 | +0.000675 |
| seq18 | 0.639817 | 0.639630 | 0.641843 | -0.000187 |
| seq19 | 0.845400 | 0.845386 | 0.845370 | -0.000014 |
| seq20 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| seq21 | 0.767995 | 0.770812 | 0.770786 | +0.002817 |
| seq22 | 0.915598 | 0.914846 | 0.915165 | -0.000752 |
| seq23 | 0.106367 | 0.108845 | 0.108411 | +0.002477 |

V-only gains +0.1106 percentage points versus native, below the +1.0 point minimum; it wins only 9/21 sequences and does not significantly beat the joint control. Seq14 contributes approximately 76.3% of the summed native gain. The native bootstrap interval is narrowly positive, but the Holm-adjusted paired test is not significant; these test different summaries and do not establish a reliable gain. False alarms and returns are unchanged versus native. The candidate stops under its frozen screen; no parameter sweep is authorized by this result. Runtime includes matching and audit overhead, so no speed claim is made.

Evidence: Drive `outputs/experiment/native_value_consolidation_polypgen_v1/full_attempt01/{results.json,completion.json,audits/full_cohort_integrity_20261112.json}`.

### Saved-trajectory diagnosis (post hoc; no additional inference)

| Finding | Evidence |
|---|---|
| Gain concentrated in seq14 | Removing seq14 reduces mean gain to +0.000275 Dice (+0.0275 percentage points) |
| Better coverage in some frames | seq14/frame185 Dice0.4435→0.9251; recall0.2924→0.9264 |
| Worse coverage elsewhere | seq14/frame165 Dice0.6622→0.4511; recall0.6505→0.3691 |
| No recovery from zero-Dice failures | Zero recovered frames and zero newly zero-Dice frames across all21 sequences; seq8 remains34zero-Dice present frames, seq20 remains30 |

The observed benefit mainly changes mask extent on already overlapping predictions; it does not repair the complete-miss cases motivating the method. This is descriptive evidence, not proof that a particular merged token caused a change. Preserve this candidate as a completed negative screen; investigate a distinct mechanism only after literature/implementation review, not by tuning this cohort. Evidence: `audits/saved_trajectory_diagnosis_20261112.json`.

## MPM engineering diagnostic — no Dice result

Official MPM reciprocal pairing versus official ToMe bipartite matching, same local per-window representative count. Original Q/K retained; consolidated values are evaluated as shadows, and every returned output remains native. Four existing seq6/seq8 engineering targets × four attention layers repeated exactly in two fresh processes; masks/presence, weights and source banks unchanged. Mean attention-output relative squared error: MPM0.00025349 versus ToMe0.00086259. This measures fidelity to native attention, not segmentation benefit. No full-video MPM method has been evaluated.

Evidence: Drive `tome_region_identification_v1/post_value_source_review_20261112/shadow_gate_independent_receipt.json`. Existing gate-loop labels `value`/`joint` both refer to the same native-returning observer here; they are not intervention arms.

MPM real-value intervention gate also passed the same four targets in two fresh processes with byte-identical mask/presence hashes and audits. Native/no-op parity, same-bank K/V, matched logical counts, pointer preservation and unchanged parameters/banks passed. This does not cover propagated treatment banks; no MPM Dice comparison exists yet. Receipt: `post_value_source_review_20261112/intervention_gate_independent_receipt.json`.

MPM full-seq3 propagation gate passed in two independent processes: all14post-prompt frame records per arm are byte-identical, including prediction/stored-memory hashes; no-op exact native, native source schedules, physical token counts, weights and temporal slots verified. This is engineering coverage on one short sequence, not a full-cohort Dice result. Receipt: `post_value_source_review_20261112/propagation_gate_independent_receipt.json`.

## CMSA-Net source review — no SAM2 result

The official CMSA-Net repository was cloned at commit `4e4602e93211b5d0006a8a7415e8477de413b79d`. Its CMA block uses learned multi-scale feature projections and temporal attention in CMSA-Net's encoder/decoder pipeline; it does not operate on SAM2's stored spatial K/V tokens. The checked code uses two fixed reference frames; it does not implement the paper-described confidence-based DMR selection. CMSA reference frames are different from SAM2 object pointers. No compatible training-free SAM2 component was integrated and no efficacy run was started. Details: [`CMSA_MULTISCALE_TOME_REVIEW_20261113.md`](../protocols/CMSA_MULTISCALE_TOME_REVIEW_20261113.md).

## Completed full-video comparisons

| Method | Mean Dice | IoU | Boundary F | GT-empty false alarms | Returns detected | CPU wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | 0.651707 | 0.616994 | 194/323 | 40/51 | 2150.9 |
| Run155 generic local ToMe | 0.712491 | 0.654475 | 0.620593 | 190/323 | 41/51 | 5845.9 |
| Run155 hard-region ToMe | 0.734707 | 0.673994 | 0.635016 | 198/323 | 43/51 | 6121.9 |
| 12-map generic ToMe | 0.731557 | 0.670966 | 0.630262 | 192/323 | 42/51 | 5407.1 |
| 12-map soft-region ToMe | 0.731901 | 0.671359 | 0.629961 | 189/323 | 42/51 | 5808.3 |

Run155 restricts matches to hard predicted-role groups under the native selected-frame schedule. The 12-map study adds soft role similarity to matching and shares the native saturated K/V budget over a longer read. These are different interventions. Recorded CPU times include implementation/instrumentation costs and are not a controlled speed benchmark.

| Planned paired contrast | Mean Δ Dice | 95% sequence-bootstrap CI | Holm p | Wins / ties / losses |
|---|---:|---:|---:|---:|
| Run155 regional − native | +0.024483 | [-0.001366, +0.071965] | 0.8813 | 10 / 1 / 10 |
| Run155 regional − generic | +0.022217 | [-0.000691, +0.066451] | 0.5579 | 13 / 1 / 7 |
| 12-map regional − native | +0.021677 | [-0.019112, +0.082999] | 0.8721 | 8 / 2 / 11 |
| 12-map regional − generic | +0.000344 | [-0.000389, +0.001271] | 0.2730 | 12 / 2 / 7 |

Both studies failed their frozen robustness screens. Run155 has the strongest positive full-cohort point estimate, but its runtime dependency provenance is incomplete despite exact replay of all 2,012 regional targets. The 12-map saved-output audit passed. Neither result establishes a reliable method or a benefit from regional division.

### Sensitivity to seq8

| Study | Regional − native, all 21 | Without seq8 | Regional − generic, all 21 | Without seq8 |
|---|---:|---:|---:|---:|
| Run155 | +0.024483 | +0.002031 | +0.022217 | +0.000606 |
| 12-map | +0.021677 | -0.005308 | +0.000344 | +0.000345 |

Run155’s regional–generic advantage is 97.4% attributable to seq8’s contribution to the cohort mean. In seq8 it detects 3/4 returns versus 1/4 for native/generic, but false alarms rise to 12/19 from 4/19. The 12-map regional gain reverses when seq8 is excluded.

### What Run155 can and cannot establish

Run155 is retained because it has detailed saved trajectories for post-hoc case analysis. It is **not** the current efficacy headline. The later 12-map, ten-map, V-only, and pointer-conditioned studies change the intervention or protocol, so their variation cannot be called a seed-to-seed repeat of Run155. They do, however, show that the large Run155 regional advantage is not robust across the tested ToMe designs: the ten-map regional arm is −0.004575 vs native and only +0.000053 vs its generic control; the pointer-conditioned cohort is −0.000011 vs native. A replay using recovered Run155 regional-method source matched all 2,012 saved targets, which corroborates trajectory reproducibility. The original runtime import/dependency provenance is still incomplete, and this replay does not show that the gain repeats under a new seed or an independently reproduced full comparison. Keep its trajectory chart explicitly labeled as a single-run exploratory case study.

Saved Seq8 trajectories explain this tradeoff: regional ToMe changes 29/34 native zero-Dice GT-present frames into nonzero masks, including frames21–44, where native and generic remain at zero. It also raises GT-empty false alarms from4/19 to12/19, including frames around the recovery/return intervals. This is a sequence-specific continuity/false-alarm tradeoff, not evidence of robust cohort-wide benefit. Frame-level audit: Drive `155_region_constrained_tome_polypgen_v1/full/seq8_trajectory_review_20261113.md`.

### Per-sequence mean present-frame Dice

| Sequence | Present / empty targets | Native | Run155 generic | Run155 regional | 12-map generic | 12-map regional |
|---|---:|---:|---:|---:|---:|---:|
| seq2 | 59 / 0 | 0.888730 | 0.883262 | 0.889810 | 0.893665 | 0.893738 |
| seq3 | 14 / 0 | 0.874597 | 0.873375 | 0.872144 | 0.874265 | 0.874194 |
| seq4 | 45 / 2 | 0.814211 | 0.816358 | 0.816085 | 0.813905 | 0.813795 |
| seq5 | 198 / 51 | 0.541987 | 0.541241 | 0.541107 | 0.547218 | 0.547154 |
| seq6 | 61 / 5 | 0.864721 | 0.861456 | 0.860179 | 0.701496 | 0.702763 |
| seq8 | 53 / 19 | 0.279446 | 0.298546 | 0.752970 | 0.840500 | 0.840824 |
| seq9 | 41 / 6 | 0.906740 | 0.903267 | 0.903709 | 0.907341 | 0.908249 |
| seq10 | 6 / 1 | 0.936679 | 0.970110 | 0.970418 | 0.936679 | 0.936679 |
| seq11 | 135 / 90 | 0.280751 | 0.280386 | 0.281104 | 0.279846 | 0.279899 |
| seq12 | 249 / 0 | 0.944438 | 0.940974 | 0.942156 | 0.941824 | 0.942271 |
| seq13 | 198 / 44 | 0.769889 | 0.769800 | 0.769874 | 0.768536 | 0.764259 |
| seq14 | 199 / 49 | 0.839942 | 0.837637 | 0.838220 | 0.850291 | 0.857975 |
| seq15 | 115 / 0 | 0.904580 | 0.905916 | 0.905269 | 0.905034 | 0.905049 |
| seq16 | 32 / 1 | 0.975692 | 0.975664 | 0.976087 | 0.974756 | 0.974682 |
| seq17 | 43 / 3 | 0.817131 | 0.821615 | 0.825691 | 0.859165 | 0.859225 |
| seq18 | 55 / 7 | 0.639817 | 0.636087 | 0.626231 | 0.636993 | 0.637357 |
| seq19 | 54 / 1 | 0.845400 | 0.880183 | 0.876698 | 0.845018 | 0.844983 |
| seq20 | 30 / 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| seq21 | 24 / 23 | 0.767995 | 0.769748 | 0.770652 | 0.764749 | 0.764606 |
| seq22 | 43 / 2 | 0.915598 | 0.914659 | 0.914783 | 0.913392 | 0.913731 |
| seq23 | 35 / 14 | 0.106367 | 0.082020 | 0.095669 | 0.108019 | 0.108492 |

## Frame-wise YOLO-box reference

| Saved-mask averaging | Mean Dice |
|---|---:|
| All frames, 21 eligible sequences | 0.748099 |
| Matched post-prompt GT-present frames, equal sequence weight | 0.718842 |

The matched subset covers 1,689 present and 323 empty frames; YOLO false alarms are 17/323. All 2,012 saved-mask Dice values reproduce the CSV, and frame/GT/prediction hashes were checked. Detector weights, box policy, checkpoint, preprocessing, prompt cadence and runtime remain undocumented. Thus 0.748099 is not a matched present-frame benchmark, and system-level competitiveness is unproven.

| Saved output on the same post-prompt GT-present frames | Equal-sequence Dice | Difference vs YOLO | Per-sequence wins vs YOLO | Empty-frame false alarms |
|---|---:|---:|---:|---:|
| Frame-wise YOLO-box masks | 0.718842 | — | — | 17/323 |
| Native SAM2 | 0.710224 | −0.008617 | 7/21 | 194/323 |
| Generic wider-history ToMe | 0.731557 | +0.012715 | 7/21 | 192/323 |
| Region-aware wider-history ToMe | 0.731901 | +0.013060 | 7/21 | 189/323 |

The wider-history ToMe mean is numerically above the saved YOLO-mask mean under this matched denominator, but its regional condition beats YOLO on only 7/21 sequences and has 189/323 empty-frame false alarms versus 17/323. Large gains on seq3, seq4, seq9 and seq18 coexist with large losses on seq5, seq6, seq11 and seq23. The equal-frame output comparison is reproducible; the original YOLO inference recipe is not, so this is not yet a fair system-level victory or a robust-competitiveness claim. Excluding seq8, the means are YOLO 0.710510, native 0.731763, generic 0.726110 and regional 0.726455; the ToMe result still has substantial sequence-to-sequence variation.

## Ten-map full-video comparison — complete

All 21 eligible sequences; same history frames and total spatial K/V budget. Uniform/recent-dense allocation crossed with generic/soft-region matching. Frozen SAM2; no frame-quality selection.

| Method | Present Dice | Boundary F | False alarms | Returns detected | CPU seconds |
|---|---:|---:|---:|---:|---:|
| native | 0.710224 | 0.616994 | 194/323 | 40/51 | cached reference |
| uniform_generic | 0.706793 | 0.614355 | 177/323 | 40/51 | 4953.6 |
| uniform_regional | 0.706820 | 0.614610 | 184/323 | 40/51 | 5443.4 |
| recent_dense_generic | 0.705597 | 0.614051 | 178/323 | 40/51 | 4941.2 |
| recent_dense_regional | 0.705649 | 0.613701 | 180/323 | 40/51 | 5305.1 |

| Planned contrast | Mean Dice Δ | 95% sequence-bootstrap CI | Win/tie/loss | Holm p |
|---|---:|---|---|---:|
| recent_dense_regional_minus_native | -0.004575 | [-0.017683, +0.005561] | 7/2/12 | 1.00 |
| recent_dense_regional_minus_recent_dense_generic | +0.000053 | [-0.000230, +0.000410] | 8/2/11 | 1.00 |
| recent_dense_generic_minus_uniform_generic | -0.001196 | [-0.003257, +0.000310] | 10/2/9 | 1.00 |
| recent_dense_regional_minus_uniform_regional | -0.001171 | [-0.003208, +0.000325] | 12/2/7 | 1.00 |
| uniform_regional_minus_uniform_generic | +0.000027 | [-0.000262, +0.000254] | 12/2/7 | 1.00 |

**Frozen screen: failed.** Recent-dense regional loses 0.004575 Dice versus native and wins only 7/21 sequences. Excluding seq8, its delta remains −0.004807. Regional matching adds only +0.000027 (uniform) or +0.000053 (recent-dense) over generic. Fewer false alarms do not offset reduced present-frame Dice; return coverage is unchanged. No further efficacy variant is launched from this result.

Seq23 complete integrity audit passed: 49 targets/arm, 784 layer reads; native-prefix parity, budgets, pointer/mass conservation, source alignment and launch-source hashes pass. The four-target K/V engineering gate passed in two fresh processes with exactly equal results; no efficacy run followed.

### Ten-map Dice by sequence

| Sequence | Native | Uniform generic | Uniform regional | Recent-dense generic | Recent-dense regional |
|---|---:|---:|---:|---:|---:|
| seq2 | 0.888730 | 0.893564 | 0.893758 | 0.893544 | 0.893781 |
| seq3 | 0.874597 | 0.874253 | 0.874189 | 0.874188 | 0.874135 |
| seq4 | 0.814211 | 0.816743 | 0.816810 | 0.816965 | 0.816952 |
| seq5 | 0.541987 | 0.541163 | 0.541303 | 0.541323 | 0.541312 |
| seq6 | 0.864721 | 0.824065 | 0.823860 | 0.824582 | 0.824357 |
| seq8 | 0.279446 | 0.289864 | 0.289948 | 0.279530 | 0.279514 |
| seq9 | 0.906740 | 0.907955 | 0.907567 | 0.907672 | 0.907821 |
| seq10 | 0.936679 | 0.936679 | 0.936679 | 0.936679 | 0.936679 |
| seq11 | 0.280751 | 0.280320 | 0.280323 | 0.280360 | 0.280426 |
| seq12 | 0.944438 | 0.943923 | 0.943982 | 0.944086 | 0.943799 |
| seq13 | 0.769889 | 0.767062 | 0.767610 | 0.770584 | 0.770553 |
| seq14 | 0.839942 | 0.728721 | 0.729426 | 0.727146 | 0.729987 |
| seq15 | 0.904580 | 0.904712 | 0.904580 | 0.904722 | 0.904970 |
| seq16 | 0.975692 | 0.974574 | 0.974984 | 0.974986 | 0.974980 |
| seq17 | 0.817131 | 0.870100 | 0.869751 | 0.869898 | 0.869774 |
| seq18 | 0.639817 | 0.639597 | 0.640781 | 0.641294 | 0.639587 |
| seq19 | 0.845400 | 0.861884 | 0.861796 | 0.845104 | 0.845058 |
| seq20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| seq21 | 0.767995 | 0.763617 | 0.763745 | 0.763724 | 0.763762 |
| seq22 | 0.915598 | 0.915796 | 0.913651 | 0.915277 | 0.915301 |
| seq23 | 0.106367 | 0.108055 | 0.108479 | 0.105863 | 0.105890 |

## Where the ten-map design degrades (post-hoc)

Recent-dense regional versus native, saved full-video trajectories only:

| Sequence | Dice Δ | Observed failure / gain |
|---|---:|---|
| seq14 | −0.109955 | 36/199 present frames have Dice=0 versus 0/199 native; recall drops by 0.115580. |
| seq6 | −0.040364 | Frame50/51 precision falls ≈0.99→0.36/0.34 (leakage); frame52/53 recall falls ≈0.89/0.83→0.55/0.51 (missed foreground). |
| seq17 | +0.052642 | Mean precision +0.101304 and Boundary F +0.100972, despite one additional zero-Dice frame. |
| seq8 | +0.000069 | Both methods have 34/53 present frames with Dice=0; Run155's large recovery is not reproduced. |

For all five inspected sequences (6/8/14/17/19), the first changed prediction coincides with the first compressed/expanded-history read. This is temporal association: compression, wider history and subsequent memory propagation are not separated. Precision on an empty prediction is defined as 1 by the metric implementation; the seq14 precision increase is not evidence of better localization. Source: completed cohort `failure_analysis/trajectory_analysis.json`.

## Fixed-bank diagnosis of seq6 degradation

Six reserved targets passed two-process exact replay; all native scores match archive. Same native bank, no memory writes:

| Frame | Native | Dense ten-map | Compressed ten-map | Full-rollout uniform generic |
|---|---:|---:|---:|---:|
| 50 | 0.940996 | 0.947998 | 0.946401 | 0.509620 |
| 51 | 0.914540 | 0.912088 | 0.911899 | 0.482957 |
| 52 | 0.933979 | 0.933915 | 0.932807 | 0.700014 |
| 53 | 0.907014 | 0.907071 | 0.906364 | 0.673111 |

A single compressed read on native history does not reproduce the large rollout loss at these frames. This motivated the bank/component interventions below; different banks prevent attributing the gap solely to propagation. Seq8 frames21/32 remain Dice=0 in all three fixed-bank arms. No new efficacy candidate is promoted. Source: `kv_selectivity_mechanism_v1/history_compression_reserved01/analysis.json`.

Seq6 timing: compression starts at frame32; frames32–37 remain near/slightly above native. At frame50, Dice falls 0.9410→0.5096 despite object-presence logit +8.10; frame51 is 0.9145→0.4830 at +7.60. These losses are not rejection by a negative presence score. The treatment bank was recovered by a bounded replay through frame53; both independent processes matched all29 archived prediction/stored-state records and bank fingerprints.

## Implementation audit

| Finding | Resolution / scope |
|---|---|
| Diagnostic norm weighting used log mass | Corrected to actual cluster mass; historical affected norm fields flagged. Dice and SDPA were unaffected. |
| FP64 diagnostic reporter cast to FP32 | Corrected and regression-tested. Original FP32 identity failure retained; FP64 residual supports numerical precision as its cause. |
| Dense receipt schema assumption | Corrected; same-bank gate and native scoring match archive in independent processes. |
| Regional benefit | No robust gain over matched generic controls; ATC/PiToMe remain stopped. |

## DAVIS transfer — completed Run164

| Method | Official equal-object J&F (%) |
|---|---:|
| Native SAM2 | 89.01 |
| Generic local ToMe | 88.03 |
| Run155 regional ToMe | 88.03 |

All 30 DAVIS 2017 validation videos and 61 object IDs completed. Equal-video paired regional − native J&F is −0.004684 (95% CI [−0.012378, −0.000285], Holm p=0.003728; 7/30 wins). Regional and generic are indistinguishable. This does not support transfer of the PolypGen gain; DAVIS val was already exposed during development. No complete SA-V/MOSE evaluation is available. LVOS v2 remains the prospective long-term confirmation set: the [official evaluation archive](https://lingyihongfd.github.io/lvos.github.io/dataset.html) metadata was reverified (11.83 GB), but raw contents, integration and untouched-data status are not yet established.

## Recovered treatment-bank diagnosis

Four seq6 loss frames; all reader interventions use the same frozen bank within each row. Original treatment reader exactly reproduces archived masks/presence; two independent processes match.

| Frame | Native-bank / native reader | Treatment-bank / native reader | Treatment-bank / dense ten | Treatment-bank / ToMe ten |
|---|---:|---:|---:|---:|
| 50 | 0.940996 | 0.512935 | 0.512641 | 0.509620 |
| 51 | 0.914540 | 0.493879 | 0.483883 | 0.482957 |
| 52 | 0.933979 | 0.702369 | 0.700004 | 0.700014 |
| 53 | 0.907014 | 0.673661 | 0.672564 | 0.673111 |

Changing the reader alone does not recover the degraded treatment bank. Native versus treatment bank content is the main observed difference at these targets; spatial memory, object pointers and mask-conditioned history remain coupled. This is a bounded mechanism result, not a new segmentation method or recovery rollout. Receipt: `kv_selectivity_mechanism_v1/treatment_bank_reader_diagnostic01/analysis.json`.

## Spatial memory versus pointers — diagnostic intervention

Recovered seq6 treatment bank; same frozen compressed-ten reader. Replace one component from the cached native trajectory; masks/presence metadata stay with treatment. Spatial features move together with their positional encodings.

| Frame | Original treatment | Native spatial memory | Native pointers only |
|---|---:|---:|---:|
| 50 | 0.509620 | 0.942361 | 0.510570 |
| 51 | 0.482957 | 0.915511 | 0.677971 |
| 52 | 0.700014 | 0.932200 | 0.686794 |
| 53 | 0.673111 | 0.904206 | 0.801380 |

Spatial-memory substitution restores high Dice in these four selected loss cases; pointer substitution gives inconsistent partial recovery. Both processes match exactly; banks/weights remain unchanged and no new memories are written. This localizes a mechanism in these cases, not a deployable correction: a native reference bank is not supplied to the method at inference. Source: `kv_selectivity_mechanism_v1/memory_component_diagnostic01/analysis.json`.

## Stored spatial-memory audit — seq6 frames25–53

| Check | Result |
|---|---|
| Positional encodings / tensor shape / dtype | Identical across both banks |
| First differing mask logits and spatial memory | Frame32 in both |
| Object-presence sign differences | 0/29 frames |
| Frame49 binary-mask disagreement | 0.084% of the low-resolution grid; memory relative squared difference 0.00631 |
| Frame50 binary-mask disagreement | 10.11%; memory relative squared difference 0.04915 |

Native SAM2 `_encode_new_memory` uses current image features plus scaled sigmoid mask probabilities. Thus near-equal binary masks do not imply equal spatial memory. The observed loop is compatible with read perturbation → changed soft masks → changed spatial writes → later read differences; tensor comparison alone does not prove each causal link. No position-layout defect or presence-sign switch was found in this segment. Receipt: `kv_selectivity_mechanism_v1/recovered_spatial_memory_audit.json`.

## Evidence

Paths relative to Drive `AdeSEG/outputs/experiment/`:

- Current study/source snapshot: `native_value_consolidation_polypgen_v1/`.
- Run155: `155_region_constrained_tome_polypgen_v1/full/`.
- 12-map study: `tome_region_identification_v1/wider_history_fixed_budget_v1/full_attempt01/`.
- Completed ten-map study: `ten_map_allocation_tome_polypgen_v1/full_attempt02_resumed/`.
- Bank/component diagnostics and gates: `tome_region_identification_v1/kv_selectivity_mechanism_v1/`.
- YOLO provenance: `tome_region_identification_v1/framewise_baseline_audit01/`.
- DAVIS: `164_davis17_region_constrained_tome_v1/full/`.
- Full archived reports: `tome_region_identification_v1/reporting_snapshot_20261112/`.

[Method](REGION_TOKENS.md) · [Active goal](../protocols/ACTIVE_RESEARCH_GOAL.md)

## Seq5 failure analysis and attention-aware value fusion — 2026-11-13

Seq5’s prolonged loss is shared by native SAM2, value-only ToMe and the feature control: all have the same 85 GT-present zero-Dice frames (92, 157–182, 184–241), one missed return of four, and negative object-presence scores on the lost frames. The evidence points to a native presence/reacquisition failure; it does not show that ToMe materially caused it. :codex-annotation{index="1"}

| Candidate | What changes | What stays native | Status |
|---|---|---|---|
| Attention-aware MCTF value fusion | Uses K similarity and current-query attention importance to choose which V vectors to weighted-merge within local 4×4 memory tiles; restores each merged group to its original spatial slots. | SAM2 weights, Q, K, all spatial slots, pointers, frame schedule, writes, decoder and presence handling. | Seq5 diagnostic completed: it did not shorten object-loss intervals or improve Dice. |

The implementation reuses the [official MCTF implementation](https://github.com/mlvlab/MCTF) and pins the [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Multi-criteria_Token_Fusion_with_One-step-ahead_Attention_for_Efficient_Vision_Transformers_CVPR_2024_paper.html). The paper evaluates ViT image classification and reports token fusion by similarity, informativeness and fused-token size; it does not establish a SAM2 or video-segmentation gain. This adaptation changes **values only** and restores dense source slots: it does not reduce SAM2’s attention token count or claim inference speedup. It also leaves native object-presence suppression intact, so it should not be expected to override the shared Seq5 absence decision.

Seq5 is a post-hoc, exploratory diagnostic, not confirmation. On 198 GT-present frames, native SAM2 scored Dice 0.541987; generic ToMe value fusion 0.541792; MCTF uniform-importance fusion 0.541966; and attention-aware MCTF 0.541826. Attention-aware MCTF was −0.000161 Dice versus native and −0.000141 versus the matched MCTF uniform control. It won 43, tied 86 and lost 69 of the 198 present frames against that control. All four conditions had the same 85 zero-Dice frames (92, 157–182, 184–241), the same four reappearance outcomes (three detected, one missed), and 1/51 false alarms on GT-empty frames. Thus attention-aware fusion did not resolve Seq5’s prolonged disappearance and provides no useful gain here. :codex-annotation{index="1"}

| Seq5 condition | Present-frame Dice | IoU | Boundary F | Zero-Dice GT-present frames | Missed returns | Empty-frame false alarms |
|---|---:|---:|---:|---:|---:|---:|
| Native SAM2 | 0.541987 | 0.517093 | 0.471983 | 85 | 1/4 | 1/51 |
| Generic ToMe value | 0.541792 | 0.516752 | 0.471497 | 85 | 1/4 | 1/51 |
| MCTF uniform importance | 0.541966 | 0.517062 | 0.472098 | 85 | 1/4 | 1/51 |
| MCTF attention-aware | 0.541826 | 0.516814 | 0.471581 | 85 | 1/4 | 1/51 |

The longest zero-Dice interval is unchanged at 58 sampled frames (184–241); the earlier 26-frame interval (157–182) is also unchanged. On native SAM2, the object-presence score is nonpositive on all 85 GT-present zero-Dice frames (85/198 present frames, 42.9%). It is also negative on 50/51 GT-empty frames; only one GT-empty frame is a false alarm. In the official decoder path, the score is computed from the object-score head applied to the decoder's object token. SAM2 then replaces the mask logits with `NO_OBJ_SCORE` whenever the score is nonpositive and writes a no-object spatial embedding into memory. Thus the visible zero masks are caused immediately by the presence decision; the saved outputs alone do not establish why the decoder's object token becomes negative. See [`sam2_base.py`](../external/sam2/sam2/modeling/sam2_base.py) and [`mask_decoder.py`](../external/sam2/sam2/modeling/sam/mask_decoder.py). This supports the failure diagnosis, not a claim that the memory fusion caused or fixed it. Attention-aware MCTF took 831 seconds versus 586 seconds for its uniform control and 476 seconds for generic ToMe in these runs; because the method restores all spatial slots and does not reduce key count, it has no token-count or speed benefit. This single, post-hoc sequence cannot establish a general effect, but it is sufficient to stop pursuing this MCTF variant for Seq5.

The complete output, frozen protocol, checksums and concise audit are on Drive under `tome_region_identification_v1/mctf_attention_memory_v1/seq5_diagnostic_attempt06/`. No full-cohort MCTF run is warranted from this negative diagnostic. The next work should target the measured failure location: understand how the native presence score suppresses raw masks, while preserving its protection against false positives. Do not claim a solution from attention/value changes alone.

**Seq5 in the active pointer-conditioned ToMe cohort (interim):** native SAM2 Dice is 0.541987, pointer-conditioned ToMe 0.541059, and the matched generic local ToMe 0.541216. All three have the same 85/198 GT-present zero-Dice frames (92, 157–182, 184–241), 1/4 missed returns, and 1/51 GT-empty false alarms. Thus pointer-conditioned pairing does not recover Seq5’s long absent-object intervals; the difference from native is small. This is one sequence in the ongoing full-cohort run, not a cohort conclusion.

The saved logits confirm that this is not a failed suppression implementation: on the 85 shared misses, pointer-conditioned ToMe changes the mean object-presence logit by −0.0472 vs native (median −0.0629); generic ToMe changes it by −0.0426 (median −0.0583). Neither changes the positive/negative decision on any missed frame. During frames 157–182 and 184–241, both ToMe variants have slightly more negative mean logits than native. This token-pairing rule therefore does not supply a useful reappearance signal on Seq5. Full matched-frame audit and hashes: Drive `full_attempt01/audits/seq5_presence_margin_audit_20261113/`; reproducible analyzer: [`audit_seq5_presence_margins.py`](../scripts/audit_seq5_presence_margins.py).

**Seq5 recovery candidate review (2026-11-13):** Seq5’s prolonged loss is shared by native and both ToMe arms, and disabling the native object-presence suppression is unsafe: it would retain a large incorrect mask at frame184. The prompt-anchor concept is therefore limited to changing token matching, while leaving presence handling and false-positive protection intact. Code review found that the first prototype compared raw memory features with arbitrary pointer chunks; the revised candidate compares projected spatial K with projected prompt-pointer K within each native attention layer. This is a more defensible matching signal because both tensors are in that layer’s learned key space. It remains a hypothesis: the fixed prompt pointer may still be a poor reference after appearance changes, and Seq5 cannot serve as independent evidence because it motivated the candidate. Three focused tests pass; the revised seq2/seq3 engineering gate and any Seq5 diagnostic are queued until the active full-cohort run finishes. Implementation: [`prompt_anchor_tome.py`](../modeling/prompt_anchor_tome.py); frozen v2 protocols: [`prompt_anchor_tome_gate_v2.json`](../protocols/prompt_anchor_tome_gate_v2.json), [`prompt_anchor_tome_seq5_diagnostic_v2.json`](../protocols/prompt_anchor_tome_seq5_diagnostic_v2.json).


**Seq11 interim integrity-audited comparison:** native, pointer-conditioned, and matched generic Dice are 0.280751, 0.280258, and 0.280272. Both ToMe arms are slightly below native; pointer vs generic differs by −0.000014. Paired pointer/native present-frame wins/ties/losses are 41/0/94. All arms have false alarms on 90/90 GT-empty targets, no zero-Dice scores on 135 GT-present targets, and detect 9/9 returns. The present-frame Dice is low in all arms (median ≈0.094, precision ≈0.20, recall ≈0.90), consistent with broad masks/poor localization. This sequence does not support a pointer-specific benefit. All-arm control audit: Drive `full_attempt01/audits/seq11_all_arms_budget_integrity_audit_20261113.md`; the earlier two-arm receipt is retained as interim history.

**Seq12 interim integrity-audited comparison:** native, pointer-conditioned, and matched generic Dice are 0.944438, 0.940333, and 0.939439; Boundary F is 0.985794, 0.982173, and 0.981999. Pointer ToMe is −0.004104 Dice vs native and +0.000894 vs generic. All 249 post-prompt frames are GT-present, so this sequence gives no evidence about empty-frame false alarms or reappearance. Integrity checks confirmed exact native archived prediction/state hashes, unchanged SAM2 parameter digest, four attention layers for each ToMe target, spatial key-mass conservation, and matching pointer/generic per-target layer/source-frame token counts. The frozen runner recorded 1,763,328 exact per-window quota checks. Metrics were independently recomputed from per-frame files. Audit: Drive `full_attempt01/audits/seq12_all_arms_integrity_audit_20261113.md`. This one sequence is descriptive only.

A separate exact native replay already showed why globally bypassing presence suppression is unsafe: on the decoder's 256×256 logits grid, frame157 pre-suppression Dice is 0.574 but it is suppressed; at frame184 raw Dice is 0 with predicted area 10.9× GT, so suppression prevents a large false positive; at frame242 raw grid Dice is 0.943 and the positive score retains it. Attention overlap with GT locations is descriptive, not causal evidence. See Drive `dependency_value_intervention_v1/seq5_posthoc_analysis/seq5_raw_probe03_analysis.md`.

**Next method in the existing run:** same-frame pointer-conditioned ToMe matching compares a stored feature token with the pointer from its own source frame. It can alter which spatial K/V tokens are merged and therefore what the decoder reads, while keeping the native presence decision unchanged. This is the cleanest in-scope candidate already implemented; it may influence the decoder's presence score indirectly, but it cannot directly overrule it. The active run must finish before deciding whether it helps Seq5 or the cohort. We will not tune a Seq5-specific threshold or bypass presence suppression, because that would change the method and could turn correct empty-frame rejection into false alarms.

**Prior-work check:** DAM4SAM targets distractors and reappearance using separate appearance/distractor memories, but changes memory admission with frame-level criteria; SAM2Long uses pathway selection; neither isolates token-level attention under the current fixed-schedule/no-gate constraints. Cutie supports object-level context in VOS but is a separately trained architecture, not a drop-in frozen-SAM2 module. Efficient-SAM2 demonstrates memory-token routing for efficiency, not a reliable Dice gain. These works motivate the problem but do not validate a next method here. Source-by-source fit and links are recorded in Drive `seq5_diagnostic_attempt06/strategy_review.md`.

**Implementation decision:** Do not alter or bypass SAM2’s presence decision from this one sequence; the raw-mask examples show that suppression prevents both useful and harmful masks. The active pointer-conditioned ToMe cohort run remains frozen and will be reported against its predeclared screen. Separately, inspection of `fixed_no_obj_ptr` motivated an unevaluated prompt-anchor ToMe candidate: use the initial prompted object pointer as the matching reference on all memory frames. Its helper and seq2/seq3 engineering gate are implemented, but neither real-frame inference nor a full-video efficacy run has been launched. Seq5 informed the idea, so a Seq5 result would be exploratory only. Prior-work alternatives include [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf), which uses multiple prediction pathways, and [Cutie](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html), which adds trained object queries alongside dense pixel features; neither validates prompt-anchor ToMe. The new implementation and frozen protocols are tracked in `protocols/ACTIVE_RESEARCH_GOAL.md`.


**Matched YOLO saved-output comparison on seq2–seq11 (partial, 9/21 sequences):** exact frame IDs, manifest filenames, GT hashes, and YOLO prediction hashes passed for 786 targets. Equal-sequence present-frame Dice is native 0.709762, pointer-conditioned 0.711951, matched generic 0.711849, and saved YOLO 0.646376. Pointer ToMe gains +0.002189 vs native but only +0.000102 vs generic; it wins 3/9 sequences against YOLO. Excluding seq8 gives +0.003285 vs native, but excluding short seq10 (6 present frames) reverses the delta to −0.001721. GT-empty false alarms are pointer 102/174 vs YOLO 9/174. The low YOLO partial mean is driven by sequence-specific failures, so this is a descriptive subset comparison, not evidence of robust superiority. Full-cohort matched YOLO reference remains 0.718842 present-frame Dice; detector/checkpoint provenance remains incomplete. Audit: Drive `full_attempt01/audits/matched_yolo_partial_completed_sequences_20261113.md`.

**Pointer-conditioned ToMe full cohort — interim seq13 audit:** on all 242 post-prompt targets, present-frame Dice is native **0.769889**, pointer-conditioned **0.769826**, and matched generic ToMe **0.769607**. Boundary F is 0.673600 / 0.673556 / 0.672861; GT-empty false alarms are 42/44 in all arms. Pointer ToMe is −0.000063 vs native and +0.000219 vs generic. The saved-output audit passed frame/presence alignment, per-frame summary arithmetic, unchanged SAM2 parameter digest, four memory-attention layers per target, spatial mass conservation, and matching source frames/token counts. The generic runner recorded 1,713,152 per-window budget checks. Metrics were recomputed from saved per-frame score rows; treatment masks themselves are not stored, so mask-level metrics were not independently recomputed. This sequence is descriptive, not a cohort result. Full-video run remains active at seq14. Audit and script: Drive `full_attempt01/audits/seq13_all_arms_integrity_audit_20261113/`; local reusable script [`audit_pointer_conditioned_tome_sequence.py`](../scripts/audit_pointer_conditioned_tome_sequence.py).

**Pointer-conditioned ToMe full cohort — interim seq14 audit:** native/pointer/generic mean present-frame Dice is 0.839942/0.839046/0.837983; pointer is −0.000896 vs native and +0.001063 vs generic. Boundary F is 0.641242/0.636046/0.637286; false alarms are 32/49 empty frames in every arm. Runtime is 189.5/560.3/590.0 seconds. Saved-output integrity passed; full cohort remains active and these per-sequence values are descriptive only. Audit: Drive `full_attempt01/audits/seq14_all_arms_integrity_audit_20261113_0423/`.

**Interim aggregate through seq14 (12/21 completed):** equal-sequence present-frame Dice is native 0.745178, pointer ToMe 0.746397 and matched generic ToMe 0.746139. Pointer ToMe is +0.001220 vs native but wins only 3/12 completed sequences; against generic it is +0.000258 with 7/12 wins. False alarms are 176/267 vs native 177/267. All 12 sequence saved-output audits pass. This completion-order subset is descriptive only; the frozen 21-sequence inference and statistical screen are still pending. Audit table and receipts: Drive `full_attempt01/audits/completed_seq2_14_interim_audit_20261113_0427/`.

**Updated interim aggregate through seq15 (13/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.757439/0.758662/0.758439. Pointer ToMe is +0.001223 vs native, with only 4/13 sequence wins, and +0.000223 vs the matched generic control, with 7/13 wins. False alarms are 176/267 vs native 177/267. All 13 sequence integrity audits passed. This remains descriptive and does not satisfy the frozen full-cohort screen. Receipt/table: Drive `full_attempt01/audits/completed_seq2_15_interim_audit_20261113_0431/`.

**Updated interim aggregate through seq16 (14/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.773029/0.774156/0.773949. Pointer ToMe is +0.001127 vs native with 4/14 sequence wins, and +0.000207 vs generic with 7/14 wins. False alarms are 177/268 vs native 178/268. All 14 sequence integrity audits passed. This completion-order subset is descriptive only; the frozen 21-sequence run remains active. Receipt/table: Drive `full_attempt01/audits/completed_seq2_16_interim_audit_20261113_0439/`.

**Updated interim aggregate through seq17 (15/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.775969/0.777302/0.777113. Pointer ToMe is +0.001333 vs native, but wins 5/15 sequences; its gain over matched generic ToMe is only +0.000189 (7/15 wins). False alarms are 180/271 vs native 181/271. All 15 sequence integrity audits passed. Descriptive interim only; the full run is active. Audit table: Drive `full_attempt01/audits/completed_seq2_17_interim_audit_20261113_0442/`.

**Updated interim aggregate through seq18 (16/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.767459/0.768693/0.768497. Pointer ToMe is +0.001234 vs native with 5/16 wins, and +0.000196 vs generic with 8/16 wins. False alarms are 182/278 vs native 183/278. All 16 sequence integrity audits passed. These are descriptive results only; the full run is active. Audit table: Drive `full_attempt01/audits/completed_seq2_18_interim_audit_20261113_0442/`.

**Updated interim aggregate through seq19 (17/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.772044/0.775217/0.775034. Pointer ToMe is +0.003173 vs native with only 6/17 wins, but only +0.000183 vs matched generic ToMe (8/17 wins). False alarms are 182/279 vs native 183/279. Seq19 itself improves almost equally under both ToMe conditions, so it does not demonstrate pointer-specific utility. All 17 sequence integrity audits passed; the cohort remains active. Audit table: Drive `full_attempt01/audits/completed_seq2_19_interim_audit_20261113_0446/`.

**Updated interim aggregate through seq20 (18/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.729153/0.732150/0.731977. Pointer ToMe is +0.002997 vs native (6 wins, 1 tie, 11 losses) but only +0.000173 vs generic (8 wins, 1 tie, 9 losses). False alarms are 182/284 vs native 183/284. Seq20 scores zero Dice in all three arms. All 18 sequence audits pass. This is interim descriptive evidence, not the final cohort result. Audit table: Drive `full_attempt01/audits/completed_seq2_20_interim_audit_20261113_0449/`.

**Updated interim aggregate through seq21 (19/21 completed):** native/pointer/generic equal-sequence present-frame Dice is 0.731197/0.734108/0.733945. Pointer ToMe is +0.002911 vs native (7 wins, 1 tie, 11 losses) but only +0.000163 vs generic (8 wins, 1 tie, 10 losses). False alarms are 186/307 vs native 192/307. The seq21 false-alarm reduction is shared by both ToMe variants. All 19 sequence audits pass; two sequences remain. Audit table: Drive `full_attempt01/audits/completed_seq2_21_interim_audit_20261113_0454/`.

## Pointer-conditioned ToMe full PolypGen result — 2026-11-13

All 21 eligible sequences completed (seq1 and seq7 excluded). Independent saved-output integrity audits passed. Metrics are equal-sequence means of present-frame scores.

| Method | Mean Dice | Δ vs native | Sequence wins vs native | GT-empty false alarms |
|---|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | — | — | 194/323 |
| Pointer-conditioned ToMe | 0.710213 | −0.000011 | 7/21 | 188/323 |
| Matched generic local ToMe | 0.710073 | −0.000151 | 8/21 | 188/323 |

Pointer ToMe vs native: 95% sequence-bootstrap CI [−0.007212, +0.006357], Holm-adjusted *p*=0.9105. Pointer ToMe vs generic: +0.000140 (CI includes zero). The frozen meaningful-gain screen failed. The small pooled false-alarm reduction is shared by both ToMe variants and does not establish a pointer-specific gain. Full table and receipts: Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/`; all-sequence audit receipt: `audits/final_full_cohort_integrity_audit.json`.

## Seq5 loss diagnosis and prompt-anchor diagnostic — 2026-11-13

The three cohort arms each score zero on the same 85/198 GT-present frames: frame 92, 157–182, and 184–241. On the other 113 present frames, mean Dice is 0.949676 native, 0.948051 pointer ToMe, 0.948325 generic ToMe. This is a shared presence/reacquisition failure, not evidence that ToMe caused the long loss. A blanket presence bypass is unsafe: at frame184 the native raw mask has zero GT overlap and 10.9× GT area.

| Candidate | Intervention | Validation |
|---|---|---|
| Fixed-prompt-pointer ToMe | Pair spatial keys using affinity to the initial prompt pointer after the same frozen K projection and positional transform; retain native presence handling, memory writes, decoder, frames and ToMe budget. | Seq2/seq3 real-frame gate passed. Completed Seq5 result: Dice 0.541171; below native (0.541987) and matched generic ToMe (0.541216). Same zero-Dice intervals and missed return. |

This Seq5 study is exploratory because Seq5 informed the candidate. Its four arms completed: native 0.541987, fixed-prompt pointer 0.541171, same-frame pointer 0.541059, and matched generic ToMe 0.541216. Each has the same 85 zero-Dice frames (92, 157–182, 184–241), 1/4 missed returns, and 1/51 empty-frame false alarms. No pointer-based matching variant improved recovery.

### Strategy audit: what has and has not been resolved

| Strategy already evaluated | Seq5 result | What it tells us |
|---|---|---|
| Run155 local ToMe (generic / regional) | Dice 0.541241 / 0.541107; 85 zero-Dice frames in both | Regional matching does not explain or repair the shared long loss. |
| Attention-aware MCTF value fusion | Dice 0.541826; same two long zero-Dice intervals | Using current-query attention to guide value fusion did not restore the object. |
| Pointer-conditioned ToMe (same-frame / initial-prompt pointer) | Dice 0.541059 / 0.541171; same two long intervals | Adding object-pointer compatibility to the merge metric did not provide a recovery cue. |
| Fixed-budget wider-history ToMe (generic / regional) | Dice 0.547218 / 0.547154; 84 zero-Dice frames; intervals 157–182 and 184–241 remain | More history and a dense conditioning map slightly raise mean Dice and recover frame 92, but do not fix the prolonged loss or improve regional over generic. This combined intervention does not isolate the conditioning-map effect. |

**Implementation review.** The ToMe and MCTF interventions preserve SAM2's normal object-presence decision. The failure evidence does not indicate an implementation defect: the raw decoder mask at frame 184 has zero GT overlap and an area 10.9 times the GT area, so the negative presence score prevents a large false positive. At frame 157 the raw mask is partly aligned (Dice 0.574), but its negative score suppresses it. A global presence bypass would trade these two cases unsafely. No source-code correction is justified by these results.

**Decision.** Stop Seq5-specific ToMe tuning. The remaining failure is a native presence/reacquisition problem, and token merging variants tested so far do not change the relevant presence decisions. Published SAM2Long and DAM4SAM address reappearance with pathway or memory-management choices, which would change this study's fixed memory/frame policy; Efficient-SAM2's token retrieval is primarily an efficiency method and does not establish a Dice-recovery method for this failure. The next valid token-level study must show, on fixed saved banks, that a specified set of memory values causally improves the unsuppressed decoder mask and preserves empty-frame rejection before any full-video rollout. If such evidence is absent, do not claim ToMe solves Seq5. No process is currently running.

Protocol, artifacts and audit: Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/prompt_anchor_seq5_diagnostic/`; wider-history source outputs: `tome_region_identification_v1/wider_history_fixed_budget_v1/full_attempt01/{global,regional}/seq5/frames.json`.

## Seq5 failure-window token-group diagnostic — 2026-11-13

**Resolution note:** the earlier native raw-mask probe reports frame157 Dice **0.574** on the decoder's 256×256 logits grid, after nearest-neighbor downsampling the GT to that grid. The later fixed-bank diagnostic reports **0.602104** after the same raw logits are postprocessed to original image resolution and scored against the original-resolution GT. These are different scoring grids, not conflicting replays. Both use the same object-presence logit (−0.8830), and both show that native presence handling suppresses the mask. Frame184 raw Dice is zero under both evaluations.

Two native failure frames were tested with fixed-bank single-tile removals. The same 160 counterfactual reads were byte-identical in two fresh processes. This is a post-hoc mechanism check, not an efficacy study: Seq5 frames 157 and 184 were selected because they fail.

| Target | Native raw Dice | Native visible Dice | Native presence logit | Result of 80 tile removals |
|---|---:|---:|---:|---|
| Frame 157 | 0.602104 | 0.000000 | −0.8830 | All 80 scores stayed negative; visible Dice remained 0. Mean raw Dice effect was −0.000201 to +0.000152 by source frame. Best post-hoc tile improved raw Dice by +0.002147 but did not change the output mask. |
| Frame 184 | 0.000000 | 0.000000 | −1.9156 | All 80 raw masks remained Dice 0; no removal corrected localization. Presence suppression avoids displaying a raw mask with 421,814 foreground pixels (about 10.9× GT area). |

At frame 157 the raw decoder mask has partial localization (precision 0.503, recall 0.749), then native object-presence handling suppresses it. At frame 184 both localization and presence fail; bypassing presence would expose a large false positive. Removing one 16×16 tile from one intermediate source frame did not fix either failure. The “best tile” was chosen using GT after observing outcomes, so it is not a deployable rule.

**Decision:** this test gives no basis for another Seq5-specific ToMe modification or a full-cohort run. Stop tile-removal tuning on this sequence: the tested source groups did not restore frame 157’s suppressed output or repair frame 184’s localization. This does not disprove all token utilisation; it only bounds this single-tile fixed-bank intervention on two selected frames. The full-cohort meaningful Dice-improvement goal remains unmet.

**Solution review:** the Seq5 failure is inherited from native SAM2 rather than a material ToMe regression. Keep native presence suppression: it discards a partly aligned mask at frame157 but also blocks the extreme false positive at frame184. The official [SAM2Long implementation](https://github.com/Mark12Ding/SAM2Long) addresses disappearance/reappearance by maintaining and pruning multiple prediction paths; official [DAM4SAM code](https://github.com/jovanavidenovic/DAM4SAM) adds distractor-aware memory and an introspection-triggered update. These are credible reappearance strategies, but they change path or memory management and use selection/update decisions. They do not fit this study's fixed-frame, fixed-memory, no-gating token-utilisation protocol. Since regional/generic ToMe, pointer-conditioned matching, MCTF, wider history and the fixed-bank removals did not change Seq5's long failure, no further Seq5-specific model modification is supported. A ToMe change here would be speculation, not a justified fix.

**Efficient-SAM2 SMR source audit (2026-11-13):** corrected trace confirms the official checkout applies attention-derived top-k masks to spatial memory K/V and preserves retained RoPE positions. This is hard token pruning for efficiency, not ToMe and not a validated Seq5 remedy; the paper reports 1.68× speedup with a 1.0% SA-V accuracy drop. It conflicts with the no-gating constraint, so no experiment was launched. See [`EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md`](../protocols/EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md).

**Related 2026 paper:** Puspitasari et al. report a SAM2 spatiotemporal memory-token pruning method that prunes up to 87% of tokens and reaches 1.7× speedup with negligible performance disturbance on their benchmarks ([TPAMI paper](https://doi.org/10.1109/TPAMI.2026.3723878)). It supports unequal token utility as an efficiency finding; it does not establish a Dice gain on PolypGen, and its hard pruning is outside the current no-gating constraint. No implementation or experiment is claimed.

**Temporal-merging source screen (2026-11-13):** official STTM (ICCV 2025) merges cross-frame nodes using a cosine-similarity threshold, which conflicts with the no-threshold protocol and was evaluated on video QA, not dense segmentation. *Video, How Do Your Tokens Merge?* applies fixed-budget ToMe inside action-recognition encoders, not SAM2 memory attention; it is not evidence for a new VOS method. Neither is a supported Seq5 repair. Source review: [`TEMPORAL_TOKEN_MERGING_SOURCE_REVIEW_20261113.md`](../protocols/TEMPORAL_TOKEN_MERGING_SOURCE_REVIEW_20261113.md).

**Run155 object-size check (post hoc, no inference rerun):** GT area strata do not support a simple “ToMe overpredicts large polyps and underpredicts small ones” explanation. Regional predicted/GT area medians are 0.998×, 0.957×, 0.988×, and 0.960× across `<1%`, `1–5%`, `5–15%`, and `≥15%` GT area. Regional-minus-native Dice is +0.0189, +0.0248, +0.0141, and +0.0691, but every sequence-bootstrap CI includes zero; in the `≥15%` bin the effect falls to +0.0021 after removing seq8. Regional-minus-generic in that largest bin is +0.0763, driven strongly by seq8; it is only +0.0099 without seq8 and remains exploratory/post hoc. Do not turn this into a size-conditioned inference rule. Reproducible report, code, all per-sequence effects and GT-mask hashes: Drive `outputs/experiment/tome_region_identification_v1/run155_area_strata_20261113/`.

Protocol, raw/visible masks, per-layer receipts, two exact result files and failure-attempt receipts: Drive `outputs/experiment/seq5_failure_window_token_utility_v1/`. Completed attempts `attempt_05` and `attempt_06`; their `results.json` SHA-256 is `a83667cd3c4980628194b80f3041d5ee135c1495c62620ea611d3ab2d914fe5a`. Concise analysis: `analysis.md`.
# Statistical-calibration ToMe — attempt stopped for protocol deviation

The Seq5 diagnosis does not support a Seq5-specific token fix: prolonged loss is shared with native SAM2, and the object-presence rule suppresses both a partly aligned mask and a later severe false positive. Do not bypass that rule or tune a merge heuristic on Seq5.

The next general ToMe candidate transcribes the fixed correction in [Statistically Calibrated Scaling for Token Merging in Transformers (ICML 2026)](https://proceedings.mlr.press/v306/zhou26bj.html): for spatial-key retention ratio `r`, scale merged spatial keys and values by `sqrt(r)`, and scale the existing `log(cluster size)` bias by `1-sqrt(r)`. This addresses aggregation-induced attention distortion; it is not a Seq5 reappearance solution. SAM2 object pointers remain untouched. The paper's derivation does not cover SAM2's appended pointer tokens, so this is a testable adaptation, not a guarantee.

| Status | Evidence |
|---|---|
| Implementation | Wrapper reuses the pinned ToMe merge and mass operations; paper-derived correction is in `modeling/stat_calibrated_tome.py`. |
| Focused checks | 4 passed: equation scaling, unchanged pointers and inputs, exact no-merge identity, invalid-count rejection. |
| Real-frame gate | Repeatability passed twice, but the gate omitted the required calibrated/unscaled assignment parity. It is insufficient to authorize efficacy. |
| Full-cohort Dice | Attempt stopped; PID 65755 terminated. No valid five-arm efficacy result. Saved calibrated/unscaled regional plans differ on 117/120 completed targets across seq2–4. |

The frozen five-arm design is `protocols/stat_calibrated_tome_gate_v1.json`. Its exact cross-calibration assignment control was not implemented. The run is stopped and preserved; do not resume it or treat its partial Dice as a valid controlled comparison.

## Statistical-calibration ToMe candidate — status 2026-11-13

The original real-frame gate was repeatable on seq2/seq3: all ten per-sequence condition records were byte-identical across processes. Those checks did not cover the promised calibrated/unscaled assignment equality. The gate therefore does not establish the required experimental control.

The attempt is stopped after verifying different calibrated/unscaled regional merge-count plans on 117/120 completed targets across seq2–4. The runner reset the calibrated regional plan and regenerated decisions on its own propagated bank; it did not replay exact token assignments from the uncalibrated counterpart. PID 65755 was terminated and confirmed absent. Preserve all artifacts on Drive `outputs/experiment/stat_calibrated_tome_polypgen_v1/full_attempt01/`; no valid five-arm efficacy result is available.

**Audit correction:** the earlier partial audit passed its listed within-setting checks but missed this calibration-specific invariant. The corrected checker rejects the saved attempt. Gate and runner guards now reject missing cross-arm assignment evidence; 13 focused regression checks passed, and both CLI entry points rejected before inference as expected. Exact assignment replay remains unimplemented, so new efficacy runs are blocked by code. Evidence: Drive `full_attempt01/audits/calibration_protocol_deviation_20261113.json` and `corrected_calibration_control_audit_20261113.json`; checker: [`audit_stat_calibrated_tome_completed_sequences.py`](../scripts/audit_stat_calibrated_tome_completed_sequences.py).

This method applies the fixed scaling in [Statistically Calibrated Scaling for Token Merging in Transformers (ICML 2026)](https://proceedings.mlr.press/v306/zhou26bj.html) to SAM2's compressed spatial K/V while leaving pointers unchanged. Its source derivation does not cover SAM2's mixed spatial/pointer cross-attention. This stopped attempt establishes no efficacy result. Repair the controlled comparison and pass fresh gates before any further inference; replaying another rollout's assignments would be a diagnostic dependency, not a standalone adaptive-token method.

## Signed feature attribution vs. local token utility — exploratory diagnostic (2026-11-13)

This directly revisits the original hypothesis: a poor whole-frame prediction may still contain useful local memory tokens. The four targets were selected post hoc from the saved tile-removal study and compared with Captum Input×Gradient on each same source→target pair. Positive deletion utility means removing that tile reduced target Dice.

| Sequence | Source → target | Source Dice | Target native Dice | Mask-support ρ | Presence ρ | Helpful / harmful / zero tiles |
|---|---:|---:|---:|---:|---:|---:|
| seq2 | 4 → 6 | 0.926676 | 0.935537 | +0.095 | +0.261 | 1 / 9 / 6 |
| seq3 | 1 → 3 | 0.950152 | 0.856208 | −0.306 | +0.212 | 3 / 13 / 0 |
| seq17 | 17 → 19 | 0.003777 | 0.903168 | −0.184 | −0.138 | 7 / 8 / 1 |
| seq22 | 41 → 45 | 0.000000 | 0.942278 | −0.305 | −0.204 | 5 / 2 / 9 |

**Interpretation:** the poor source frames in seq17 and seq22 each contain both locally helpful and harmful tiles under these fixed-read Dice interventions. That is consistent with the original intuition that frame quality does not determine the utility of every token. However, the signed gradient correlations are weak and inconsistent, so these two confidence/logit gradients do not identify useful tiles reliably. This is not a learned or inference-time selector and it establishes no full-video Dice gain.

**Integrity:** frozen source/checkpoint/manifest/utility hashes verified; 4/4 archived native predictions replayed exactly in an independent process; four attention layers, source IDs, memory/pointer invariants and finite attribution maps verified. Thirty-two feature-scaling finite differences were reported as arithmetic diagnostics; feature scaling is not token deletion because deletion also changes attention normalization.

Detailed table and machine-readable audit: Drive `outputs/experiment/tome_region_identification_v1/signed_memory_feature_utility_overlap_v1/attempt_02/results.md` and `audit.json`. Re-run audit with `python scripts/audit_signed_memory_feature_utility_overlap.py --output-dir <attempt_02>`.

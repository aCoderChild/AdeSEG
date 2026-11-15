# Active research goal — revised 2026-11-13

## Original hypothesis — governs all candidate decisions

A past frame with a poor segmentation prediction can still contain memory-token
representations useful for a later query. The aim is to exploit useful token
subsets under SAM2's native temporal schedule, without judging whole frames as
good or bad, and establish a meaningful reproducible full-video Dice improvement.
ToMe, region labels, calibration and query-conditioned matching are optional
mechanisms. Do not substitute making one mechanism succeed for this research aim.
Distinguish conditional source-local utility, inference-time predictability, and
full-video efficacy. GT-derived source quality and outcome-selected token extrema
are retrospective diagnostic tools; they cannot be used by a deployed selector.

## Upstream implementation rule

For each new candidate, first clone and pin the official implementation, then
integrate its actual operators/modules where they fit SAM2. Do not recreate a
paper method from prose. Make only necessary compatibility or bug fixes, record
the upstream commit and every local patch, and add focused regression checks
before real-data inference. If the upstream method depends on incompatible
weights, frame selection, gates, or changed SAM2 components, record the source
as screened and do not force a transplant or call it an integration.

**Verified original-hypothesis evidence (2026-11-13).** Existing source-local ablations already contain local support: seq17 source17 has Dice0.003777, yet removing its tile14 worsens target19 by0.005281 and removing tile10 improves it by0.010921. Seq22 source41 has Dice0 and mixed much smaller effects on target45. All59 targets/1,180 arm receipts were verified,42 mask Dice values were independently reproduced, and these examples' source/target hashes exactly match native Run143. Coverage is sparse:38 GT-present source/target pairs, with only2 of the6 lowest-quality source pairs having nonzero visible effects. GT-selected extremes are diagnostic only. Full results: Drive `within_frame_token_utility_v1/original_hypothesis_reanalysis_20261113/`. Do not revive the stopped coarse-tile selector: its best-of16 oracle gain is only0.002184 on sampled immediate targets. The remaining research need is a GT-free token-level mechanism with reliable full-video benefit.

## Current candidate: cross-frame Video-ToMe value sharing — durable cohort active

**Upstream implementation.** Clone `external/video-token-merging` is pinned to
`267fb4915c0dcb55f87f555fd72dbb6df4a618b7`; the unmodified upstream
`tome/merge.py` is SHA256 `b70f6d48…22af57`. The integration imports the
author's bipartite matcher, weighted merge and unmerge instead of reimplementing
ToMe. Local adapter: `modeling/spatiotemporal_tome_value_exchange.py`.

**Method.** At native SAM2 memory-attention SDPA, match projected memory tokens
inside each 4×4 spatial tile across selected past maps using the mean-head key
metric. Apply the upstream weighted merge to projected V, then unmerge to every
original slot. The matched control does this separately in each frame. SAM2
weights, query, K, positions, pointers, bank writes, frame schedule and decoder
stay unchanged. This is temporal value sharing, not token-count reduction;
there is no efficiency claim.

**Engineering status.** Two fresh v4 full-propagation runs passed on seq6/8,
covering 138 targets and 414 condition-frame outputs each. Both exactly match
native in the no-op arm and produce byte-identical results (SHA256
`b2ab812f8cb751d27902f8a58b7877e3361539ac7b550daa1cc20bb34a85802e`). The
runner verifies all four attention layers, memory layout, budgets, pointers and
frozen parameters. The corrected focused suite has 17 passing tests. The
development effects were tiny and mixed: seq6 cross-frame Dice 0.864724 vs
native 0.864721; seq8 0.279219 vs 0.279446.

**Recovery engineering passed.** Hash-bound condition checkpoints, preservation of
incomplete folders, a process lock and resume orchestration are implemented.
Four focused tests passed. An actual SIGKILL and independent-process resume on
four seq6 targets reproduced 12 frame records and 48 attention calls; the durable
cohort main separately reproduced 42 frame records and 168 attention calls on
seq3 after SIGKILL/resume. A five-frame real-image fixture with two synthetic IDs
passed object-bank isolation and no-op parity. This fixture does not establish
performance on anatomical multi-object videos. Gate receipts: Drive
`durable_real_replay_gate_v1/comparison.json`,
`durable_main_real_gate_v1/comparison.json`, and
`spatiotemporal_multiobject_gate_v2/attempt01/results.json`.

**Interruption (2026-11-13).** PID55492 is absent with no completion/failure receipt; last saved state seq14/local144/248. Attempt01 outputs are preserved. Eleven complete score triplets are descriptive only. Detailed raw attention calls and sequence audit outcomes were held in memory, so saved scores cannot authorize a certified full-cohort resume. The original process is no longer running; the completed recovery gates and new durable run are recorded below. Receipt: Drive `cohort_v1/attempt01/interruption.json`.

**Durable v2 cohort active (2026-11-13 10:12 UTC).** PID615 runs seq2/local target19/59 in a detached process. Protocol `protocols/spatiotemporal_tome_value_exchange_cohort_durable_v2.json` freezes the original 21-sequence method, data, metrics and stopping screen, and binds passed real-frame durable-main replay and synthetic two-ID bank-isolation receipts. Output: Drive `cohort_durable_v2/attempt01/`; launch and postrun audit command receipts are in its parent. Raw attention calls and hash-bound condition receipts persist as each condition completes. The original interrupted 11-sequence subset is exploratory only; all 21 sequences are being rerun for a complete auditable cohort.

**Interrupted v1 cohort.** Protocol `protocols/spatiotemporal_tome_value_exchange_cohort_v1.json`
is frozen and preflight passed against 27 source hashes, 4,330 input hashes,
the pinned SAM2 checkpoint/runtime, and all 21 native references. The interrupted attempt planned 21 eligible sequences (2,012 target frames; 6,036 outputs across
three conditions) on the Drive mount. It compares native/no-op, within-frame
ToMe, and cross-frame ToMe. The full run is exploratory because seq6/8 informed
method design, not untouched confirmation. Its screen requires a meaningful
Dice gain, uncertainty support and no worsening of the predeclared reliability
checks. If it fails, stop this candidate without tuning on these sequences.

**Prior evidence.** A different 21-sequence within-frame V-pooling study scored
native 0.710224, dependency-aware treatment 0.710984, and feature-only control
0.711333; its frozen screen failed. The current run specifically tests sharing
across past frames. It makes no token-count or runtime-reduction claim because
ToMe unmerges to all native slots. Artifacts and live progress are in Drive at
`outputs/experiment/spatiotemporal_tome_value_exchange_v1/cohort_v1/`.

## Previous candidate: calibrated ToMe — stopped for a control defect

**Question.** Does the fixed statistical correction for ToMe aggregation reduce attention distortion when compressing SAM2 spatial memory, and does regional matching beat a matched generic merge?

**Method.** Training-free calibration of merged spatial K/V using the ICML 2026 paper’s stated scaling: multiply K/V by `sqrt(r)` and reduce log-cluster-mass bias by `1-sqrt(r)`. Pointer K/V, native SAM2 weights, memory writes, decoder, prompts and frame schedule stay unchanged. This adapts a self-attention result to SAM2’s mixed spatial/pointer cross-attention; it is a hypothesis, not a proven segmentation method or Seq5 remedy.

**Reliability correction.** The original gate was repeatable across two processes but incomplete: it checked generic/regional quotas within each calibration setting and omitted the protocol's exact calibrated/unscaled assignment and budget comparison. Saved plans differ on 117/120 targets across seq2–4, and exact assignment maps were not persisted. The earlier gate cannot authorize efficacy. The original protocol remains frozen; no criteria have been relaxed.

**Full evaluation.** PID 65755 was terminated after verifying this defect; the process is absent. Preserve all outputs under Drive `stat_calibrated_tome_polypgen_v1/full_attempt01/` and do not resume this attempt. No valid five-arm efficacy result is available. The original partial integrity receipt covers only its listed within-setting checks. Corrected rejection evidence: `audits/calibration_protocol_deviation_20261113.json` and `audits/corrected_calibration_control_audit_20261113.json`.

**Repair status.** The gate now compares both calibrated/unscaled pairs and requires exact assignment fingerprints. The full runner rejects old gates lacking that verification. The method declares frozen-assignment replay unsupported, so fresh calibration gates fail before model loading. Thirteen focused regression checks passed, and both CLI guards were exercised with expected rejection receipts on Drive. If calibration is reconsidered, exact assignment replay using the pinned author ToMe operation remains required by the original protocol, followed by fresh independent engineering gates. Do not prioritize this repair without same-input real-feature evidence and justification of the source assumptions. Do not replace exact assignments with count-only matching or reinterpret this invalid attempt as a total-system efficacy result.

**Interpretation limit.** This candidate tests whether correcting the attention effect of merged tokens helps segmentation. It does not directly target Seq5’s prolonged object-presence failure; prior Seq5 runs show that loss is shared with native SAM2. Do not claim recovery or a Dice gain before the frozen full-cohort analysis.

**Adaptive-token follow-up audit.** The native frame schedule can coexist with current-frame-dependent token decisions, but this broad idea overlaps existing work here. Run170 already used first-read attention/top-k and anchors; the Seq5 MCTF diagnostic already used current-query mean attention to guide value merging. Neither supports relaunching the same mechanism. A future query-conditioned matcher needs a precise difference and a failure-based justification before implementation; see `protocols/CMSA_MULTISCALE_TOME_REVIEW_20261113.md`. No efficacy process is active after the calibration-control defect was found.

**Design caution.** Exact replay of assignments from a different uncalibrated rollout would be a controlled diagnostic, with a dependency on that reference rollout. It cannot by itself become the claimed independent deployable adaptive-token method. Preserve the failed v1 comparison; establish the isolated same-input calibration effect and exact assignment handling before proposing a fresh standalone full-propagation design. Do not silently reinterpret differing propagated plans as satisfying the original frozen invariant.

**Calibration applicability decision (2026-11-13).** Two fresh processes exactly reproduced four synthetic same-compression fixtures (result SHA256 `9959ebf770aecb9cc2b82cfb3b35704571c5817f2178fab1c8523f7496208d79`). Identical-token ordinary mass-aware ToMe is exact; calibration gives relative squared output error0.085786 without pointers. The scalar equations are correct, but normalized self-attention assumptions are not established for projected/post-RoPE SAM2 K/V with dense Q and unscaled pointers. This is no Dice result and no proof of PolypGen harm. Full attempt remains stopped; do not launch efficacy from this evidence. Archived source, hashes, table and repeat receipt: Drive `stat_calibrated_tome_polypgen_v1/source_applicability_20261113/report.md`. Prioritize the original useful-subset hypothesis, not making calibration succeed.


**Shared-input measurement implementation (2026-11-13):** the existing native-returning attention observer now optionally evaluates calibration on the exact same merged tensors as ordinary ToMe (one matcher call). It checks merged-input hashes, unchanged pointers and finite outputs, and returns the native result. Fifteen focused tests passed, including the identical-token counterexample and rejection of the original invalid controls. Real-feature replay subsequently completed (see the result below); this observer does not repair full-run assignment replay. Source snapshots and test receipt: Drive `stat_calibrated_tome_polypgen_v1/shared_merge_observer_engineering_20261113/`.

**Real-feature applicability check complete (2026-11-13):** two independent native-returning replays exactly match (results SHA256 `b70676cb73d6403f1150a7d3cade9a981885a8fb980e6808d227398535de49fd`). On six seq2/seq3 targets × four layers, calibration worsens all48 same-merge attention comparisons. Mean relative squared error: regional0.005137→0.217725; generic0.007004→0.222524. These are not Dice scores. Native masks/presence, Q/K/V, pointers, banks and weights stay unchanged; no GT or shadow propagation. Sixteen focused tests passed; verifier negative checks reject missing shared inputs and incorrect completion hashes. Stop this calibration integration under the frozen diagnostic rule; no full efficacy or tuning follows. Protocol, source snapshots, per-read records and report: Drive `stat_calibrated_tome_polypgen_v1/shared_merge_real_feature_diagnostic_v1/`.

## Pointer-conditioned ToMe cohort — complete; frozen screen failed

**Method.** Training-free local ToMe matching appends each spatial memory token’s cosine compatibility with its own frame’s object pointer to the normalized mean-head post-RoPE key metric. It uses the pinned official ToMe matcher and weighted merge. The generic control has identical per-target/window quotas. SAM2 weights, native writes, Q/K/V attention, current-frame queries, prompt/frame schedule, decoder, presence handling and pointer tokens remain unchanged.

**Engineering gate.** Two independent real-frame seq2/seq3 replays passed native parity, pointer/source-frame alignment, unchanged weights and source banks, temporal schedule restoration, pointer-tail preservation, four-layer coverage, and matched generic quotas. The outputs were byte-identical; 11 focused tests passed. Gate receipt: `tome_region_identification_v1/pointer_conditioned_tome_v1/attempt03/`; protocol and source pins are frozen in `protocols/pointer_conditioned_tome_efficacy_v1.json`.

**Full run.** All 21 sequences completed. Equal-sequence present-frame Dice: native 0.710224, pointer-conditioned ToMe 0.710213, generic ToMe 0.710073. Pointer−native is −0.000011 (95% sequence-bootstrap CI [−0.007212,+0.006357], Holm p=0.9105; 7/21 wins); pointer−generic is +0.000140 with interval spanning zero. The frozen meaningful-improvement screen failed. Full-cohort integrity audit passed; artifacts are on Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/`. No process remains active.

**Interim evidence, seq2–seq11 (9/21 complete).** Equal-sequence present-frame Dice is native 0.709762, pointer-conditioned ToMe 0.711951, matched generic 0.711849, and saved YOLO 0.646376. Pointer-vs-native is +0.002189, pointer-vs-generic +0.000102; it wins 3/9 sequences vs YOLO. False alarms on 174 empty targets: native 103, pointer 102, generic 102, YOLO 9. Sensitivity: pointer-native is +0.003285 without seq8 but −0.001721 without short seq10 (only 6 present targets); the apparent partial gain is not robust. This subset is descriptive, not the 21-sequence result. Exact frame, GT, native reference, and YOLO-mask checks passed for 786 targets. Audits: `full_attempt01/audits/matched_yolo_partial_completed_sequences_20261113.md` and `seq11_all_arms_budget_integrity_audit_20261113.md`.

Seq11: native/pointer/generic Dice 0.280751/0.280258/0.280272; generic and pointer merge quotas agree across all 1,800 target-layer checks. All arms have 90/90 empty-frame false alarms. This is a shared low-baseline case, not evidence of pointer-specific benefit.

Seq12: native/pointer/generic Dice 0.944438/0.940333/0.939439; pointer is −0.004104 vs native and +0.000894 vs generic. All 249 frames are GT-present. Native archived hashes match; metrics recompute; parameter digest and four-layer/mass invariants pass. Frozen-runner generic window-budget checks: 1,763,328. Read-only audit: `full_attempt01/audits/seq12_all_arms_integrity_audit_20261113.md`. Descriptive only.

Seq14: native/pointer/generic Dice 0.839942/0.839046/0.837983; pointer is −0.000896 vs native and +0.001063 vs generic. Boundary F is 0.641242/0.636046/0.637286; empty-frame false alarms are 32/49 in every arm. Saved-output arithmetic, frame/presence alignment, parameter/reference hashes, four attention-layer coverage and matched-budget receipt passed. Runtime was 189.5/560.3/590.0 seconds for native/pointer/generic. Audit: `full_attempt01/audits/seq14_all_arms_integrity_audit_20261113_0423/`. Descriptive only.

**Interim aggregate through seq21 (19/21 completed sequences).** Equal-sequence present-frame Dice: native 0.731197, pointer ToMe 0.734108, generic ToMe 0.733945. Pointer-minus-native is +0.002911 with 7 wins, 1 tie and 11 losses; pointer-minus-generic is +0.000163 with 8 wins, 1 tie and 10 losses. False alarms are 186/307 vs native 192/307. All 19 saved-output audits passed. Seq21's false-alarm reduction is shared by pointer and generic ToMe. This interim result is not the frozen 21-sequence result or a significance test. Full table: `full_attempt01/audits/completed_seq2_21_interim_audit_20261113_0454/` on Drive.

Seq5’s prolonged object loss remains shared across native and ToMe conditions; do not tune another Seq5-specific merge rule. The fixed-budget wider-history study (which keeps the conditioning map dense) raises Seq5 mean Dice to 0.5472 but leaves the 26-frame and 58-frame zero-Dice intervals unchanged and shows no regional advantage over generic matching. The cohort objective remains unproven.

## Completed diagnostic — attention-aware MCTF value fusion on Seq5

The two-process real-frame engineering gate passed. The frozen post-hoc Seq5 propagation diagnostic completed all 249 targets in each treatment, with 996 memory-attention layer calls per treatment, unchanged SAM2 weights, restored temporal slots and the native six-frame history limit. Exact result SHA256: `0a265b67f0b61401b86de2a26f9761425b7adaf9dd4e1ac56533965bc505255c`; protocol SHA256: `d46fa90e7e0b13ee4f934d139aaaf49636b7693499e1c2950ac3aba3375ce2a0`.

| Seq5 condition | Present-frame Dice | Longest zero-Dice run | Missed returns | Empty-frame false alarms |
|---|---:|---:|---:|---:|
| Native SAM2 | 0.541987 | 58 frames | 1/4 | 1/51 |
| Generic ToMe value | 0.541792 | 58 | 1/4 | 1/51 |
| MCTF uniform importance | 0.541966 | 58 | 1/4 | 1/51 |
| MCTF attention-aware | 0.541826 | 58 | 1/4 | 1/51 |

Attention-aware MCTF is −0.000161 Dice vs native and −0.000141 vs its uniform control; it changes none of the 85 failed frames or return outcomes. This candidate does not resolve the observed prolonged loss. No full-cohort run is justified. Continue by auditing the native presence-score/raw-mask suppression pathway and its false-positive tradeoff; do not bypass it or label this ToMe variant a fix. The diagnostic report and complete artifacts are on Drive `tome_region_identification_v1/mctf_attention_memory_v1/seq5_diagnostic_attempt06/`. The project’s primary goal—reliable, meaningful full-cohort Dice improvement over native SAM2—remains unmet.

## Current status — 2026-11-13, verified after the Seq5 diagnostic

Pointer-conditioned ToMe did not improve the full cohort and is stopped under its frozen screen. The fixed-initial-prompt-pointer follow-up also failed on its development sequence: Seq5 Dice 0.541171 versus native 0.541987 and generic matched-budget ToMe 0.541216. All four full-video arms shared the same zero-Dice frames (92; 157–182; 184–241), the same 1/4 missed returns and 1/51 empty-frame false alarms. Audit: Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/prompt_anchor_seq5_diagnostic/diagnostic_v1_attempt01/posthoc_integrity_audit.json`. No candidate or cohort process is active. The fixed-budget wider-history arm recovers only frame92; it leaves both long miss intervals intact, so preserving a dense prompt map and adding history are insufficient Seq5 fixes. The raw-mask/presence path is behaving as designed: the negative score suppresses a partly aligned mask at frame157 and an invalid, 10.9×-area mask at frame184. Do not bypass it. No ToMe implementation defect has been identified.

The wider benchmark audit confirms that the historical YOLO all-frame 0.748099 is not comparable to SAM2's post-prompt GT-present metric. On identical saved post-prompt present frames, YOLO is 0.718842 and has 17/323 empty-frame false alarms. All 2,012 saved masks and frame identities were rechecked; generating checkpoint, detector weights, preprocessing, prompt cadence, and runtime remain undocumented. Saved-output arithmetic is verified; the historical inference recipe is not.

**Fixed-bank utility follow-up is complete.** The 21-sequence source-group study found near-zero mean effects for uniform tile removal (+0.000455 Dice; 95% sequence-bootstrap CI [−0.000071, +0.001384]); its best-of-16 oracle was +0.002184 and outcome-selected, not a deployable rule. Run171's rejection-based selector ranked poorly (rejected-minus-random −0.000891; Holm *p*=0.0179), and Run173's regional predictor had worse MSE than the zero and generic controls. None establishes useful predictable token utility or an efficacy gain. These studies sampled fixed targets; they did not intervene on every token group at Seq5 frames 157 or 184.

**Seq5 implementation decision:** no code correction or safe token-level recovery rule is supported. All tested ToMe/value variants preserve the same presence failures; frame184's raw decoder mask is entirely wrong and 10.9× the GT area, while frame157's partly aligned raw mask is suppressed. A presence bypass is unsafe, and a new merge metric is not justified by the existing evidence. The bounded fixed-bank token-group diagnostic at frames157 and184 is complete: none of 160 single-tile interventions restored the visible mask at157 or fixed localization at184. Stop Seq5-specific ToMe tuning; the primary full-cohort Dice goal remains unmet. :codex-annotation{index="1"}



The earlier seq2–seq10 interim snapshot is superseded by the matched seq2–seq11 comparison above; retain its audit files only as run history. Do not use it as the current partial-cohort estimate.

The frame-wise YOLO-box masks were audited offline: all 2,012 saved predictions and frame identities reproduce. The reported 0.748099 is an all-frame average over 21 sequences. Under the matched post-prompt, GT-present, equal-sequence protocol it is 0.718842 over 1,689 frames, with 17/323 false alarms on GT-empty frames. The generating checkpoint, detector weights, box policy, preprocessing, prompt cadence and runtime remain unknown. The 0.74 figure is therefore a system-level reference with a different denominator and incomplete provenance, not a matched benchmark.

On those same matched present-frame outputs, the completed 12-map regional ToMe run scores 0.731901 (generic ToMe 0.731557; native 0.710224) versus saved YOLO masks at 0.718842. This is descriptive, not a robust win: regional ToMe has only 7/21 sequence wins over YOLO and 189/323 empty-frame false alarms versus YOLO's 17/323. Excluding seq8, regional is 0.726455, YOLO 0.710510, native 0.731763. The 21-sequence point mean alone hides large opposing sequence effects, and the original YOLO inference recipe is unverified.

Superseded next-step note: the pointer-conditioned cohort is complete and its frozen screen failed. Do not restart ATC, PiToMe, MPM, or MCTF, and do not implement a Seq5-only presence bypass. A new token-level method requires causal fixed-bank evidence that it improves raw localization while retaining correct rejection at frame184; otherwise stop Seq5-specific recovery work.

Initial source triage is complete. DAM4SAM and SAM2Long address occlusion/reappearance, but depend on frame/mask/pathway management, outside this token-level fixed-schedule study. Cutie provides object-level VOS context but requires a separately trained architecture. Efficient-SAM2 has a faithful top-k memory-token retrieval path, but it is a hard-pruning efficiency method and conflicts with the no-gating constraint; its paper does not establish a Dice gain. Detailed sources and rationale: Drive `tome_region_identification_v1/mctf_attention_memory_v1/seq5_diagnostic_attempt06/strategy_review.md`.

**Efficient-SAM2 SMR source audit (2026-11-13):** corrected trace confirms the author checkout applies fixed top-k masks to spatial memory K/V in `build_model/window_bypass.py`, with rotary positions indexed from the original grid. This is hard token pruning for efficiency, not ToMe and not a demonstrated Dice-recovery method; the paper reports 1.68× speedup with a 1.0% SA-V accuracy drop. It conflicts with the standing no-gating constraint, so no local adaptation or run was launched. Audit: [`EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md`](EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md).

**Run155 object-size audit (2026-11-13):** post-hoc analysis of all 1,689 saved GT-present frames found regional predicted/GT area medians of 0.998×, 0.957×, 0.988× and 0.960× across the four increasing GT-size bins; therefore the proposed large-over/small-under pattern is not supported. Regional-minus-native large-bin Dice is +0.0691 (sequence-bootstrap 95% CI [−0.0030,+0.2060]) and falls to +0.0021 without seq8. This is descriptive, post hoc evidence, not a new efficacy result or permission for size-based routing. Drive artifact: `outputs/experiment/tome_region_identification_v1/run155_area_strata_20261113/`.

## Last completed cohort — dependency-aware V-only ToMe; predefined screen failed

The dependency-aware V-only ToMe full-video run completed all21 eligible sequences (seq2–23 excluding seq7). Exact result hash: `03e7a559508e5b6e3f5eb5216bd9455c89fce1e9776d881d2f81ec206e7985f3`. Arithmetic/provenance audit passed: 63 sequence summaries and42 treatment frame files verified; final source pins passed. This audit does not independently replay all model inference.

| Condition | Equal-sequence present-frame Dice | Boundary F | False alarms | Missed returns |
|---|---:|---:|---:|---:|
| Native SAM2 | 0.710224 | 0.616994 | 194/323 | 11/51 |
| Dependency-aware V ToMe | 0.710984 | 0.618188 | 194/323 | 11/51 |
| Feature-only V control | 0.711333 | 0.618011 | 194/323 | 11/51 |

Dependency-aware V gains only+0.000759 Dice vs native (95% sequence-bootstrap CI [+0.000015,+0.001939], paired Wilcoxon p=.179, Holm p=.537; 12/21 wins) and loses−0.000349 vs matched feature-only V control (CI [−0.001087,+0.000052], Holm p=.911). False alarms and missed returns do not change. The predefined meaningful-gain/statistical/consistency screen failed; no test-sequence evaluation followed. Do not tune or relaunch this candidate.

Seq5's prolonged loss remains shared by all three arms: the same85 GT-present zero-Dice frames (92,157–182,184–241); native/value/control Dice=.541987/.541849/.541792 and one of four returns is missed in each. This agrees with the earlier conclusion that Seq5 is primarily a native object-presence/reacquisition failure, not a substantial ToMe-specific degradation. :codex-annotation{index="1"}

A read-only seq5 native replay completed with exact parity at249/249 frames, 996 attention calls, unchanged weights and restored temporal slots. At frame157 raw mask Dice=.574 is rejected by object score−.883; at frame184 raw Dice=0 and mask area is10.9× GT, so score−1.916 prevents a false positive; at frame242 raw Dice=.943 with score+.476 is retained. Globally bypassing presence logic is therefore unsafe. Attempt01 caught an observer dict/tuple bug before any capture; corrected attempt02 is preserved. Attempt03 completed at exact parity249/249, with996 attention-layer reads. Object queries attend GT-object coordinates in conditioning and some historical maps, but the historical masks at frames178–182 had negative presence scores and were encoded as no-object. At frame242, object queries show stronger latest-map object-cell selectivity than background queries in layer1. This is descriptive; it does not prove the attended values are object-semantic or causally useful. Do not bypass presence logic or launch another ToMe variant from this one-sequence finding. Full details: Drive `dependency_value_intervention_v1/seq5_posthoc_analysis/seq5_raw_probe03_analysis.md`.

Full protocol and result: Drive `dependency_value_intervention_v1/full_protocol.json` and `full_attempt01/results.json`; audit `full_attempt01/audits/final_arithmetic.json`. Probe protocol and source snapshot: Drive `dependency_value_intervention_v1/seq5_diagnostic_attempt02/`.

## Previous execution — MPM full cohort complete; screen failed

PID49697 exited after all 21 sequences and both treatment arms completed. MPM Dice 0.710453 versus native 0.710224 and matched-count ToMe V-only 0.712917. MPM−native +0.000229, CI95 [−0.000085, +0.000638], Holm p=1; MPM−control −0.002464. Frozen continuation screen failed; do not retune or relaunch this candidate. All 63 summaries and planned comparison recomputed exactly; 42 treatment runs frame-aligned, weights unchanged, temporal slots restored, all 60 source pins match. Completion/result hash and audit on Drive `outputs/experiment/mpm_value_read_polypgen_v1/full_attempt01/`. Full objective remains unmet. Saved-trajectory diagnosis completed: both arms change 1,677/2,012 prediction hashes but recover zero native zero-Dice frames; extent/coverage changes can help or hurt. ToMe seq8/frame56 improves recall; seq14/frame165 loses recall. Receipt saved under full_attempt01/audits/saved_trajectory_diagnosis.json. Native-returning pair-selectivity diagnostic completed and reproduced byte-identically in two processes: four targets × four layers, 64 sampled queries/layer; nonzero MPM averaging distortion, pair identity/direct SDPA errors ≤1.22e−14. This establishes arithmetic/native integrity, not token utility. Summary on Drive post_value_source_review_20261112/value_selectivity_diagnostic_v1/summary.md. Outcome-case native-bank coverage and two-process replay are now complete for seq8/56–57 and seq14/164–165. MPM has lower shadow output distortion on all four native-bank reads, but lower full-cohort Dice than ToMe; distortion reduction alone does not justify a new candidate. See Drive value_selectivity_diagnostic_v1/outcome_summary.md. Primary-source triage is recorded in post_value_source_review_20261112/segmentation_source_triage.md: StructSAM targets encoder efficiency with minor accuracy drops, not a demonstrated memory-attention Dice gain; deprecated G2TM source redirects to its current author repository. Next: inspect distinct utility mechanisms and official implementations within the frozen constraints; no new efficacy launch justified yet.

Dependency-aware ToMe diagnostic: DTMFormer official source inspected and pinned; new incoming-cross-attention-signature adaptation explicitly differs from DTMFormer. Fixed four-target native-returning run attempt01 passed integrity, but held-out error worsens on all four targets; independent attempt02 replay completed with byte-identical results (receipt independent_replay.json). Do not tune settings or launch efficacy from these diagnostics. Drive `post_value_source_review_20261112/dependency_matching_v1/`.

Native failure decomposition completed from saved outputs:187 empty +7 disjoint of194 zero-overlap GT-present frames; all187 empties have nonpositive presence logits and cached−1024 masks (10 bank SHA256s verified). MPM/ToMe move logits in both directions but recover0/187 suppressed frames. See Drive dice_deficit_audit/{summary.md,cached_presence_results.json,presence_margin_comparison.json}. Next mechanism must test segmentation/presence effects while retaining native suppression; distortion-only improvement is insufficient. No completed candidate may be retuned from these post-hoc subsets.

Actual dependency-aware V-only gate now passed4targets×4layers, two-process byte-identical replay; no-op/source invariants verified, no GT efficacy read or propagation. Single-object gate only; full goal unmet. Next: separately freeze a bounded segmentation-effect/propagation design if justified; no tuning from fixture masks. Drive dependency_value_intervention_v1/independent_replay.json.

Dependency-aware V propagation attempt01 completed14/14 targets in all three arms, with native/noop predictions and states exact and schedule/budget invariants passed. Independent propagation02 completed with byte-identical results. Combined receipt saved; full protocol frozen and launched as Current execution above. Synthetic tensor batch-isolation check passed; persistent-ID multi-object video integration remains unverified.

### Previous completed candidate (do not restart)

Native-schedule value consolidation completed all21 sequences on 2026-11-12. PID95838 exited; completion receipt and result hash verified. Final integrity checks passed56 source pins and all42 treatment summaries. Dice: native0.710224, value0.711331, joint0.710318. Value−native+0.001106, Holm p1.0,9/1/11 wins/ties/losses; value−joint+0.001013, CI spans zero. Frozen efficacy screen failed. Stop this candidate; do not tune or relaunch from this result. Final tables are in docs/REGION_TOKENS_RESULTS.md and Drive native_value_consolidation_polypgen_v1/full_attempt01/final_report.md. The full research objective remains unmet. Next work is bounded diagnosis/research review, not another automatic efficacy sweep.

Saved-trajectory follow-up: no zero-Dice frame recovered in the full cohort; withoutseq14 the gain is only0.000275. Seq14 has both recall recovery(frame185) and recall degradation(frame165). Diagnosis saved on Drive; no new inference launched. A next candidate must address a distinct supported failure mechanism, not retune value averaging.

Source follow-up saved on Drive `tome_region_identification_v1/post_value_source_review_20261112/review.md`: ALGM repeats a threshold-based encoder route; calibrated-scaling official software repository is empty (zero git refs); SCMerge paper has no verified author implementation in inspected HTML. These are references, not implemented SAM2 candidates. No new efficacy run launched.

MPM follow-up: official source pinned and synthetic/offline raw-memory probes saved on Drive. Reciprocal matching still merges pairs with cosine as low as0.603 in inspected windows, so it does not establish semantic compatibility. No SAM2 integration or efficacy run; source review must precede a distinct frozen diagnostic.

Matched-count MPM/ToMe raw-memory comparison repeated exactly in two fresh processes: MPM RSE0.0128–0.0145 vs ToMe0.0232–0.0284; maximum cluster2 vs6–7. This motivates only an actual-Q/K/V engineering diagnostic, not efficacy. Script/results/source hashes on Drive post_value_source_review_20261112.

MPM actual-Q/K/V shadow gate attempt02 completed4targets×4layers: native masks/presence exact, banks/weights unchanged, original Q/K retained, matched per-window logical counts. Mean attention-output RSE MPM0.00025349 vs ToMe0.00086259; descriptive engineering targets only, no Dice claim. Independent fresh-process replay now passed with byte-identical full results and identical two-shadow records; receipt shadow_gate_independent_receipt.json. Gate is complete for engineering distortion only. v1 stale inherited hash was rejected before inference; v2 pins verified full-run source and preserves refusal receipt. Drive post_value_source_review_20261112/shadow_gate_attempt02. No efficacy process active.

MPM intervention engineering attempt01 passed4targets×4layers using official matcher/value merge-map aggregation and existing native gate: no-op parity, same K/V inputs, keys/pointers/banks/weights unchanged, matched logical representative counts. Here value=MPM and joint=matched-count ToMe; these are real value interventions. Independent replay now passed byte-identically across all mask/presence hashes and four-layer audits; receipt intervention_gate_independent_receipt.json. No Dice evaluation/full-video efficacy authorized. Drive post_value_source_review_20261112/intervention_gate_attempt01.

MPM full-seq3 propagation engineering attempt01 passed all14post-prompt targets ×noop/MPM/matched-count-ToMe, preserving source schedules/physical counts/pointers/weights and native temporal slots. Noop all predictions and stored-state hashes exact. Independent fresh-process replay now passed byte-identically across all14frame records perarm including prediction and stored-memory hashes; receipt propagation_gate_independent_receipt.json. No efficacy claim. Drive post_value_source_review_20261112/propagation_gate_attempt01.

## Objective

Develop and rigorously evaluate training-free token-level utilisation in pretrained SAM2 memory attention that produces a meaningful, reproducible full-video PolypGen Dice improvement over native SAM2 and is competitive with the frame-wise YOLO-box segmentation strategy (user-reported mean Dice approximately 0.74). Evaluate all 21 eligible sequences among seq1–23, excluding seq1 and seq7. Verify the frame-wise result artifacts and exact averaging/protocol before treating 0.74 as a matched benchmark. Use identical evaluation frames, annotations and equal-sequence present-frame Dice; document checkpoint, resolution, prompt frequency, detector inputs and compute differences. Compare native SAM2 and matched generic token controls to isolate the proposed mechanism; retain Run155 as an exploratory reference, not the primary success target. Keep original SAM2 weights, memory writes, dense current-frame queries, frame processing/prompt schedule and decoder frozen; the memory representation may summarize a longer history under the same native K/V token budget. No good/bad frame selection or confidence gates. Methods need not use rigid foreground/background/boundary partitions. Reuse pinned official paper implementations, inspect prior runs to avoid repeating failed ideas, and verify every integration before scientific inference: native/no-op parity, K/V alignment, positional/temporal handling, pointer preservation, object-ID isolation, unchanged weights/banks and independent-process replay. Follow protocols/ACTIVE_RESEARCH_GOAL.md, protocols/TOME_REGION_IDENTIFICATION_STUDY.md and protocols/TOME_STRATEGY_REVIEW_20261111.md. Current ATC/PiToMe candidates failed their frozen screens; do not relaunch them or relax criteria. Diagnose mechanisms and freeze settings, controls, minimum meaningful gain, uncertainty analysis and stopping rules before new efficacy runs. Full-video propagation, not sampled immediate-read Dice, must support the final performance claim. Report Boundary F, false alarms with denominators, reappearance, runtime, paired uncertainty, sequence wins and sensitivity to seq8; require robust gains rather than one-sequence-driven means. Treat already analysed PolypGen/DAVIS data as exploratory and require untouched-data confirmation for generalization. Save artifacts, source hashes, commands, receipts, progress and concise result tables on the requested Drive mount; update docs/REGION_TOKENS_RESULTS.md. Stop unproductive variants under declared criteria; do not force positive results or claim clinical usefulness/publishability from a target score alone.

## Performance targets

| Comparison | Required evidence |
|---|---|
| Native SAM2 | Meaningful positive paired Dice gain, with a minimum practical gain declared before efficacy evaluation; independent-process reproducibility and sequence-level uncertainty |
| Frame-wise YOLO-box segmentation | Report both the historical all-frame value 0.748099 and the matched post-prompt, GT-present saved-output value 0.718842. The former uses a different denominator; neither is a fully matched inference benchmark until checkpoint/detector/protocol provenance is recovered. Compare paired per-sequence Dice and false alarms, not only the headline mean |
| Generic token utilisation control | Matched local token budget and insertion point to isolate the proposed mechanism |
| Run155 | Historical exploratory reference (0.734707); no longer the primary success target |

The user's reported ~0.74 remains a historical reference, not a valid matched present-frame threshold. Competitive equivalence requires a predeclared margin and paired uncertainty; superiority requires an appropriate positive comparison. Report the mean without seq8 and the spread of sequence effects so a single exceptional sequence does not establish reliability.

## Benchmark protocol

- Use seq2–seq23 excluding seq7: 21 eligible sequences; seq1 and seq7 excluded.
- Evaluate every eligible post-prompt frame under the frozen full-video protocol. Compute Dice within each sequence on GT-present frames, then average the 21 means equally. Assess GT-empty frames separately with explicit false-alarm denominators.
- Verify the frame-wise artifact source, model/checkpoint, masks, resolution, frame coverage and aggregation. Reuse existing verified native outputs where compatible.
- Native versus token method must have the same checkpoint, initialization, native frame schedule and evaluation rules. For the frame-wise comparison, document its repeated detector/box access, prompt frequency and runtime: these are system-level strategy differences, not an isolated token-only ablation.
- Freeze settings, controls, practical effect, uncertainty procedure and stopping criteria before efficacy inference. No tuning to the final reported cohort or selecting a best seed from evaluation results.

## Implementation requirements

Reuse pinned official paper implementations through minimal adapters. Verify native/no-op parity, K/V assignments and budgets, spatial/temporal handling, pointer preservation, persistent object IDs, unchanged SAM2 parameters and native-write procedure, and independent-process replay before scientific inference. Full propagation may naturally develop different predicted memories; do not confuse an unchanged writing procedure with identical intervention histories.

Region identification, token matching and aggregation are separate design choices. Grad-CAM, max pooling or k-means must not be described as tested segmentation methods unless an actual verified experiment supports that statement. Do not require rigid mask-derived roles when evidence does not justify them.

## Strategy comparison map

Keep the four design questions separate when interpreting results:

| Design question | Strategies considered | Existing evidence / status |
|---|---|---|
| Which spatial tokens are grouped together? | Generic ToMe feature similarity; predicted-region-constrained ToMe; ATC average-linkage clustering; PiToMe protected-token matching | Run155 and the fixed-budget wider-history study are complete. The new regional arm is only +0.000344 Dice over generic ToMe; the full-cohort robustness screen failed. ATC and PiToMe failed prior continuation screens. Do not restart those variants. |
| How are regions identified? | Stored predicted-mask probabilities; prompt-oriented TokenCut feature partition; Grad-CAM / gradient attribution; object-pointer prototypes | Predicted-mask roles have been used for ToMe comparisons. TokenCut and Grad-CAM were assessed as identification signals only, on two fixture sequences; Grad-CAM did not provide a reliable foreground label. Pointer-centre grouping remains untested. None of these identification diagnostics proves a segmentation gain. |
| How are grouped features aggregated? | ToMe mass-weighted average; channel-wise max pooling | Weighted average is the tested aggregation. Max pooling has not been tested here; keep it as a future candidate only if current evidence motivates it, because a per-channel maximum may combine incompatible spatial evidence. |
| How is a fixed token budget distributed over time? | Native six-frame read; 12-map uniform allocation; 10-map uniform versus recent-dense allocation | The completed 12-map run couples wider coverage with lower per-map density and oldest-slot reuse. The 10-map 2×2 comparison completed all21 sequences and failed its screen; it is stopped. |

The completed run did **not** compare every combination in this matrix. It compared generic with region-aware matching inside one frozen wider-history design. Evidence from one axis does not validate the others; no broad parameter sweep is justified by these results.

## Current evidence and next work

**Statistical-calibration ToMe candidate (2026-11-13):** the paper-derived scaling primitive passed its focused checks, but the five-arm comparison was stopped for missing cross-calibration assignment replay. The original engineering gate was repeatable but incomplete. Exact replay must be implemented and independently gated before a fresh efficacy attempt; see the current status at the top of this file. The original failed attempt is retained, with no efficacy claim or threshold changes.

**Partial integrity audit (2026-11-13):** seq2–4 completed all five arms (120 targets total). A reusable read-only audit confirmed identical frame IDs, target counts, and setup hash across arms; four attention layers per treatment target; matched regional/generic quotas and source-frame metadata; mass conservation; and unchanged SAM2 parameters. No discrepancies were found. This does not report an interim efficacy result. Checker: `scripts/audit_stat_calibrated_tome_completed_sequences.py`; receipt: Drive `stat_calibrated_tome_polypgen_v1/full_attempt01/audits/completed_sequences_seq2_seq4_integrity_20261113.json`.

| Completed candidate | Native Dice | Treatment Dice | Disposition |
|---|---:|---:|---|
| Run155 regional | 0.710224 | 0.734707 | Exploratory; uncertain paired gain and seq8 dependence |
| 12-map regional | 0.710224 | 0.731901 | Failed frozen robustness screen |
| 10-map recent-dense regional | 0.710224 | 0.705649 | Failed frozen screen; all four arms complete |
| Native-schedule V-only | 0.710224 | 0.711331 | Failed frozen screen; full integrity audit passed |
| Pointer-conditioned ToMe | 0.710224 | 0.710213 | Full 21-sequence screen failed; stop |
| Fixed-prompt-pointer ToMe (Seq5 diagnostic) | 0.541987 | 0.541171 | Same loss intervals; development diagnostic only |
| Seq5 MCTF attention-aware V fusion | 0.541987 | 0.541826 | Post-hoc single-sequence test; no recovery or useful gain |
| Fixed-bank token utility (uniform tile removal) | — | — | Mean paired change +0.000455; CI spans zero; sampled targets, not propagation |
| Run171 rejected-group selector | 0.714797 | 0.714756 | −0.000041 vs native; loses to random control |
| Run173 regional utility predictor | — | — | Failed frozen prediction criterion; no efficacy run |

No full-cohort or diagnostic process is currently active. Do not resume any completed candidate because an archived checkpoint below uses the word active or next. Historical ledger entries record what was known at that time; they are not current run instructions.

1. Preserve completed results and stop decisions. Do not tune budgets, seeds or matching rules on this cohort to force a positive result.
2. The fixed-bank utility studies and the Seq5 frame-157/frame-184 token-group diagnostic are complete and did not validate a selector or recovery rule. Do not repeat them or start another Seq5-specific ToMe variation. Continue only with a distinct, source-backed full-cohort token-level hypothesis under the frozen goal.
3. Treat YOLO as a saved-output comparison, not a matched inference benchmark, until its generating checkpoint/detector/protocol is recovered.
4. Before another efficacy run, require a literature-supported mechanism, matched generic control, frozen minimum effect and stopping rule, and a passing engineering gate.
5. Confirm any surviving method on untouched data. Existing PolypGen/DAVIS results remain exploratory; do not claim generalization from them.

**Source-screen update (2026-11-13):** reviewed official STTM (ICCV 2025; cross-frame similarity-threshold merging for video QA) and *Video, How Do Your Tokens Merge?* (CVPR 2025 Workshop; fixed-budget ToMe in action-recognition encoders). Neither is a faithful, no-threshold SAM2-memory method or evidence of dense VOS Dice improvement. No code change or efficacy run is justified. Full source notes: `protocols/TEMPORAL_TOKEN_MERGING_SOURCE_REVIEW_20261113.md`.

**CMSA-Net source audit (2026-11-13):** cloned the official repository at `4e4602e93211b5d0006a8a7415e8477de413b79d` to `external/CMSA-Net`. CMA is a learned three-scale feature-cross-attention block used before CMSA-Net's mask guidance/decoder. Correction: its test code updates the two initially fixed references with semantic and confidence criteria; the earlier claim that no dynamic update was implemented was incorrect. Its frame references are distinct from SAM2's object-pointer vectors. No compatible training-free SAM2 K/V component or transferable weights are supplied for direct transplantation. Source audit: `protocols/CMSA_MULTISCALE_TOME_REVIEW_20261113.md`.

**SAM2 spatiotemporal-pruning literature check (2026-11-13):** the TPAMI paper is directly relevant to token operations inside SAM2 memory attention, but its goal is acceleration with negligible accuracy disturbance, not Dice improvement. Exact-title/DOI/author searches did not identify an official code repository. Per the upstream-first rule, do not recreate its pruning method from prose or run it as an efficacy candidate. Reassess only if official code becomes available and an efficiency baseline is useful; any accuracy claim still needs a matched PolypGen study. Review: `protocols/SAM2_SPATIOTEMPORAL_TOKEN_PRUNING_REVIEW_20261113.md`.

**Additional upstream candidate screen (2026-11-13):** RS³-Prune has no code link in its arXiv source and changes both dense query participation and memory writes; its reported SAM2 J&F effects vary by benchmark (including a DAVIS-17 drop). ERMU's advertised GitHub URL returns “Repository not found”; its described relevance–redundancy memory update also changes frame-memory management. Efficient-SAM2 SMR is already cloned and audited, but repeats the failed hard top-k direction. Official DTMFormer is cloned at `47d28fea4fa2e92ce39b5f521245b58a4540753b`; it provides useful trained segmentation-aware merge/reconstruction code, but its learned SETR stack cannot be transplanted into frozen SAM2 memory K/V. The 2026 PTM paper's cited repository also returns “Repository not found.” No new candidate meets both the frozen protocol and upstream-integration requirements. Do not implement these from scratch or start an efficacy run. Full screen: `protocols/MEMORY_TOKEN_ACCURACY_CANDIDATE_REVIEW_20261113.md`.

## Evidence locations

- Design: [identification study](TOME_REGION_IDENTIFICATION_STUDY.md).
- Source review: [strategy review](TOME_STRATEGY_REVIEW_20261111.md).
- Results: [results tables](../docs/REGION_TOKENS_RESULTS.md).
- Drive: `outputs/experiment/tome_region_identification_v1/` — completed studies `causal_series02/`, `pitome_causal_series01/`; synthesis `evidence_synthesis01/`; official-source review `task_aware_source_review01/`.
- Current strategy inventory and run checkpoint: `outputs/experiment/tome_region_identification_v1/strategy_comparison_inventory_2026-11-12.md`.
- The preceding work order is preserved on Drive in `goal_revision_framewise_20261111/previous_active_research_goal.md`; its completed or superseded steps are not new run instructions.

## Historical evidence ledger — chronological, superseded actions are not instructions

The following entries preserve earlier checkpoints. The current research status and next-work sections above take precedence for run state.

### Frame-wise baseline averaging audit

The supplied MedSAM2_frame_yolov8_boxes CSV mean 0.748099 across all 21 sequences is reproduced from saved masks. On 1,689 matched post-prompt GT-present frames, the equal-sequence mean is 0.718842; 17/323 GT-empty frames are false alarms. All 2,012 saved-mask Dice scores match the CSV exactly at the stored precision (zero mismatches at 1e−8). No inference rerun. Historical checkpoint/detector/preprocessing/runtime provenance remains unverified, so 0.748099 is not a matched present-frame benchmark. Receipt and per-sequence values are on Drive under `framewise_baseline_audit01/`; concise table is in `docs/REGION_TOKENS_RESULTS.md`.

## Bounded diagnostic design

`protocols/tome_attention_distortion_diagnostic_v1.json` specified six fixed seq2/seq3 targets and existing 50%/25% token-reduction merges as shadow reads from identical native Q/K/V at each layer. Only native outputs propagated. Three focused analytic tests passed; protocol and 23 source hashes were frozen. Both real-frame attempts completed, and their parsed results were exactly identical in a fresh process. The receipt and descriptive analysis are on Drive in `attention_distortion_diagnostic_v1/`. This measures attention-output distortion, not Dice, and does not authorize an efficacy claim.

## Archived 12-map study and current 10-map comparison

The 12-map full cohort is complete and failed the frozen robustness screen. Its generic and regional means are 0.731557 and 0.731901; the positive mean over native depends strongly on seq8. It must not be relaunched as a pending candidate.

The active study is the four-arm ten-map allocation comparison specified above. Representative-position and joint K/V matching have passed engineering gates but have no efficacy result. They are separate candidates, not modifications to the active run.

## Implementation review correction — 2026-11-12

Fixed the shadow observer's norm-weighting bug: norm statistics now use cluster mass rather than log mass. Both regression cases failed before the fix; 35 focused tests pass. Flagged 288 affected scalar fields per historical shadow replay; original raw outputs remain preserved. Attention-output distortion and full-video Dice are unaffected, and all 21 active-run launch-source hashes still match. A pinned-ToMe synthetic counterexample confirms that same-region membership does not guarantee attention-preserving K/V averaging; it does not identify a causal PolypGen failure. Resolve K/V and history confounding before promoting another candidate. Details and code/test receipt: Drive `outputs/experiment/tome_region_identification_v1/attention_distortion_diagnostic_v1/norm_weighting_correction_20261112/`. The full cohort remains active and no new efficacy run was launched.

Fixed-assignment diagnostic review: pinned official ToMe merge/unmerge operations confirm that expanded mean keys with original values yield the same output as joint mass-corrected K/V compression. Two fresh processes reproduced the synthetic identity (maximum difference 2.22e-16). Future K/V counterfactuals must account for their interaction rather than interpret errors as independent additive contributions. This is mechanism-design evidence only; scripts and replay receipts are in the Drive correction folder above.

## Verified inputs for the next mechanism diagnostic

The seq6 and seq8 archived native banks passed raw-file checksums, receipt provenance and payload-fingerprint verification. Ten fixed frames are available: seq6 31, 32, 50–53; seq8 7, 8, 21, 32. These were selected after inspecting failures/recovery, so they are explicitly exploratory. No GT masks were read and no model inference was launched. The input plan does not authorize execution: the diagnostic code and hashed protocol must still be completed, including native replay, fixed source tensors, pointer/bank preservation, K/V interaction handling and independent replay. Drive: `outputs/experiment/tome_region_identification_v1/kv_selectivity_mechanism_v1/design/verified_input_plan.json`.

The fixed-assignment helper is now implemented in `modeling/fixed_assignment_attention_diagnostic.py`, reusing existing compression plus pinned official unmerge. Forty focused tests pass and two fresh processes produce identical synthetic results. The bounded four-target observer gate is now implemented, tested and frozen. The ten-target diagnostic remains unauthorized for automatic execution until that gate passes and its results are reviewed. Drive engineering receipt: `kv_selectivity_mechanism_v1/engineering_helper_gate01/receipt.json`. Active-run launch sources remain unchanged.

## Original queued gate — superseded before inference

`protocols/fixed_assignment_attention_gate_v1.json` freezes the four-target gate and 50 source dependencies after 46 focused tests pass. Queue PID 67000 is alive, waiting for the specific cohort process PID 44917 and its complete 21-sequence/four-arm ledger. It then runs only seq6 31/32 and seq8 7/8 in two fresh processes. Compare exact native six-map replay, dense ten-map observer parity, counterfactual identities, source-frame/pointer alignment and frozen bank/parameter state. Failures stop the queue without retry or tolerance changes; no broader diagnosis or efficacy run follows automatically. Drive queue/progress/logs: `outputs/experiment/tome_region_identification_v1/kv_selectivity_mechanism_v1/queued_engineering_gate01/`; protocol source snapshot: `design/`. Goal success remains unproven.

Latest live verification at 10:48 +07:00: cohort PID44917 and waiting queue PID67000 both exist. The seq15 recent-dense prefix audit passed (generic 115 targets, regional 29 at snapshot), including exact native parity before compression. Receipt is under the active cohort's `audits/seq15_recent_dense_native_prefix_20261112.json`. Full-cohort results and real diagnostic-gate results remain pending.

Milestone at 10:53 +07:00: all seq15 arms completed and their complete audit passed across 115 targets/arm and 1,840 layer reads. Native-prefix parity, summaries, matched generic/regional quotas and temporal source lists, pointer/mass budgets and frozen-state receipts passed. Receipt: active cohort `audits/seq15_complete_all_arms_20261112.json`. Cohort PID44917 has advanced to seq16; queue PID67000 remains alive and waiting. Thirteen of 21 sequences are complete in all arms; final results remain unavailable.

## Current queue and reporting state — 2026-11-12 11:00 +07:00

Cohort PID44917 remains active at seq17, with 14/21 sequences complete in all arms. Seq16's complete audit passed across 33 targets/arm and 528 attention-layer reads (`audits/seq16_complete_all_arms_20261112.json`). Waiting queue PID67000 was stopped before any diagnostic attempt to correct GT-access metadata. Replacement PID69974 is confirmed live and waiting, using `protocols/fixed_assignment_attention_gate_v2.json`; protocol SHA-256 `49fad34fa59abcc7d8a723cb0b4fe1cef6d5b9bce568feee1ea35617ff72443b`. Standard GT initialization is used; post-prompt GT masks are not read. Targets, tolerances and inference behavior did not change. Output folder: `kv_selectivity_mechanism_v1/queued_engineering_gate02/`. The old queue and v1 protocol remain archived with an explicit supersession receipt.

`docs/REGION_TOKENS_RESULTS.md` is condensed to current verified comparisons, all 21 per-sequence rows, uncertainty/seq8 sensitivity, DAVIS transfer and next actions. The full 1,401-line prior report and numeric source hashes remain on Drive in `reporting_snapshot_20261112/`. All 105 displayed per-sequence Dice values were checked against JSON. Keep ongoing audit logs on Drive rather than growing the concise results report.

Connected Drive provenance follow-up (2026-11-12): the connector profile differs from the mounted experiment owner. Searches for the specific YOLO result folder, AdeSEG, yolov8, MedSAM2 and PolypGen returned no accessible matches. This cannot establish absence from the owner's Drive and does not close the benchmark-provenance gap. Receipt: `framewise_baseline_audit01/connected_drive_provenance_search_20261112.json`. Continue using the mounted artifacts and keep competitiveness claims unverified until historical protocol evidence is available.

Current milestone at 11:05 +07:00: seq17 is complete in all four arms and its complete audit passed across 46 targets/arm and 736 layer reads, with exact native-prefix parity, matched quotas and source frames, valid summaries, conserved mass/pointers and unchanged launch sources. Receipt: active cohort `audits/seq17_complete_all_arms_20261112.json`. Cohort PID44917 is on seq18 (15/21 complete); replacement queue PID69974 remains alive and waiting under v2.

External-data follow-up (2026-11-12): retained the already planned LVOS v2 evaluation split. Official Drive metadata resolves `valid.zip` (11,833,302,254 bytes); the authors document 140 evaluation videos with pixel masks. No raw archive or outcome was downloaded/inspected. No LVOS-named prior run directory was found, but untouched/pretraining exposure is not fully established. Do not substitute small legacy SegTrack results for primary PolypGen success. Receipt: `external_confirmation_availability_20261112/receipt.json`; dataset plan availability is updated.

Seq18 prefix audit at 11:10 +07:00 passed: uniform-regional 62/62 targets and recent-dense-generic 18/62 at snapshot align with native frames/GT presence; both match native prediction and stored-memory hashes on their first seven targets, with four-layer/no-merge coverage. Receipt: active cohort `audits/seq18_native_prefix_20261112.json`. Both cohort and replacement queue handles remained live; no terminal receipt was present.

Seq18 complete audit at 11:15 +07:00 passed across all four arms: 62 targets/arm, 992 layer reads, seven exact native-prefix targets/arm, matched quotas/source frames, conserved pointers/mass, recomputed summaries and unchanged launch sources. Receipt: active cohort `audits/seq18_complete_all_arms_20261112.json`. Cohort advanced to seq19; no efficacy conclusion is drawn from this integrity check.

Seq19 complete audit at 11:24 +07:00 passed across all four arms: 55 targets/arm, 880 layer reads, seven exact native-prefix targets/arm, matched quotas/source frames, conserved pointers/mass, recomputed summaries and unchanged launch sources. Receipt: active cohort `audits/seq19_complete_all_arms_20261112.json`. Cohort advanced to seq20; this is an integrity check, not an efficacy conclusion.

Seq20 complete audit at 11:30 +07:00 passed across all four arms: 35 targets/arm, 560 layer reads, seven exact native-prefix targets/arm, matched quotas/source frames, conserved pointers/mass, recomputed summaries and unchanged launch sources. Receipt: active cohort `audits/seq20_complete_all_arms_20261112.json`. Cohort advanced to seq21; no efficacy conclusion is drawn from this integrity audit.

Seq21 complete audit at 11:38 +07:00 passed across all four arms: 47 targets/arm, 752 layer reads, seven exact native-prefix targets/arm, matched quotas/source frames, conserved pointers/mass, recomputed summaries and unchanged launch sources. Receipt: active cohort `audits/seq21_complete_all_arms_20261112.json`. Cohort advanced to seq22; no efficacy conclusion is drawn from this integrity audit.

Seq22 complete audit at 11:45 +07:00 passed across all four arms: 45 targets/arm, 720 layer reads, seven exact native-prefix targets/arm, matched quotas/source frames, conserved pointers/mass, recomputed summaries and unchanged launch sources. Receipt: active cohort `audits/seq22_complete_all_arms_20261112.json`. Cohort advanced to seq23; no efficacy conclusion is drawn from this integrity audit.

2026-11-12 11:54 +07: ten-map full cohort completed 21/21; frozen screen FAILED. Native Dice 0.710224; uniform generic/regional 0.706793/0.706820; recent-dense generic/regional 0.705597/0.705649. Primary delta −0.004575 (95% CI [−0.017683,+0.005561], Holm p=1, 7/2/12 W/T/L); without seq8 −0.004807. Regional-vs-generic increments negligible. Seq23 full audit passed (49 targets/arm,784 layer reads). Diagnostic queue began attempt_01 PID83931 after validating predecessor completion. Do not relaunch efficacy variants or relax screen; inspect diagnostic gate next. Full objective remains unmet.

2026-11-12 11:55 +07: fixed-assignment attention engineering gate v2 PASSED, 4 targets ×4 layers ×2 fresh processes, exact parsed-results replay SHA d25a3365f655aa2e8dc278a802bd893b260d2690cb2333d2500064e53a4d0366. Native replay/dense observer parity, no writes, bank/weight integrity and K/V identity checks passed. First compressed-read relative squared error 0.000016–0.000699 is diagnostic only. Queue terminal complete; no efficacy job. Review remaining six reserved targets and freeze broader diagnostic before launch.

Reserved-six-target diagnostic launched after gate review: seq6 frames50–53 and seq8 frames21/32, fixed native banks, unchanged official ToMe operator and tolerances, two independent processes. Protocol `protocols/fixed_assignment_reserved_diagnostic_v1.json`; output Drive `kv_selectivity_mechanism_v1/reserved_six_target_diagnostic01/`. No post-prompt GT supervision, writes, Dice inference or efficacy launch. Any failure stops; inspect exact replay and per-layer measurements before changing design.

Reserved-six-target attempt01 STOPPED at fixed-cluster identity FP32 tolerance check. Attempt02 not launched. Do not relax tolerance or treat this as a completed mechanism result. Preserve raw failure/traceback and inspect failing frame/layer plus numerical vs assignment origin before any amended diagnostic. No efficacy job active.

Failure forensics completed with original failure preserved: seq6/frame50 layer0, FP32 maxabs5.9485e−5 vs5e−5 bound, RSE5.05e−10 below1e−8. FP64 recomputation reduces reported residual to5.96e−8 (existing reporter casts toFP32). Evidence supports numerical precision rather than establishing an assignment bug. No tolerance change, retry-to-pass or efficacy run. Any higher-precision verification requires explicit amended protocol retaining original failure and separate FP32 execution-fidelity checks.

Precision-reporting fix verified: `_output_distortion` retains FP64 when either reference is double; FP32 semantics unchanged. Regression failed before fix;17 focused tests pass. One-target amended forensic protocol reproduced identical FP32 failure and precise FP64 residual max1.6768e−9,RSE1.2392e−18. This supports numerical reduction error; no assignment bug established, no original gate reclassification. Evidence Drive `identity_failure_precision_forensics02/review.json`. Next design decision must account for failed full cohort and cannot equate numerical gate diagnosis with efficacy support.

Posthoc saved-trajectory audit completed for seq6/8/14/17/19 with exact frame/presence alignment and source hashes. Recent-dense regional: seq14 adds36 zero-Dice present targets (native0), seq6 leakage at50/51 then missed foreground52/53; seq17 gain mostly precision; seq8 retains34 zero-Dice present targets in both native/treatment. First prediction divergence matches first compressed wider read in all5, not a causal separation. Receipt/script under cohort `failure_analysis/`. Do not tune region labels from these outcomes; next experiment must separate wider history from compression/propagation.

Independent aggregate arithmetic audit verified all5 equal-sequence Dice means from21 sequence summaries; full ten-map per-sequence table added to results report. Next bounded diagnosis specified in strategy review: native-six/dense-ten/compressed-ten on identical cached bank, score after forward and never write; dense reference explicitly overbudget diagnostic. Not yet launched or represented as implemented.

Implemented same-bank output contrast runner by reusing fixed-assignment replay runner, cached_read, existing scoring, and FixedBudgetLongHistoryToMe. Fixed dense-receipt schema assumption;19 focused tests pass. Engineering gate v2 passed4targets in2fresh processes with exact results SHA69a600324902d3b4b6952dc9fa10c4cab44fba61220fecc1591e382e26178c48; native resize/scoring equals archived metrics exactly. Original gate01 failure preserved. Broader reserved-target diagnostic not yet launched; no efficacy run.

Reserved same-bank diagnostic completed6targets in2fresh processes with exact results SHAfb648edcd62c33ce06353ee3df8c0e83636db68931b6481baa5f2a0b9cf4b974; native scoring matches archive. Seq6frames50–53: compressed/native-bank Dice0.9464/0.9119/0.9328/0.9064 vs propagated generic0.5096/0.4830/0.7000/0.6731. Immediate compression does not reproduce rollout collapse on native history. This motivates paired memory-trajectory/feedback diagnosis, not causal proof or another matcher sweep. Seq8frames21/32 remain0 across all3arms. Results/analysis in Drive `history_compression_reserved01/`; no efficacy candidate promoted.

Seq6 trajectory inventory: archived uniform-generic directory contains frames.json only, no treatment bank. First mergeframe32 does not immediately degrade (frames32–37 near/above native); large lossframe50 has object-presence+8.10, frame51+7.60, so negative-presence suppression does not explain these losses. Evidence cohort `failure_analysis/seq6_trajectory_inventory.json`. Next bounded replay must recover treatment states and match every archived prediction/stored hash through endpoint before cross-bank tests. Avoid full cohort rerun; no live inference job currently.

Bounded seq6 replay01 complete: prompt24 through53,29targets116layerreads; every archived mask/stored hash/score/object-score exact; weights unchanged, temporal slots restored. Treatment bank recovered Drive `kv_selectivity_mechanism_v1/bounded_seq6_replay01/uniform_generic/seq6/replay_bank.pt`, payloadSHA c1231a5200df7ba54d5e8b6bf93fca48cfbc4dc245f2d85955d908d0f0f3e791. Independent replay02 active (exec session45100); poll same handle. Protocol bounded_seq6_replay_v1.json. Logging-only issue: progress denominator retains full66 targets, actual bounded29, do not amend source while replay02 active. No bank intervention yet; compare exact rows/payload fingerprints after replay02 completes.

Independent bounded seq6 replay02 completed; all29 records and treatment-bank payload fingerprint exactly equal replay01. Treatment-bank three-reader diagnostic completed4targets in2processes, exact resultsSHA358573bb931405d2f0263e80a628f6636ce7e166d6e9a78def2373fb84b894f7. On native bank readers remain~0.91–0.95; on treatment bank native-six/dense-ten/compressed-ten all remain~0.48–0.70. Reader-only change does not recover degraded history. Next mechanism separation should distinguish spatial memory vs pointer/mask-conditioned content before any repair candidate; no efficacy job active. Logging denominator fixed after replay completion (full-video defaults unchanged).

Stored-component intervention completed4targets×2fresh processes with exact resultsSHA863e439a77851d2581a329e66a4f70598deb0b07e9b423a99c594529d2f9a16b. Swap native spatial features+positions into treatment history with treatment pointers retained: Dice0.9424/0.9155/0.9322/0.9042 vs treatment0.5096/0.4830/0.7000/0.6731. Native pointers alone0.5106/0.6780/0.6868/0.8014. Localizes these four failures primarily to spatial memory trajectory; does not supply a deployable repair or universal conclusion. Donor/base fingerprints/weights unchanged; no writes. Evidence Drive `memory_component_diagnostic01/analysis.json`. No efficacy job active; next inspect spatial write-history divergence/source frames before proposing one repair.

Read-only recovered spatial tensor audit: seq6frames25–53 positionencodings/shapes/dtypes all identical, firstmasklogit andspatialmemorydifferenceframe32, presence-sign differences0/29. Frame49 binarymaskdisagreement0.084% but spatialmemoryRSE0.00631;frame50 10.11% and0.04915. Verified original `_encode_new_memory` fuses current unconditioned imagefeatures with scaledsigmoid masks. No spatial-layout or presence-sign defect found; continuous mask-conditioned feedback remains plausible. Audit/script onDrive. Do not treat binarymask parity as memory parity; original writes stay unchanged under goal. No efficacy job active.

Official-source screening/probe completed for vid-TLDR: training-free video retrieval/QA, not direct SAM2 memory cross-attention; query-aligned objective scores, mean clipping and uniform-attention nonfinite case verified from unmodified author function. No candidate promoted or efficacy run. MCTF official recipe includes finetuning; not evidence for frozen-SAM2 gains. Drive sourcepin/probe and strategy review updated.

New distinct engineering candidate implemented at primitive level: native-schedule value-only local ToMe merge/unmerge, retaining every original Q/K, pointer and conditioning token, no extra frames or regional partitions. Matched joint K/V consolidation control and noop included.7 primitive/ToMe tests pass; real-frame gate pending, no efficacy inference. Design `protocols/native_value_consolidation_design_v1.json` pins source, quarter-merging and4x4 windows; no parameter sweep. This is an accuracy hypothesis with no speed/token-count benefit. Do not promote from synthetic tests.

Native-schedule value-only consolidation real-frame gate PASSED:4targets×4layers×2fresh processes with exact resultsSHA d187842bf6b67b64c3c9b7d69da36b6bce7241ed4dc4afe9456f50935ebd7be1. Native/noop exact, native K and pointer preservation, matching source schedule/KVinputs, unchanged banks/parameters and seven temporal slots. First Q identical acrossmodes; subsequent Q legitimately varies.12 focused tests passed. Receipt Drive `kv_selectivity_mechanism_v1/native_value_consolidation_gate01/completion.json`. No post-prompt GT masks read; no writes or efficacy result. Next requires bounded full-propagation/noop archive replay gate before any21sequence efficacy study, with frozen controls and stopping criteria.

Native-schedule propagation gate launched using existing run_condition with a reader factory and exact-native checking policy; default older runners unchanged. Seq3 all14post-prompt targets ×noop/value/joint. Attempt01 complete and passed: no-op predictions AND stored-memory hashes exact at every target, source schedules/token counts/pointers match acrossarms, weights/temporal slots unchanged. Attempt02 active execsession85899 (childPID94758) atnoop12/14 snapshot. Gate protocol `protocols/native_value_propagation_gate_v1.json`, Drive `kv_selectivity_mechanism_v1/native_value_propagation_gate01/`.15focused tests passed. No cohort efficacy run yet; poll same handle and require exact independent rows.

Full-seq3 propagation gate completed two-process exact replay SHA a3cd7af4b79f5545976443538e7953686f6c6401f0240a84772153bc2d306d2d; all14 no-op masks/stored states equal archive. Full-study runner/statistical tests passed15focused checks, including rejection of incompletecohort and one-sequence-driven syntheticgain. Launched fullvalue/joint study PID95838 under frozenprotocol, not a guarantee of success.

Seq2 native-schedule audit passed at12:36+07: value59/59 and joint7/59 snapshot,264 layerreads; exact source frame IDs/age slots, strictlypast memory, native spatialtoken budget/mass, firsttarget native mask/stored-state parity and all56sourcepins. Receipt full_attempt01/audits/seq2_schedule_2arms_59targets.json. Cohort process confirmed live; no efficacy conclusion from partial output.

Seq2 both-arm complete integrity audit passed:59targets/arm472layerreads, source frame/age slots and dense native budget match, first native prediction/stored-state parity,56sourcepins, recomputed summaries/runtime receipts with unchangedweights/restored temporal slots. Receipts `audits/seq2_schedule_value59_joint59.json` and `seq2_complete_summary_runtime.json`. Audit receipt filename collision was corrected without overwriting prior snapshot; full-study code unchanged. Cohort moved toseq3.

Seq3 completed full-study records exactly equal the independent propagation-gate records in botharms (all14targets including stored memory). Native-schedule/budget audit112layerreads and all56sourcepins pass; summaries/runtime receipts checked. Receipt `audits/seq3_complete_gate_reproduction.json`. Report stale waiting/running gate text removed after archiving before/after/hashes onDrive under reporting_snapshot_20261112; completed numeric tables retained. Cohort nowseq4/joint,2/21 complete; no implementation changed while active.

Seq4 complete integrity audit passed:47targets/arm376layerreads, exact native frame/age schedule and token/mass budgets, firsttarget parity,56sourcepins, recomputed summaries and unchangedweights/restored temporal slots. Receipts `seq4_schedule_value47_joint47.json` and `seq4_complete_summary_runtime.json`. Cohort advanced toseq5/value,3/21complete; no efficacy inference.

Seq5/value full-arm audit passed249targets996layerreads; joint10targets40layerreads at schedule snapshot. Native source/age schedule, strictlypast memory, dense budgets/mass, firsttarget parity and56sourcepins pass; value summary/runtime recomputed and weights/temporal table unchanged. Receipts `seq5_schedule_value249_joint10.json` and `seq5_value_complete_summary_runtime.json`. Joint now11/249; no efficacy conclusion from this incomplete pair.

Active-run environment receipt saved at `full_attempt01/audits/runtime_environment.json` (Python/Torch/macOS/architecture, CPU2threads and deterministic dispatch from pinned source). All56sourcepins still match. Wall times include integrity/matching overhead; no speedgain claim or historicalYOLO runtime comparison. PID95838 confirmed advancing seq5/joint100→126/249; no terminal/error receipt.

Seq5 both-arm full integrity audit passed249targets/arm1992layerreads, exact native source/age schedule, nofuture frames, dense native budget/mass, firsttarget parity and56sourcepins. Both summaries recompute exactly; weights unchanged and temporal slots restored. Receipts `seq5_schedule_value249_joint249.json` and `seq5_complete_summary_runtime.json`. Process advanced toseq6/value12/66;4/21complete. No efficacy conclusion from partialcohort.

## 2026-11-13 update — Seq5 diagnosis and candidate check

Pointer-conditioned ToMe full cohort PID50946 completed all21 eligible sequences; completion receipt passed and independent saved-output integrity audits passed for all21. Equal-sequence present Dice: native0.710224, pointer-conditioned0.710213, generic0.710073. Pointer−native−0.000011 (sequence-bootstrap CI95[−0.007212,+0.006357], Holm p0.9105;7/21 wins); pointer−generic+0.000140 with CI spanning0. Frozen meaningful-gain screen failed. No claim of improvement. Report and audit: Drive `pointer_conditioned_tome_polypgen_v1/full_attempt01/results.md` and `audits/final_full_cohort_integrity_audit.json`.

Seq5 diagnosis: all three arms share85/198 GT-present zero-Dice frames (92,157–182,184–241), with nonpositive object-presence scores. On the other113 present frames Dice≈0.95. Native raw frame184 mask has zero GT overlap and10.9× GT area, so a global presence bypass is unsafe. Candidate is fixed-initial-prompt-pointer ToMe in each layer's actual projected-key space; SAM2 weights, memory writes, native presence handling, frame schedule and decoder remain unchanged. It is a mechanism diagnostic, not a promised fix.

Implementation: `modeling/prompt_anchor_tome.py` and `scripts/run_seq5_prompt_anchor_diagnostic.py`; frozen protocol `protocols/seq5_prompt_anchor_diagnostic_v1.json`. Seq2/seq3 real-frame gate passed; the four-arm full-video Seq5 diagnostic completed with Dice native 0.541987, fixed-prompt pointer 0.541171, same-frame pointer 0.541059, and matched generic ToMe 0.541216. It did not recover the shared long-loss intervals. Treat as exploratory because Seq5 motivated the candidate; no parameter tuning or full-cohort efficacy rerun follows from that result.


## Current status — Seq5 failure-window diagnostic complete (2026-11-13)

The bounded fixed-bank token-group diagnostic at Seq5 frames 157 and 184 is complete. Two fresh processes produced byte-identical results (SHA-256 `a83667cd3c4980628194b80f3041d5ee135c1495c62620ea611d3ab2d914fe5a`) after 160 single-tile removals each. At frame157, raw decoder Dice was 0.602104 but the native negative presence score suppressed the visible mask; none of 80 tile removals made that score positive. At frame184, raw Dice remained 0 across all removals and the raw prediction covered 421,814 pixels (~10.9× GT); retaining native suppression avoids a large false positive. This exploratory study does not justify another Seq5-specific ToMe modification or a cohort run. Stop tuning single-tile removal on Seq5; the overall PolypGen full-cohort improvement goal remains active and unmet. Report: Drive `outputs/experiment/seq5_failure_window_token_utility_v1/analysis.md`; concise result: `docs/REGION_TOKENS_RESULTS.md`.

The 0.602104 frame157 value is scored after upsampling raw decoder logits to the original frame; the older 0.574 value in the attention probe was scored on the 256×256 logits grid against downsampled GT. They use different grids and are not a replay discrepancy. Literature check: official SAM2Long and DAM4SAM mechanisms address reappearance through prediction-path/memory-management decisions, outside the frozen no-selection/no-gating token-level goal. No code fix is supported for Seq5 by the current evidence.

## Literature follow-up — CMA versus object pointers (2026-11-13)

CMSA-Net's learned CMA temporal multi-scale aggregation is a possible source of inspiration; its DMR reference-frame selection conflicts with the fixed-frame protocol. SAM2 object pointers are per-frame decoder-token vectors appended alongside spatial memories, not multi-scale reference maps. Pointer-conditioned ToMe already failed the frozen full-cohort screen. A scale-context-guided spatial ToMe matcher is only a pending design candidate; do not implement or launch it until the active statistical-calibration cohort finishes. Review: [`CMSA_MULTISCALE_TOME_REVIEW_20261113.md`](CMSA_MULTISCALE_TOME_REVIEW_20261113.md).

**Run155 Seq8 trajectory audit — 2026-11-13:** read-only inspection of saved per-frame outputs found regional ToMe changed 29/34 native zero-Dice GT-present frames into nonzero masks, including a 24-frame recovery stretch (frames21–44). It also increased GT-empty false alarms from4/19 to12/19. This explains the large Seq8 mean-Dice gain as a sequence-specific recovery/false-alarm tradeoff; it does not establish a robust regional mechanism. No new inference or code change. Report and input hashes: Drive `155_region_constrained_tome_polypgen_v1/full/seq8_trajectory_review_20261113.md`.

## DecAF source review — 2026-11-13

Pinned official DecAF revision `bf385b9a7fc73201dd79bd967cc0154a8888c0bc` and inspected paper/code. Its contrastive maps arise from separate language-model prompts and feed new SAM2 point prompts plus thresholded tracklet selection; this is not a native memory-token method and does not fit the frozen prompt/no-gating goal. Six exact-author normalization fixtures passed (no model inference). Do not transplant the full pipeline or claim a SAM2 memory gain. Source hashes, exact function and applicability table: Drive `tome_region_identification_v1/decaf_source_review_20261113/`. Conceptual relevance only; no new efficacy candidate.

## Signed feature attribution vs. local token utility — 2026-11-13

The original hypothesis remains that a poor whole-frame prediction can still contain useful local memory tokens. A bounded, post-hoc diagnostic aligned signed Captum Input×Gradient with saved tile-removal effects on the same four source→target pairs. Engineering integrity passed: source/checkpoint/data pins match; native output replay is exact for all four pairs in a separate process; all four memory-attention layers execute; source IDs, weights, memory, pointers and finite feature maps are checked. The local audit is `scripts/audit_signed_memory_feature_utility_overlap.py`; receipts and report are on Drive at `tome_region_identification_v1/signed_memory_feature_utility_overlap_v1/attempt_02/`.

| Sequence | Source Dice | Target Dice | Mask-support ρ | Presence ρ | Helpful / harmful / zero tiles |
|---|---:|---:|---:|---:|---:|
| seq2, 4→6 | 0.926676 | 0.935537 | +0.095 | +0.261 | 1 / 9 / 6 |
| seq3, 1→3 | 0.950152 | 0.856208 | −0.306 | +0.212 | 3 / 13 / 0 |
| seq17, 17→19 | 0.003777 | 0.903168 | −0.184 | −0.138 | 7 / 8 / 1 |
| seq22, 41→45 | 0.000000 | 0.942278 | −0.305 | −0.204 | 5 / 2 / 9 |

Seq17 and seq22 show mixed local deletion effects despite very poor source Dice, consistent with the intuition that frame quality does not determine every token's usefulness. The tested gradients do not rank those effects consistently; they are not a supported GT-free selector. This diagnostic does not modify SAM2 or test a full-video method, so the full-cohort Dice objective remains unmet. Do not relaunch this gradient selector without a new signal and frozen protocol.

## Multi-scale memory utility diagnostic — completed

A CMSA-inspired but independently implemented pooled SAM2 Q/K descriptor was
compared with local similarity, native attention mass and exact uniform tile
selection on the original 42 GT-present fixed-history targets (21 sequences).
Pooled-vs-local selected-removal advantage was +0.0000207 Dice, sequence-bootstrap
95% CI [−0.0000094,+0.0000606]; both chose the same tile for 34/42 targets.
Equal-sequence utility Spearman ρ was −0.0059 local and +0.0033 multi-scale;
correlations were defined on 37/42 targets. Excluding four earlier pilot reads,
ρ was −0.0349 and −0.0241 respectively. These are conditional tile-deletion
measurements, not full-video scores; no GT-empty safety analysis or independent
confirmation. The frozen descriptor is not promoted. Full results and limitations:
[`docs/REGION_TOKENS_RESULTS.md`](../docs/REGION_TOKENS_RESULTS.md#pooled-multi-scale-memory-utility-diagnostic--completed).
Source/target, native-replay and arithmetic audit passed; Drive:
`outputs/experiment/multiscale_memory_utility_v1/cohort_attempt01/`.

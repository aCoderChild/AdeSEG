# Region-token study: idea and methods

Branch: `experiment`. Updated 19 October 2026.
[Results and statistics](REGION_TOKENS_RESULTS.md) are reported separately.
This document consolidates the former code guide and v2 execution record.

## 1. Research idea

Test how **foreground-interior, boundary, near-background and far-background
spatial tokens** contribute to SAM2 memory attention and endoscopic segmentation.

The study distinguishes attention allocation from causal segmentation benefit:
a region receiving high attention is not necessarily more useful. Region
identity must also be separated from token count, distance and temporal composition.
No assumption of universal boundary superiority or improved Dice is made.

The segmentation models are reused from upstream repositories. Region labels,
experimental controls, measurements and evaluation adapters are project-specific.
The custom learned utility MLP, region readout and training entry point have been
removed; the retained diagnostic controller has no trainable parameters.

## 2. Region definition

The default taxonomy area-resizes a binary mask to a **64 × 64 token grid**.
Coverage is the foreground fraction of each grid cell, not a token-feature value.

| Region | Assignment |
|---|---|
| Boundary | Coverage strictly between 0.1 and 0.9, dilated by one grid cell |
| Foreground interior | Coverage ≥0.9 outside the boundary band |
| Near background | Coverage ≤0.1 within a three-cell dilation of the boundary band, excluding boundary |
| Far background | Remaining cells |

Boundary takes precedence; categories do not overlap. These labels identify
spatial support, not features proven to encode one exclusive semantic concept.
The mixed-coverage rule can miss grid-aligned edges. Sensitivities include
`contour_union`, different coverage thresholds and boundary-band widths.

GT-defined supports are oracle diagnostics. Predicted-mask supports are used
for deployable-policy diagnostics and Efficient-SAM2 selection visualizations.
Neither mask coverage nor prediction uncertainty should be called learned
boundary semantics.

## 3. Protocol and experiment map

Primary model: frozen **SAM2.1 Hiera-B+, 1024 input**, one fixed checkpoint,
seven memory frames and native object pointers. Initialize with the first
labeled visible GT mask, propagate forward, and give no later corrections.
Evaluate post-prompt frames, including empty targets.

PolypGen has 21 promptable sequences; seq1/7 are reported as unpromptable.
Development uses seq2–6 and seq8–15; seq16–23 form the held-out partition.
Keep all clips/classes from each source video together for CholecSeg8k.
Additional datasets/backbones remain planned or partial unless the results report
explicitly marks them complete.

| Experiment | Question and method |
|---|---|
| E0 — Exactness | Check detached/empty/zero-bias no-ops, restored-state replay, intervention equivalence and attention reconstruction |
| E1 — Pilot | Test feasibility and the registered decision gate on development sequences |
| E2 — Regional effects | Remove regional keys versus matched random controls; evaluate single-frame replays and separate persistent rollouts |
| E3 — Corruption | Compare clean GT spatial memory, c1/c2 controlled corruption and natural predicted memory |
| E4 — Conditions | Relate regional effects to measured imaging/motion conditions; implementations are specified in the protocols |
| E5 — Agreement | Compare regional attention allocation with measured segmentation effects, including query-region endpoints where available |
| E6 — Method | Evaluate a justified, development-selected strategy against upstream baselines and controls; no validated learned-method gain is established |

The governing definitions are in [protocols](../protocols/):
`region_tokens_v1_1.json`, the Cholec data/split amendments, review amendment
and `region_tokens_v2_amendment.json`. Later amendments are explicitly
post-interim, not retrospectively outcome-blind preregistration. Frozen ancestor
protocols remain for referenced partitions and provenance; the latent-utility
study is inactive.

## 4. Measurements and controls

### Segmentation contribution

For a region g, removal harm is:

`H(g) = score(baseline) − score(region removed)`

Positive H means removal hurts. Excess harm subtracts the mean random-control
harm at the corresponding budget. Apply exclusions as additive −∞ key masks;
preserve key layout, RoPE positions and native pointers for spatial-only tests.
Restore the same cached state/features before each target-frame replay and
do not write the altered prediction back.

Persistent rollouts are separate experiments: changed predictions update later
memory, so their effects are non-additive and their realized supports can differ.
The seq13/14 failure-selected rollouts are descriptive, not population evidence.
Sampled E3 re-encodes only the selected spatial bank, retaining natural pointers
and other history; it is not accumulated corruption throughout a whole video.

### Geometry controls

The original random controls share a boundary-referenced budget. Only the
boundary contrast is distance-matched; other regions cannot be directly ranked
as if proximity were controlled.

The follow-up instead matches each region to its own **disjoint** control by
memory frame and exact squared token-centre distance. Mixed/pure comparisons
also match inside/outside. Cap the sampled support at 64 keys; unsupported
strata remain skipped. This estimates effects on matched subsets, not entire regions.

### Attention and reweighting

Observe native post-RoPE attention in 32-query chunks and return the unchanged
native output. Aggregate over native queries/heads and report layers, spatial
regions and pointers separately. Normalize mass by support size before
interpreting enrichment. Saliency alone is not a causal contribution measure.

The regional headroom diagnostic tests four regions × ±1 logit biases against
equal-search, within-frame permuted nulls: eight candidates, three null seeds,
with baseline selectable in either search. GT selects the winner; these are
**oracle searches, not deployable gains**. The completed follow-up did not meet
the method gate; no training was launched.

### Statistics

Average seeds/targets within sequence before inference; the sequence is the
independent unit. Use paired Wilcoxon comparisons, 10,000 whole-sequence bootstrap
resamples and the declared endpoint-specific Holm families. Report feasible
support, missing/skipped rows and unadjusted CI status. Do not pool repeated
validation rows or different estimators.

## 5. Efficient-SAM2 reuse and MPS validation

Official submodule: `external/Efficient-SAM2`, revision
`8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5`.

SMR uses the author's attention-ranked token selection and cached retrieval
patterns. It does not explicitly separate foreground, boundary and background.
Region labels are **measurement only** and never change its selection.

| Setting | Implementation |
|---|---|
| Sparse arm | Native SMR, 95% pruning of eligible older-frame tokens |
| Matched dense control | Same SMR classes with 0% pruning, avoiding a different attention kernel |
| Image encoder | SWR/bypass disabled; no encoder blocks replaced |
| MPS port | Two explicit CUDA allocations changed to predictor/feature device |
| Precision/storage | FP32 inference without autocast; native bf16 memory storage; state on MPS, frames on CPU |
| Observation | Native retained indices, per-region counts and image overlays |
| Selection detail | Pinned author code averages queries from the first head of batch 0; not replaced with an all-head mean |

The exact device patch is [efficient-sam2-mps.patch](../patches/efficient-sam2-mps.patch).
After initializing the submodule, apply it with:

```sh
git -C external/Efficient-SAM2 apply --unidiff-zero ../../patches/efficient-sam2-mps.patch
```

Bounded seq3 smokes execute and record actual sparse reads, but **strict
repeat/no-op validation failed**. Dense repeat maximum frame-Dice differences
were 0.00809, 0.00571 with deterministic algorithms, and 0.00517 with explicit
Metal preference. The cause remains unresolved; none is an accepted method gain.
Small-tensor CPU–MPS RoPE tests passing does not establish full-model equivalence.
Full runs require a passing same-device `--validation-dir`; no full
Efficient-SAM2 experiment has been launched.

Overlays show original image | mask-defined regions | native retained positions.
Colors: boundary gold, foreground green, near background blue, far background
purple. The final frame's new selection is unavailable after upstream cache
clearing and is not reconstructed. MPS timings include measurement/I/O and are
not the paper's CUDA speed benchmark.

## 6. Validity, implementation and recording

| Check | Required interpretation |
|---|---|
| Chronology | Renumbered mirrors require content-bound source-frame mapping; sampled Kaggle mirror ordering was inconsistent, so those temporal results are quarantined |
| Cached-state replay | Synchronize accelerator offloads before host-side snapshots; compare logits at the same native resolution |
| Attention reconstruction | Preserve the declared threshold; rejected seq15 attention remains excluded despite its completed segmentation-only retry |
| Coverage | A completed smoke, partial shard or successful orchestration does not complete the study |

| Component | Code |
|---|---|
| Upstream predictor loading | `modeling/medsam2.py` |
| Frozen intervention controls / corruption | `modeling/memory_attention.py`, `modeling/memory_interventions.py` |
| Regions, matching, attention observation | `modeling/region_tokens.py` |
| E0/E1 and E2–E6 execution | `scripts/run_region_token_pilot.py`, `scripts/run_region_experiments.py` |
| Local MPS execution / aggregation | `scripts/run_region_v2_local.py`, `scripts/summarize_region_v2_local.py` |
| Upstream baseline / Efficient-SAM2 adapter | `scripts/run_region_baseline.py`, `scripts/efficient_sam2_adapter.py` |
| Condition analysis / statistics | `scripts/measure_region_conditions.py`, `scripts/analyze_region_experiments.py`, `evaluation/statistics.py` |
| Chronology audit / delivery | `scripts/audit_polypgen_alignment.py`, `scripts/collect_region_results.py` |
| Kaggle / tracking | `kaggle/region-tokens/runner.py`, `training/tracking.py` |

Inspect entry points with `--help`. Verify project tests and lint with:

```sh
python -m pytest -q
python -m ruff check modeling evaluation scripts tests training datasets
```

Unit tests do not replace runtime validation. Use separate outputs for changed
definitions, controls and revisions. Require online Forge W&B tracking and
artifact delivery; Kaggle needs its attached `WANDB_API_KEY` secret. Pinned
jobs do not acquire subsequent local edits.

Outputs remain under the configured Drive `outputs/16:10:2026/` directory.
Its desktop-sync path still uses local disk; filesystem presence does not prove
cloud synchronization. Current evidence links and completion tables are in
[REGION_TOKENS_RESULTS.md](REGION_TOKENS_RESULTS.md). The MPS audit is
`efficient_sam2_mps_port_validation/validation.json` under that output root.

# Regional memory tokens: final results

Completed **2026-10-24**. PolypGen: **23 source sequences, 21 promptable
sequences, 2,012 post-prompt targets**. Seq1/seq7 are unpromptable and reported
as N/A, not assigned fabricated Dice scores. CPU FP32; first-visible-frame mask
prompt, then native propagation. Results are exploratory, not held-out validation.
See [methods](REGION_TOKENS.md).

Separate [post-hoc analyses](REGION_POSTHOC_RESULTS.md) report equivalence,
GT-empty targets, query/layer attention and size; they do not replace the frozen
final analyses below.

## Segmentation benchmarks

Equal-sequence present-frame mean Dice on the promptable cohort.

| Method | Dice |
|---|---:|
| Native SAM2 | 0.71904 |
| SAMURAI | 0.76652 |
| SAM2Long | 0.74253 |
| Lean-SAM2 | 0.73446 |

None of the recorded author-versus-SAM2 comparisons passes Holm correction.
SAM2Long uses offline final-path selection, not equivalent online information.
These benchmarks do not measure a new region-aware method.

## Longer-history groundwork: E0/E1 (exploratory)

Protocol and implementation committed as `8d22f5a` before computation; no new
inference. Original archived predictions reproduce Dice exactly on 2,012
targets per method (1,689 present, 323 GT-empty). These E0/E1 identifiers belong
to the new longer-history proposal, not historical experiment numbering.

### E0 — where present-frame Dice is lost

Equal-sequence means. Each deficit column is in Dice units; the four columns
sum to 1−Dice. Empty prediction takes priority; other errors are partitioned by
the frozen operational contour-tolerance band and error direction.

| Method | Dice | Empty-miss deficit | Contour-band deficit | Non-band FN deficit | Non-band FP deficit | Empty-miss share of deficit |
|---|---:|---:|---:|---:|---:|---:|
| SAM2 | 0.71904 | 0.09968 | 0.09860 | 0.01883 | 0.06386 | 35.5% |
| SAMURAI | 0.76652 | 0.03589 | 0.10979 | 0.01652 | 0.07128 | 15.4% |

| Method | Empty misses / present targets | Nonempty zero-overlap predictions | Empty misses >6 sampled steps after prompt | False alarms / GT-empty targets |
|---|---:|---:|---:|---:|
| SAM2 | 126 / 1,689 | 20 | 122 / 126 | 163 / 323 |
| SAMURAI | 28 / 1,689 | 22 | 26 / 28 | 217 / 323 |

The frozen pragmatic screen passes for SAM2: miss-deficit share 35.5% ≥20%,
and 20 sequences have ≥14 post-prompt targets (required ≥11). Across the
manifest: 51 observed disappearance/return episodes; source-sequence lengths
15–250 sampled frames. This supports investigating history, **not** claiming
longer memory works. Most exposure is late; late-miss counts alone do not show
increasing risk, and prompt conditioning persists. Geometry proxies are not
semantic appearance measurements; binary masks do not identify mucosa leakage.

The contour band is explicitly asymmetric: published inside-only Boundary IoU
band dilated outward/inward by the same radius; tolerance extends approximately
two radii inward and one outward. A wording-only clarification removed
“symmetric”; the frozen formula and results did not change.

### E1 — spatial-key budget

Complete: **2,033 stored frames**, including 21 prompts. Native-logit capture
hashes and unchanged author occupancy match for every frame. Recovered geometry
archive masks match all **2,012 native SAM2 target masks** exactly.

| Equal-sequence per-frame mean | Tokens |
|---|---:|
| Foreground interior | 312.7 |
| Boundary | 72.2 |
| Near background: fixed one-token ring | 87.9 |
| Far background | 3,623.3 |
| Retained: interior + boundary + near background | 472.7 |

| Finite-history budget check | Result |
|---|---:|
| Full prompt/latest spatial keys reserved | 8,192 |
| Older sparse-frame spatial-key budget | 20,480 |
| Mean older frames fitting actual history, equal-sequence mean | 30.8 |
| Evaluated contexts with distinct prompt/latest frames | 1,991 |
| Contexts fitting >5 older frames | 1,861 / 1,991 |
| Stored frames retaining zero tokens | 286 / 2,033 |
| Maximum verified spatial read keys | 28,672 |

The first post-prompt target has a shared prompt/latest frame and is excluded
from the distinct-frame packing summary. Empty frames can cost zero selected
spatial tokens, so 30.8 is not a count of informative memories. A fixed global
K is not justified: retained occupancy varies from 0 to 3,685 tokens/frame.
Pointer keys, temporal encoding, stored-memory bytes and actual sparse compute
remain outside this audit. Earlier exclusions make discarding background a
scientific risk, not an established safe compression rule. E0/E1 did not launch
downstream inference; E2 is now a separate engineering gate below.

Evidence under `21_region_sparse_E0_E1_v1`: `E0/E0_results.json` and
`verified_artifact_audit/E1/E1_results.json`, with all sequence/frame counts and
an independent budget/source verification receipt. The initial cloud-only E1
read stalled; preserved W&B geometry artifacts were recovered into Drive,
checksum-verified and matched to native predictions instead. No inference.
[E0 W&B run](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/6q892hmk).
[E1 W&B run](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/94w1uj1r).
Both result artifacts finished uploading. This does not independently verify
Drive cloud synchronization.

### E2 — sparse-read correctness gate

Implementation and [protocol](../protocols/sparse_memory_E2_v1.json) were committed
as `9e9ff87` before the complete seq2/seq3 CPU FP32 gate launched. **198 tests
passed at E2 launch**, including all-token bit parity and retained original
RoPE positions. **The complete real-data E2 gate passed**, including fresh
sparse replay, unchanged banks/caches, all four layers and 118 reads containing
older nonconditioning history. Result artifacts finished uploading.

The integration executes Efficient-SAM2's gather statements and pruned RoPE
helper unchanged, retaining native SAM2 SDPA. This is an adaptation, not an
unchanged full Efficient-SAM2 model. Old-frame temporal embeddings saturate at
the oldest native nonconditioning slot; prompt/latest and pointer handling
remain native. Gate conditions: native, all-token native assembly, all-token
rebuilt assembly, native-horizon regional retrieval, and regional long retrieval
(at most 32 frames, 28,672 spatial keys). Any integrity failure stops progression.
E4 is explicitly deferred by the user (2026-10-25); no suitable untouched
external dataset has been selected. No E4 search or experiment is running.

[E2 tracking](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/8v90x8we).
Evidence: `22_sparse_memory_E2_v1/gate`. This is an engineering pass, not a
segmentation improvement or held-out validation.

### E3 — first-stage C0/C1 screen

The separate [protocol](../protocols/long_history_E3_screen_v1.json) fixes a
32-frame full-memory C1 control versus fresh native C0. The C1 engineering
gate and **full 21-sequence C0/C1 screen completed**; artifacts finished
uploading. Seq1/seq7 remain N/A. E2's validated source is unchanged.

| Condition | Equal-sequence present-frame Dice |
|---|---:|
| C0 native SAM2 | 0.71904 |
| C1 full-frame longer memory | 0.69114 |
| C1−C0 | −0.02789 |

Paired mean difference 95% sequence-bootstrap CI **[−0.11006, +0.05235]**;
Wilcoxon **p=0.51966**. The numerical decrease is not statistically significant.
The original positive history-alone screen **did not pass**. This does not
establish that longer history is intrinsically harmful or refute selective
regional history.

Evidence: `23_long_history_E3_screen_v1/results.json`;
[C0/C1 tracking](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/khwf1wjm).

### E3 — C2/C3 exploratory amendment

The user approved continuing **after the negative C1 screen**. A separate
[dated amendment](../protocols/region_sparse_C2_C3_amendment_v1.json) overrides
that stopping rule without rewriting the original design. PolypGen remains
development/exploratory data; this is not held-out confirmation.

| Condition | Fixed strategy |
|---|---|
| C2 | Validated E2 region-long reader; full prompt/latest, older interior + boundary + one-token near-background ring |
| C3 | C2's exact frame IDs and per-frame counts; random token positions, three fixed seeds, reported as a within-sequence seed average |

Both use the existing 32-frame cap, 28,672 spatial-key budget and saturated
native temporal slots. Pointer values remain each arm's own native outputs.
C3 is a reference-trajectory-conditioned ablation, not a standalone deployable
baseline. Only past C2 counts determine selection; postprompt GT never does.

Complete seq2/seq3 fresh-model replay gates precede full-cohort inference.
**218 tests passed** before the new gate launch, including exact count/RNG
checks, allocation mismatch failures, reference integrity and seed-mean statistics.
Primary contrasts: **C2−C0** and **C2−mean(C3 seeds)**, sequence-level bootstrap
intervals and Wilcoxon with Holm2 correction. Save all seeds, false alarms,
censored re-detection, actual read-key work and timing. No weights, widths or
memory limits are tuned. **No C2/C3 efficacy result is available yet.** C4 and
SAMURAI+C2 are not launched; E4 stays deferred.

## Memory-attention allocation

Four-layer/all-query summary: exponential of the median sequence log pooled
enrichment, N=21 per row. Enrichment above 1 means more attention per token than
the spatial-key average, not greater total mass or causal usefulness. Memory
regions use native predicted masks; GT-reference query grouping is offline only.

| Definition | Background | Foreground interior | Boundary |
|---|---:|---:|---:|
| Mixed cells (primary) | 0.7020 | 3.5594 | 9.3387 |
| Transfiner incoherence | 0.6614 | 2.5654 | 7.2809 |
| Trimap k15 | 0.6899 | 2.9481 | 7.6301 |
| Trimap k21 | 0.6663 | 2.5813 | 6.9082 |
| Trimap k29 | 0.6521 | 2.3544 | 6.0568 |
| Inner band 0.02 | 0.7020 | 2.0493 | 7.0742 |

Boundary enrichment exceeds foreground enrichment under all six definitions.

## Contiguity-matched exclusions

Completed: **21 sequences, 2,012 targets, 36,216 definition–region cases**.
One-target exclusions preserve the native memory trajectory and pointer keys.
Three rigid-control seeds match excluded count, shape and connectivity within
each memory frame, but not anatomical position or border proximity.

Primary analysis uses informative present-GT targets: average within sequence,
then equally across sequences. Excess loss = control Dice − excluded Dice.
Positive values indicate more damage from regional exclusion than its controls.
Intervals are 95% whole-sequence bootstrap intervals (10,000 replicates), not
simultaneous intervals. Paired Wilcoxon tests use Holm correction over 18 tests.

| Primary mixed-cell region | Native Dice | Excluded Dice | Control Dice | Excess loss [95% CI] | Holm18 p |
|---|---:|---:|---:|---|---:|
| Background | 0.7190 | 0.2696 | 0.5675 | +0.2979 [0.2134, 0.3771] | 0.000172 |
| Foreground interior | 0.7190 | 0.6763 | 0.7209 | +0.0445 [0.0261, 0.0643] | 0.007189 |
| Boundary | 0.7190 | 0.7199 | 0.7194 | −0.0005 [−0.0051, 0.0046] | 1.000000 |

### Robustness across definitions

Each cell: mean excess Dice loss [95% CI]; Holm18 p. N=21 for each comparison.
Eligible frames differ when a region or its controls are uninformative; some
foreground comparator baselines are approximately 0.7277, not 0.7190. Each
intervention is compared with its own matched baseline and controls.

| Definition | Background | Foreground interior | Boundary |
|---|---|---|---|
| Mixed cells | +0.2979 [0.2134, 0.3771]; p=0.000172 | +0.0445 [0.0261, 0.0643]; p=0.007189 | −0.0005 [−0.0051, 0.0046]; p=1.0000 |
| Transfiner incoherence | +0.1647 [0.0798, 0.2441]; p=0.012774 | +0.0157 [0.0033, 0.0300]; p=0.394234 | −0.0009 [−0.0114, 0.0107]; p=1.0000 |
| Trimap k15 | +0.1840 [0.0872, 0.2749]; p=0.016462 | +0.0229 [0.0100, 0.0370]; p=0.036671 | +0.0016 [−0.0059, 0.0096]; p=1.0000 |
| Trimap k21 | +0.1584 [0.0637, 0.2498]; p=0.039345 | +0.0137 [0.0031, 0.0257]; p=0.365613 | +0.0023 [−0.0065, 0.0134]; p=1.0000 |
| Trimap k29 | +0.1115 [0.0293, 0.1874]; p=0.173585 | +0.0086 [−0.0001, 0.0181]; p=0.417776 | +0.0016 [−0.0103, 0.0156]; p=1.0000 |
| Inner band 0.02 | +0.2979 [0.2134, 0.3771]; p=0.000172 | +0.0057 [−0.0015, 0.0135]; p=0.818213 | +0.0130 [−0.0037, 0.0321]; p=0.818213 |

**Frozen claim rule:** positive mean excess loss, positive CI lower bound and
Holm p<0.05 under all six definitions. **No region passes this rule.**

## Interpretation

- **Attention allocation differs by region.** Boundary positions receive the
  highest attention per token, consistently across definitions.
- **High attention is not demonstrated indispensability.** Boundary exclusion
  has no significant excess Dice loss under any definition. This is not an
  equivalence test or proof that boundary information is unnecessary.
- **Primary regional effects are definition-dependent.** Background and
  foreground exclusions are significant for mixed cells. Background excess loss
  is positive throughout, but k29 fails the adjusted test; foreground evidence
  weakens for wider/comparator definitions. Neither meets the all-definition rule.
- **These are positional interventions.** Contextual token vectors are not
  semantically pure. Removing background also determines the small complement
  retained, so its large effect does not establish intrinsic semantic importance.
- **No deployment gain is established.** Pointer attention can renormalize;
  redundancy and compensation remain possible, untested explanations. One-step
  exclusions are not modified full-video propagation or evidence that regional
  weighting improves Dice. Same-cohort analysis does not establish generalization.

## Final evidence

| Result | Folder under Drive output root |
|---|---|
| Author segmentation benchmarks | `08_region_readout_polypgen_v1/serial_release_v1` |
| Per-token attention and secondary strata | `16_per_token_attention_polypgen_cpu_v1/full` |
| Exclusions, eligibility and size/outcome strata | `17_bank_integrity_repair_v2/continuation/full` |

[Drive output root](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment>).
Final summaries and W&B artifact delivery are confirmed; independent Drive
cloud synchronization is not verified. Detailed provenance remains with the
saved evidence, outside this final-results summary.

[Attention run](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/gqscihxk)
· [Exclusion run](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/p42rakqp).

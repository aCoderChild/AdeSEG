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

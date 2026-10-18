# Protocols

These are scientific records, not interchangeable runtime configurations.
Keep frozen filenames and contents: pinned notebooks, saved hashes and descendant
protocols depend on them. Later amendments do not retroactively preregister results.

| Record | Purpose |
|---|---|
| `region_tokens_v1.json` → `region_tokens_v1_1.json` | Region-study definitions and revised primary protocol |
| `region_tokens_v1_2_cholec_amendment.json` → `region_tokens_v1_3_cholec_split.json` | Cholec data and source-video split amendments |
| `region_tokens_execution_v1.json` | Execution settings and validation requirements |
| `region_tokens_restore_control_v2.json` | Memory-read restoration diagnostic amendment |
| `region_tokens_review_v1.json` | Review/sensitivity controls |
| `region_tokens_v2_amendment.json` | Post-interim local chronological diagnostics, geometry controls and headroom gate |
| `cutie_development_comparator.json` | External comparator and correction-budget accounting |
| `latent_utility_v2.json`, `utility_retrieval_v1.json` | Inactive ancestors retained solely for provenance and the referenced DAVIS split; not the active hypothesis |

The DAVIS manifest builder, analysis and Kaggle runner still read the explicit
partition in `utility_retrieval_v1.json`. Deleting it would break the region
workflow; its old model/method claims are not adopted by the current study.

See [methods](../docs/REGION_TOKENS.md) for the active idea and
[results](../docs/REGION_TOKENS_RESULTS.md) for completion and validity limits.

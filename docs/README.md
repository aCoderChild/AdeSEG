# Regional memory-token study

Study how foreground-interior, boundary-associated and background memory tokens
relate to SAM2 attention and endoscopic segmentation.

- [Methods and workflow](REGION_TOKENS.md)
- [Results and limitations](REGION_TOKENS_RESULTS.md)
- [Current audit protocol](../protocols/region_definition_verification.json)
- [Citation, size bins and post-launch verification record](../protocols/region_definition_audit_v1_record.json)
- [Exploratory trimap-width amendment](../protocols/region_definition_trimap_amendment_2026_10_23.json)
- [Per-token attention rerun and fixed summary protocol](../protocols/token_attention_rerun_v1.json)
- [Contiguity-matched exclusion protocol](../protocols/contiguous_exclusions_v1.json)
- [Historical experiment protocol](../protocols/memory_token_regions.json)
- [Scripts](../scripts/README.md)
- [Frozen offline E0/E1 failure and spatial-budget audit](../protocols/region_sparse_e0_e1_v1.json)

Updated 2026-10-24:

| Stage | Status |
|---|---|
| Geometry audit | Complete |
| Per-token attention | Complete: 21 promptable sequences, 2,012 targets |
| Contiguity-matched exclusions | Complete: 21 sequences, 2,012 targets; see final results |
| Longer-history groundwork | E0/E1 complete and verified |
| Sparse-read engineering gate | E2 passed: full native parity, sparse replay and 118 older-history reads |
| Exploratory history screen | C0/C1 complete; longer dense memory did not pass the positive screen |
| Regional history amendment | C2/C3 complete; mean Dice 0.73838 / 0.70908; neither primary contrast significant |
| Author + regional history | SAMURAI/SAM2Long integrations prepared; author-specific gates required before full runs; E4 deferred |

Boundary positions receive high attention per token; causal usefulness and
segmentation improvement are not established. Historical regional results use
a different assignment. Claims concern positions, not exclusive vector contents.

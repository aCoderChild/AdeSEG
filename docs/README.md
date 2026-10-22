# Regional memory-token study

Study how foreground-interior, boundary-associated and background memory tokens
relate to SAM2 attention and endoscopic segmentation.

- [Methods and workflow](REGION_TOKENS.md)
- [Results and limitations](REGION_TOKENS_RESULTS.md)
- [Current audit protocol](../protocols/region_definition_verification.json)
- [Citation, size bins and post-launch verification record](../protocols/region_definition_audit_v1_record.json)
- [Exploratory trimap-width amendment](../protocols/region_definition_trimap_amendment_2026_10_23.json)
- [Per-token attention rerun and fixed summary protocol](../protocols/token_attention_rerun_v1.json)
- [Historical experiment protocol](../protocols/memory_token_regions.json)
- [Scripts](../scripts/README.md)

Current audit: native predicted-mask mixed cells, with author incoherence,
trimap and inner-band comparators. The seq2/seq3 gate passed; full-cohort
results remain pending. Historical regional results use a different assignment.
Claims concern token positions, not exclusive semantic contents.

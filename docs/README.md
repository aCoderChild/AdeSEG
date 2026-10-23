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

Updated 2026-10-23:

| Stage | Status |
|---|---|
| Geometry audit | Complete |
| Per-token attention | Complete: 21 promptable sequences, 2,012 targets |
| Contiguity-matched exclusions | Both diagnoses and expanded repair gate passed; full run resumed from seq12; nine completed sequences preserved; original cause unresolved |

Boundary positions receive high attention per token; causal usefulness and
segmentation improvement are not established. Historical regional results use
a different assignment. Claims concern positions, not exclusive vector contents.

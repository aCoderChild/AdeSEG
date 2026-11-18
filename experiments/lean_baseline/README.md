# Lean-SAM2 baseline and 21-sequence replication

Entry points: `scripts/run_author_baseline.py`, `scripts/run_lean_polypgen_full.py`, `scripts/audit_lean_polypgen_results.py`. The audited CPU/FP32 GT-mask-prompt result was mean present Dice 0.73446 on 21 eligible sequences; native SAM2.1 Base+ was 0.71904. The paired sequence bootstrap 95% CI for Lean - native included zero. Save frame IDs, GT-empty false alarms, submodule commit and checkpoint SHA with each report. [Original run](https://drive.google.com/drive/folders/1DmkKRrGmEcJQUp0Pb0fp5u5CS0DSo0B5).

Separate automatic improvements: the indexed-RoPE script/protocol in `scripts/run_lean_indexed_spatial_rope.py` and `protocols/lean_indexed_spatial_rope_v1.json` has a matched automatic-prompt result 0.7353 (all-frame 0.7318), *not* the unverified 0.742 claim.

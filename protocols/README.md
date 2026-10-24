# Current protocol

[full_propagation_false_alarm_v1.json](full_propagation_false_alarm_v1.json)
fixes the post-hoc native SAM2/SAMURAI/foreground-gated SAMURAI propagation
comparison, causal previous-score rule and protection/replay requirements.
Implementation does not mean the gate or full experiment has run.

[memory_token_regions.json](memory_token_regions.json) consolidates the
verified checkpoint, initialization and three-region spatial assignment.
Predicted masks supply primary labels; GT assignments are separate audits. It is not a fresh preregistration and does not authorize
claims of full-study completion or segmentation improvement.

Retired experiment protocols were removed with the superseded code/results,
as requested. The original PolypGen data partition remains under
`datasets/splits/polypgen/`; no source dataset or checkpoint was deleted.

The planned author-default SAM2/SAM2Long benchmark and controlled
author-method versus regional-reading comparison are specified in
[REGION_TOKENS.md](../docs/REGION_TOKENS.md#next-study-frame-selection-versus-regional-reading).
They are not launched or validated by the existing attention/exclusion runs.
Stride 2/4 were subsequently excluded. The separate matched-FP32 CPU batch
validates all four supported author adapters before full inference; SMR-only
remains blocked pending CUDA verification. This is not a completed GPU study.

# Tests

Run `python -m pytest -q`. Discovery is restricted to this folder.

| File | Coverage |
|---|---|
| `test_memory_token_regions.py` | Boundary IoU integration, three-region geometry, exact vector grouping |
| `test_region_definitions.py` | Native-grid mixed-cell coverage, unchanged author incoherence, thin-cell sampling counterexample and deterministic two-sided trimaps |
| `test_region_definition_statistics.py` | Aligned/mixed contour coverage, image-border convention, undefined denominators and quartile ties |
| `test_trimap_amendment.py` | Amended widths, unchanged original configuration and shared k15 labels; rectangle contour-coverage sanity check |
| `test_region_attention.py` | Efficient-SAM2 author provenance, native output preservation, chunking, pointers and overlay-pixel verification |
| `test_token_attention.py` | Per-key/query mass against dense weights, exact output preservation, empty groups, pooling, precision and stored-sequence aggregation |
| `test_author_region_measurement.py` | Author candidate-axis selection, native attention-output preservation, hook restoration and exact baseline prediction checks |
| `test_region_exclusion.py` | Native SDPA masks, matched random keys, cached-state isolation, exact buffer digests and excess-loss calculation |
| `test_contiguous_exclusion.py` | Rigid null geometry/counts, invariant masks, pointers, exact zero masks, state isolation, separate replay/buffer failures and evidence-validated resume |
| `test_memory_token_io.py` | Native capture, prompt handling, CLI imports, protocol and output locations |
| `test_full_memory_region_audit.py` | Full coverage, conditioning frames, storage and artifact delivery |
| `test_data_pipeline.py` | Mask decoding, frame staging and local/Kaggle chronology |
| `test_evaluation.py` | Retained segmentation metrics and sequence statistics |
| `test_tracking.py` | Credentials and W&B tracking |

Tests for deleted historical workflows were retired with their implementations.
A lower test count is not evidence of stronger or weaker scientific results.
Unit tests do not establish semantic feature purity, causal attention
utility, segmentation gains or cross-device numerical equivalence.

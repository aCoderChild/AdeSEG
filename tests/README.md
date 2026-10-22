# Tests

Run `python -m pytest -q`. Discovery is restricted to this folder.

| File | Coverage |
|---|---|
| `test_memory_token_regions.py` | Boundary IoU integration, three-region geometry, exact vector grouping |
| `test_region_attention.py` | Efficient-SAM2 author provenance, native output preservation, chunking, pointers and overlay-pixel verification |
| `test_author_region_measurement.py` | Author candidate-axis selection, native attention-output preservation, hook restoration and exact baseline prediction checks |
| `test_region_exclusion.py` | Native SDPA masks, matched random keys, cached-state isolation, exact buffer digests and excess-loss calculation |
| `test_memory_token_io.py` | Native capture, prompt handling, CLI imports, protocol and output locations |
| `test_full_memory_region_audit.py` | Full coverage, conditioning frames, storage and artifact delivery |
| `test_data_pipeline.py` | Mask decoding, frame staging and local/Kaggle chronology |
| `test_evaluation.py` | Retained segmentation metrics and sequence statistics |
| `test_tracking.py` | Credentials and W&B tracking |

Tests for deleted historical workflows were retired with their implementations.
A lower test count is not evidence of stronger or weaker scientific results.
Unit tests do not establish semantic feature purity, causal attention
utility, segmentation gains or cross-device numerical equivalence.

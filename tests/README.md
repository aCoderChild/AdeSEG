# Regression tests

Run `python -m pytest -q` from the repository root. `pytest.ini` limits discovery
to this folder, excluding upstream repositories, datasets and environments.

| Area | Test files |
|---|---|
| Region supports, matching and rollout controls | `test_region_tokens.py`, `test_region_v2.py`, `test_region_review.py` |
| Attention and interventions | `test_memory_attention.py`, `test_memory_interventions.py` |
| Experiment initialization and execution | `test_region_initialization.py`, `test_region_experiments.py` |
| Upstream adapters and historical comparators | `test_efficient_sam2.py`, `test_cutie_baseline.py`, `test_region_boxes.py`, `test_local_region_boxes.py` |
| Data order, decoding and provenance | `test_data_pipeline.py`, `test_polypgen_order.py`, `test_protocol_provenance.py`, `test_kaggle_dataset_roots.py` |
| Statistics and metrics | `test_statistics.py`, `test_davis_metrics.py` |
| Tracking, delivery and monitoring | `test_tracking.py`, `test_region_delivery.py`, `test_kaggle_command_monitor.py` |

Unit tests exercise contracts with small fixtures; they do not reproduce full
model experiments or certify MPS/CUDA equivalence. Runtime exactness gates and
scientific-result acceptance remain separate. Obsolete latent-utility tests are removed.

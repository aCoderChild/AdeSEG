# Tests

Focused CPU tests check the three-fold split, source-bound registry, routing,
freeze-before-score behavior and model code. A representative set is:

```sh
python -m pytest -q tests/test_threefold_training_contract_v1.py \
  tests/test_preflight_cache_release_crossfit_training_3fold_v1.py \
  tests/test_build_cache_release_crossfit_registry_3fold_v1.py \
  tests/test_bounded_crossfit_3fold_runner_v1.py \
  tests/test_bounded_crossfit_prediction_score_3fold_v1.py \
  tests/test_bounded_crossfit_shared_worker_3fold_v1.py
```

CPU tests use synthetic fixtures and do not establish MPS performance or a
segmentation improvement. Real runs have separate MPS probes, label guards,
resource checks and independent metric audits.

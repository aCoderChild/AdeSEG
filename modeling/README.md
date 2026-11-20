# Internal model implementations

The **public research API is `adeseg/`**. The `modeling/` directory
is its internal implementation layer, **not a second framework or a separate
collection of proposed contributions**.

## Where to work

| Scientific concern | Public entrypoint | Required implementation |
|---|---|---|
| FGPI | `adeseg.modules.FGPI` | `adeseg/prompts/` |
| RMFA | `adeseg.modules.RMFA` | `modeling/structured_feature_aggregation.py`, `modeling/aggregation_blocks/`, `modeling/multiscale_channel_value_refiner_v1.py` |
| TTMR | `adeseg.modules.TTMR` | `adeseg/memory/target_aware_temporal.py` |
| Native MedSAM2 | `adeseg.backbones.medsam2` | `modeling/medsam2.py`, `modeling/medsam2_latest.py` |
| Medical high-resolution adapter | `adeseg.adapters.high_resolution_skip` | `modeling/medical_highres/` |
| Lean memory hooks | `adeseg.inference.lean_memory_routing` | `modeling/lean_channel_value_routing_v1.py` and Lean author package |

## Why not physically merge these files into `adeseg/`?

Existing training artifacts verify implementation bytes. Specifically:

- `modeling/multiscale_channel_value_refiner_v1.py` is authenticated by
  its own source SHA-256 in `load_trained_state`.
- `modeling/lean_implementation_digest.py` authenticates the structured
  RMFA implementation, all four `aggregation_blocks` files, and routing.
- `modeling/medical_highres/checkpoints.py` authenticates the medical
  adapter's implementation files.
- `training/training_provenance.py` hashes the training pipeline; relocating
  source without controlled migration invalidates old provenance records.
- Tests and author-model runner imports still refer to several `modeling.*`
  paths. A filesystem move without compatibility shims breaks these users.

Thus, **the two directories are integrated at the Python API level**:
`adeseg` names and owns the scientific components, and `modeling` retains
the exact source/checkpoint-compatible parameter implementation. No model
weights are duplicated.

A future physical consolidation is a *separately versioned checkpoint
migration* requiring re-sealed checkpoints and exact forward/gradient parity;
it should not be mixed into an imminent experimental run.

The old `legacy/` directory contained only a historical file-location
index, with no executable code. The archive branch already preserves the
historical experiments, so its redundant index has been removed from `main`.

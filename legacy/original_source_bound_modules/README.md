# Original source-bound implementations (frozen in place)

The path of every historical implementation below is **intentionally
unchanged**. This directory is an index rather than a second source tree.

| Current original source | Replacement public namespace |
|---|---|
| `modeling/multiscale_channel_value_refiner_v1.py` | `adeseg.adapters.multiscale_channel_attention` |
| `modeling/structured_feature_aggregation.py` | `adeseg.adapters.multiscale_channel_attention` |
| `modeling/medical_highres/adapter.py` | `adeseg.adapters.high_resolution_skip` |
| `modeling/medical_highres/teacher_loss.py` | `adeseg.losses.teacher_kl` and `segmentation` |
| `modeling/positive_teacher_preservation_loss.py` | `adeseg.losses.foreground_preservation` |
| `modeling/lean_channel_value_routing_v1.py` | `adeseg.inference.lean_memory_routing` |
| `modeling/medsam2_latest.py` | `adeseg.backbones.medsam2` |
| `modeling/lean_indexed_spatial_rope_mps.py` | `adeseg.memory.spatial_position_encoding` |
| `modeling/lean_multiscale_anchor.py` | `adeseg.memory.temporal_anchors` |
| `evaluation/segmentation.py` | `adeseg.evaluation.segmentation_metrics` |
| `evaluation/temporal.py` | `adeseg.evaluation.temporal_metrics` |

Some region/memory controls required by retained comparators remain at their
legacy paths. Other exploratory code, including the test-sequence-specific
crop and unsupported automatic-memory runners, is preserved **only in the
full experiment backup**. These are not public model components.

Do not copy their source into the active adapters, silently rename it, or
treat an old exploratory result as an external holdout measurement.
The full historical experiment branch remains unchanged at
`backup/experiment-full-20261118`.

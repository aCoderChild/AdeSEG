# Project models

The current contribution candidates refine SAM2 mask logits while leaving the
Lean memory-pruning rule in place.

- [Foreground-conditioned boundary refiner](foreground_conditioned_refiner.py):
  early/deep features, soft foreground–background contrast, channel reweighting
  and a residual mask head. [Routing](lean_boundary_refinement_routing_v1.py)
  controls output and memory-encoder mask input separately.
- [Multiscale channel-value refiner](multiscale_channel_value_refiner_v1.py):
  learned cross-channel value aggregation and region contrast. Its
  [routing](lean_channel_value_routing_v1.py) is separate.

See [methods and evidence](../docs/METHODS.md) for the exact experiments and
limitations. Other modules in this directory are dependencies or historical
research; their presence does not imply they are part of the current method.
Sources used by frozen experiments must retain their paths and bytes.

## Finding the current pipeline

- `lean_boundary_window_runtime_v1.py` and
  `lean_channel_value_window_runtime_v1.py` connect the refiners to video
  propagation. The corresponding `lean_*_routing_v1.py` files select the
  output and memory-input routes.
- `lean_channel_feature_capture.py` and related capture modules collect the
  features used in the channel-value study. Training and study entry points
  are catalogued by `python -m workflows --list`.
- `seq23_viewport.py` handles the sequence 23 inset. Baseline-specific adapters
  include `samurai_auto_prompt.py` and `samurai_mps_kalman.py`.

Historical modules remain alongside these files because recorded protocols,
tests and archived results refer to their exact paths. They are not included
in the current method merely by being present.

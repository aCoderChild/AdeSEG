# Teacher-preserving refinement

Training-only positive teacher objectives are retained *unchanged* as `modeling/positive_teacher_preservation_loss.py` (symmetric weighted deviation) and `modeling/training_foreground_preservation_v1.py` (one-sided confidence-preservation plus training-mask expansion). Teacher predictions are detached; GT is used only in training targets and never to initialize test-time memory.

The clean offline training entry point `scripts/train_aggregation.py` exposes `--teacher-weight` to add the one-sided penalty to the multiscale refiner's supervised loss. This **new combination is a training option**, not a claim that the exact historical teacher-adapter experiment has been reproduced. See `protocols/SOURCE_GROUP_PRESERVATION_V1.md` and the original backup for full pilot runners.

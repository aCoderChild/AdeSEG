# Experiment scripts

The current fixed-read workflow uses the following source-bound entry points:

1. [Capture handoff](verify_cache_release_crossfit_handoff_v1.py) reconstructs
   the audited warm feature captures without opening labels.
2. [Three-fold preflight](preflight_cache_release_crossfit_training_3fold_v1.py)
   and [registry](build_cache_release_crossfit_registry_3fold_v1.py) bind the
   357-read grid and patient-grouped schedule.
3. [Three-fold study](run_bounded_crossfit_shared_study_3fold_v1.py) supervises
   registry verification, an MPS probe and its audit, training, 18 held-out
   prediction sets, and frozen verification and scoring.

Use each command's `--help` and the
[three-fold protocol](../protocols/BOUNDED_CROSSFIT_FIXED_READ_3FOLD_V1.md)
for its required evidence and paths. Running the same study twice is not a
substitute for checking the existing progress report. All results go to the
configured Drive experiment directory.

The directory also retains source-bound utilities from prior comparisons.
Their names do not imply they are active components of the current method.

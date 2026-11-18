# Archived positive-preservation objective

The project also investigated a one-sided positive-teacher loss using the
*training* mask to preserve confident teacher positives. This is distinct
from Bernoulli teacher KL. Both are training-only optimizers, not test-time
segmentation features or ground-truth-dependent prompt selectors.

The original experimental recipes and frozen source versions are preserved in
[the full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).
The current public API is `adeseg.losses.foreground_preservation`.

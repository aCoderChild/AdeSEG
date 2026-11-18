# Archived exploratory controls

The historical experiments include early-feature prompting, region bias,
multi-scale **memory-anchor** weighting, retained-token spatial RoPE, CPU
rotary-key fixes, and individual-sequence preprocessing. They should not be
confused with the trainable feature-aggregation method.

Some archived scripts reference verification files and vendor repositories
that are absent from the focused `main`; the original source must be restored
from the [full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118)
before an exact reproduction.

**Specific contamination risk:** `modeling/seq23_viewport.py` in the archived
source implements a crop selected after looking at the evaluation sequence.
That control is diagnostic, not a predefined general-purpose preprocessing
rule. Any result using it is exploratory, not a sealed held-out measurement.

**Reported diagnostics:** pooled memory-anchor weighting changed 8-sequence
mean Dice by approximately 0.000006 (negligible); the separate automatic
indexed-RoPE experiment reported 0.7318 all-present-frame Dice and 0.7353
matched-target Dice over 21 eligible sequences. These are not learned-adapter
scores.

[Saved indexed-RoPE diagnostic](https://drive.google.com/file/d/13gm6M61y2zKyJSBS5Vxxok5AY1lePhpt/view).

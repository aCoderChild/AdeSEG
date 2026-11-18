# PolypGen sequence splits

These lists split complete PolypGen videos, never individual frames. `seq1` and
`seq7` are excluded because every available ground-truth mask is empty, so they
cannot provide the first visible object required by `RandomUniformSampler`.

The resulting usable split is 13 training, 4 validation, and 4 test videos.
`overfit.txt` contains two short, foreground-containing training videos for the
eight-frame smoke/overfit run.

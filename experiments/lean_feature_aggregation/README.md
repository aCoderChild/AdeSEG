# Lean-SAM2 multi-stage feature aggregation

**Contribution candidate:** learned fusion of early FPN, deep
memory-conditioned features and native mask logits via foreground/background
channel-value attention and residual mask decoding.

- Backbone: Lean-SAM2 / SAM2.1 Hiera-B+; no MedSAM2 checkpoint.
- Trainable module: `adeseg.adapters.multiscale_channel_attention`.
- Control settings: global, foreground, FG/BG and diagonal channel mixing.
- Runtime: output-only or future-memory routing under source-verified leases.
- Objective: source-verified frozen TRAIN captures with optional positive
  teacher preservation.

[Six-stage architecture](../../docs/ARCHITECTURE.md) ·
[Training](../../training/README.md)

The structured class inherits the source-verified original model; it
does not demonstrate a performance improvement by renaming or splitting
a class. The recorded 0.742071 Dice was produced by a *different*
compatibility-corrected Lean author baseline **without** this adapter.
No complete paired 21-video efficacy result is established for the
learned aggregator.

Exact original experimental definitions live in
[the complete backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

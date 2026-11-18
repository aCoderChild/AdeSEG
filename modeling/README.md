# Historical implementation modules

**For new code, import the clear `adeseg/` package instead.**

Why does this directory still contain names like
`multiscale_channel_value_refiner_v1.py` and `lean_indexed_spatial_rope_mps.py`?

Older checkpoints, experiment receipts and reproduction scripts sometimes
verify the original source **path and SHA-256**. Renaming those files in
place would break that evidence. Their public, short names are under:

- `adeseg/backbones/`: MedSAM2 and Lean-SAM2
- `adeseg/adapters/`: high-resolution skip and multiscale channel attention
- `adeseg/losses/`: teacher KL, preservation, segmentation
- `adeseg/memory/`: temporal anchors and spatial position encoding
- `adeseg/inference/`: decoder hook and memory routing
- `adeseg/evaluation/`: segmentation and temporal metrics

The retired `seq23_viewport.py` is a **retrospectively selected evaluation crop**
archived in `backup/experiment-full-20261118`, not in active `main` and
not a trainable model feature. Region-bias and CBAM prototypes are exploratory
controls, not active scientific contributions. Source-bound files are indexed
in `legacy/original_source_bound_modules/README.md`.

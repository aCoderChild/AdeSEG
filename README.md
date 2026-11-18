# AdeSEG

Video segmentation research with **two separate model contributions**:

1. **MedSAM2 (Hiera-T/512):** high-resolution decoder skip adapter with a training-only teacher KL loss.
2. **Lean-SAM2 (Hiera-B+):** multi-scale feature aggregation with channel Q/K/V attention.

The original implementations and source-hash contracts are retained. The
medical adapter is a **reconstruction**, not a verified reproduction of the
missing original forward source.

## Project layout

```text
adeseg/
├── backbones/
│   ├── medsam2.py
│   └── lean_sam2.py
├── adapters/
│   ├── high_resolution_skip.py
│   └── multiscale_channel_attention.py
├── losses/
│   ├── teacher_kl.py
│   ├── foreground_preservation.py
│   └── segmentation.py
├── memory/
│   ├── temporal_anchors.py
│   └── spatial_position_encoding.py
├── inference/
│   ├── medical_decoder_integration.py
│   └── lean_memory_routing.py
└── evaluation/
    ├── segmentation_metrics.py
    └── temporal_metrics.py

experiments/
├── medsam2_high_resolution/
├── lean_feature_aggregation/
├── reference_baselines/
└── exploratory_ablations/

training/      # Source-bound training implementations
scripts/       # Short command-line entry points and legacy runners
modeling/      # Original modules; kept at their checkpoint-bound paths
protocols/     # Frozen experiment settings
legacy/        # Index of original filenames and backups
tests/         # Unit and interface checks
external/      # Pinned third-party implementations
```

See [public API](adeseg/README.md), [experiment registry](experiments/README.md),
and [checkpoint identities](docs/CHECKPOINT_PROVENANCE.md).

## Commands

```bash
git clone --recurse-submodules https://github.com/aCoderChild/AdeSEG.git
cd AdeSEG
python -m pip install -r requirements.txt
python -m pytest -q tests/

python -m scripts.train_med --help       # Reconstructed MedSAM2 adapter
python -m scripts.train_lean --help      # Lean multiscale aggregator
python -m scripts.compare_models --help  # SAM2, Lean, SAM2Long, SAMURAI
python -m scripts.evaluate_cmsa --help   # CMSA-Net image-only inference
python -m scripts.evaluate_lean --help   # GT-initialized Lean baseline
python -m scripts.validate_results --help # Rescore saved Lean/native masks
```

The public trainers now require `--split-file`, explicit `source_group`
and `capture_sha256` for every frozen training capture. They refuse
cross-split group overlap and modified capture bytes, writing a split-receipt
sidecar tied to the checkpoint. These checks still cannot independently verify
that user-provided source groups correspond to actual distinct patients. The new CMSA-Net comparison runs image-only transforms, supplies dummy
labels to the pinned author's loop, and exports prediction sizes using RGB
source images. The historical runner is preserved; the new protocol must be
re-evaluated rather than inheriting old scores.

## Historical reference

The corrected, automatically initialized author Lean-SAM2 run achieved
**0.742071 patient-macro Dice** on 21 eligible PolypGen sequences using
indexed MPS spatial RoPE and blocking pointer transfer, **without** either
learned adapter. [Exact archived result](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view).

MedSAM2-latest is a separate **Hiera-T/512** checkpoint; it must not be
confused with Lean-SAM2's Hiera-B+ weights.

## What do `v1` and `audit` mean?

- **`v1`:** “version 1” of an old experiment or saved protocol. The old name
  is retained only when changing it would break checkpoint/source hashes or
  result provenance. New active interfaces do **not** use version suffixes.
- **`audit`:** independently re-check saved masks, scores, frame coverage and
  prompt records. The new commands call this **`validate`** (for example,
  `scripts/validate_results.py`).

The original experiment source and scores remain available in
[backup/experiment-full-20261118](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

**Scope:** PolypGen is a colonoscopy proxy, not clinical adenoid data.
Results from previously inspected sequences are exploratory. The GitHub
Actions runner is currently failing before tests start; full video
reproduction and independent-test generalization are not verified.

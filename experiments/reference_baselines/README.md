# Reference-model comparisons

Use `python -m scripts.evaluate_reference_backbones --help` for SAM2, Lean-SAM2, SAM2Long and SAMURAI. Use `python -m scripts.evaluate_cmsa_net --help` for the CMSA-Net author baseline. The pinned author repos are submodules, and their different `sam2` packages may require isolated Python processes.

An archived Lean-SAM2 author baseline scored **0.73446** mean Dice over 21 GT-mask-initialized PolypGen videos. This is neither a learned aggregation score nor a final independent test.

[Original audited result](https://drive.google.com/drive/folders/1DmkKRrGmEcJQUp0Pb0fp5u5CS0DSo0B5).

**Separate 21-sequence automatic-prompt Lean result:** 0.742071 Dice on the SAM2.1 Hiera-B+ with indexed RoPE and a blocking MPS pointer correction. See [modified Lean-SAM2 benchmark](modified_lean_742.md). This is not MedSAM2-latest or learned channel-value aggregation.

## Data-flow caution

The retained CMSA-Net comparison constructs a dataset from both frame images and mask paths, with `img_only=False`. The exact author loop must be checked to establish that target labels do not enter prediction. Its prompt policy also differs from the SAM2-family baselines. Use `python -m scripts.compare_models --help` and `python -m scripts.evaluate_cmsa --help` for the retained runners.

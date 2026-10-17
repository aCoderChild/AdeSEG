# Historical utility-aware memory experiment

This document describes the earlier frame-level study, not the active runner.
Its frozen protocols and results preserve provenance. The current workflow is
documented in `docs/REGION_TOKENS.md`. Reused attention, memory, statistics and
tracking helpers remain; obsolete launchers and clinical-only tools were removed.

## What is implemented

- A single additive attention-bias hook for no-op instrumentation,
  leave-one-memory-frame-out labels, oracle exclusion, and learned utility.
- Spatial-memory frame spans are recorded explicitly. Object-pointer tokens are
  never masked or reweighted in the main method.
- Attention weights are reconstructed after Q/K projection and RoPE. The
  maximum difference between manual softmax-QK-V and PyTorch SDPA is saved.
- E2's best-of-N random control uses the same exclusion-set size and N equal to
  the number of leave-one-out evaluations.
- A two-layer utility probe trained with within-frame pairwise ranking plus a
  small normalized-utility regression loss.
- Official DAVIS J and boundary F, failure episodes, sequence-level Wilcoxon,
  whole-sequence bootstrap intervals, and Holm correction helpers.
- Pinned upstream `facebookresearch/sam2` and `Mark12Ding/SAM2Long` submodules.

## Guardrails

The frozen protocol is committed before test E2. Validation and PolypGen are
not used for selection. The Kaggle notebook stops if the official SAM2.1 versus
instrumented DAVIS sanity check differs by more than one J&F point, if an empty
exclusion mask changes logits, or if attention reconstruction exceeds `1e-5`.
It skips E6 when E5 does not beat the best available criterion by the declared
paired sequence-level rule. A stopped/skipped phase is a result, not a missing
number.

`scripts/generate_utility_labels.py` supports all annotated frames for training,
every-N sparse labels for E9, and failure-episode onsets for E2. Outputs include
the protocol and checkpoint SHA-256 hashes.

## Execution

The historical notebooks and delivery scripts are recoverable from Git history,
but are not active entry points. Use `kaggle/region-tokens/` and the verified
region collectors for the current study. Do not rerun an old launcher against
the cleaned tree.

Local verification:

```bash
git submodule update --init --recursive
python -m pytest tests -q
```

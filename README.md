# AdeSEG — polyp video segmentation

AdeSEG investigates trainable refinement for automatic polyp video
segmentation. It uses SAM2 and a Lean-SAM2-based memory reader, with PolypGen
video sequences as development and evaluation data. The research question is
whether early visual detail, deep semantic features and foreground–background
context can improve masks before their errors propagate through memory.

## Project contributions under study

There are **two distinct project refiners**. Both modify predicted mask logits;
neither replaces the Lean memory-pruning algorithm.

1. **Foreground-conditioned boundary refiner.** The earlier module combines
   early and deep features, derives soft foreground and background descriptors
   from the predicted mask, reweights feature channels and predicts a residual
   correction. Its controlled video experiment tests output-only, memory-input
   and combined routing. See [the implementation](modeling/foreground_conditioned_refiner.py)
   and [routing](modeling/lean_boundary_refinement_routing_v1.py).
2. **Multiscale channel-value refiner.** The newer candidate combines early
   and deep features with foreground–background contrast and uses channel
   attention to aggregate values across channels. It is the focus of the
   three-fold study. See [the implementation](modeling/multiscale_channel_value_refiner_v1.py)
   and [routing](modeling/lean_channel_value_routing_v1.py).

These are research hypotheses, not established clinical methods or claims of
novelty. Their checkpoints, preprocessing, prompt schedule and routing must be
identified when reporting results. [Methods and evidence](docs/METHODS.md)
explains their differences and current limitations.

## Results so far

The earlier boundary refiner raised patient-macro present-frame Dice from
0.6593 to 0.6787 on a matched 12-sequence development comparison, improving
7 sequences and worsening 5. Empty-frame false alarms were 137/184 and 138/184,
respectively. These are **not results for the newer channel-value module**;
the sequences were previously inspected and sequence 23 uses a retrospectively
chosen viewport. The channel-value module's four-sequence pilot was mixed. See
the experiment reports in the Drive `AdeSEG/outputs/experiment/contrast_refinement_v1`
directory for per-sequence values, protocols and audit evidence.

The current fixed-read study uses three patient-grouped folds, six model arms
and 18 held-out prediction sets. Each initialized sequence is in one test fold;
the other two groups are used for training under a fixed schedule. It does not
provide an untouched external test, and full-video feedback must be evaluated
separately. Active progress and results live on the configured Google Drive
mount, not in this repository.

## Project map

| Location | Purpose |
| --- | --- |
| [`modeling/`](modeling/) | Project refiner modules and SAM2 routing adapters |
| [`datasets/`](datasets/) and [`data/`](data/) | Dataset definitions, manifests and data access |
| [`scripts/`](scripts/) | Capture, training, prediction, verification and reporting entry points |
| [`protocols/`](protocols/) | Versioned experiment designs and source-bound evidence contracts |
| [`evaluation/`](evaluation/) | Segmentation and sequence-level metrics |
| [`tests/`](tests/) | CPU checks for model behavior, fold isolation and provenance |
| [`external/`](external/) | Author implementations used as backbones or comparison methods |
| [`thesis_report/`](thesis_report/) | Thesis material |
| [`docs/`](docs/) | Method explanation and project navigation |

The source files used by a running or completed experiment have recorded paths
and hashes. Moving or editing them can invalidate its receipts; use a separately
versioned implementation for changes. Author methods may need separate Python
processes because multiple repositories provide a `sam2` package.

## Environment and results

The project uses Python, PyTorch and the existing author implementations. Install
the packages for the intended runner from [requirements.txt](requirements.txt)
and use the checkpoint and data paths specified by that runner's protocol. The
current experiments use Apple MPS with explicit device and resource checks; no
CPU/MPS bit-identity claim is made. Use each runner with its accompanying data,
prompt and source-provenance checks.

All experiment outputs and progress reports belong under the user's mounted
Google Drive `AdeSEG/outputs/experiment` directory. The full local mount is
configured in the experiment scripts. A file appearing locally does not by
itself prove that Drive has finished uploading it.

# AdeSEG

**Two experimental extensions for video segmentation**, with separate
pretrained backbones, trainable adapters, and evaluation procedures.

| Research direction | Pretrained model | What we add | Evidence |
|---|---|---|---|
| **1. Medical high-resolution refinement** | MedSAM2 Hiera-T / 512 | Trainable high-resolution decoder-skip adapter + *training-only* native-teacher KL | Reconstructed adapter; four-sequence pilots mixed, **not** a validated 21-sequence improvement |
| **2. Multi-stage feature aggregation** | Lean-SAM2 / SAM2.1 Hiera-B+ | Region-conditioned channel Q/K/V feature fusion + residual mask prediction | Source-compatible implementation; **no verified 21-sequence learned-adapter gain** |

Teacher KL is a standard distillation objective; its possible contribution
here is **where and how it is applied**, not the invention of KL divergence.
These hypotheses still need controlled ablations and independent evaluation.

[**Model architecture and novelty**](docs/ARCHITECTURE.md) ·
[Experiment evidence](experiments/README.md) ·
[Checkpoint identities](docs/CHECKPOINT_PROVENANCE.md)

## Active project structure

```text
AdeSEG/
├── adeseg/
│   ├── backbones/              # MedSAM2 and Lean-SAM2 (different checkpoints)
│   ├── adapters/               # high_resolution_skip, multiscale_channel_attention
│   ├── losses/                 # teacher_kl, foreground_preservation, segmentation
│   ├── memory/                 # temporal_anchors, spatial_position_encoding
│   ├── inference/              # decoder hook, Lean memory routing
│   └── evaluation/             # segmentation and temporal metrics
├── experiments/
│   ├── medsam2_high_resolution/
│   ├── lean_feature_aggregation/
│   ├── reference_baselines/
│   └── exploratory_ablations/
├── training/                   # lean.py, medical.py, split_guard.py
├── scripts/                    # short train/evaluate/compare/validate commands
├── tests/
├── docs/
├── datasets/
├── external/                   # pinned author implementations
└── legacy/                     # historical provenance index
```

The `modeling/` and `evaluation/` roots remain as **implementation
compatibility modules**, not alternative research contributions. They are
referenced by source-hash-bound checkpoints and the public `adeseg/` APIs.
The old `protocols/` directory and experimental-only runners are absent
from active `main`; exact original files live in the immutable Git backup.

## Quick commands

```bash
git clone --recurse-submodules https://github.com/aCoderChild/AdeSEG.git
cd AdeSEG
python -m pip install -r requirements.txt

python -m scripts.train_med --help
python -m scripts.train_lean --help
python -m scripts.compare_models --help
python -m scripts.evaluate_cmsa --help
python -m scripts.evaluate_lean --help
python -m scripts.validate_results --help
python -m pytest -q tests/
```

Training from frozen feature captures requires `--split-file`, a recorded
`source_group` and a SHA256 for **each** capture. These safeguards validate
file integrity and declared disjointness; they do not independently prove
different patient identities. See [training](training/README.md).

## Reference benchmark (not a learned-adapter contribution)

An archived, compatibility-corrected **author Lean-SAM2** run obtained
**0.742071 patient-macro present-frame Dice** over 21 initialized PolypGen
sequences with automatically initialized masks, indexed spatial RoPE and an
MPS pointer-transfer correction. It did **not** contain the two learned
adapters above. [Original result](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view).

**Clinical limitation:** PolypGen colonoscopy frames are a *proxy*, not a
nasopharyngoscopy/adenoid validation cohort. Previously inspected sequences
must not be presented as untouched independent tests.

**Runtime status:** GitHub-hosted runners have failed before any CI steps
started. Static source checks do not establish successful PyTorch execution,
checkpoint loading or full-video Dice reproduction.


## Reproducible research lifecycle

The active research lifecycle is **Data → Training → Inference → Evaluation →
Ablation → Reproduction**. The lifecycle contract is implemented under
[`adeseg/research/`](adeseg/research/), with [the complete usage guide](docs/RESEARCH_LIFECYCLE.md)
and the [explicit contribution boundaries](docs/CONTRIBUTIONS.md).

```bash
# Inspect the experiment contract (requires populated real file paths and SHA256)
python -m scripts.research verify --protocol /path/to/protocol.json
python -m scripts.research run --protocol /path/to/protocol.json --stage all --dry-run

# Validate a patient-aware JSONL video manifest
python -m adeseg.research.data --manifest /path/to/frames.jsonl

# Execute a fully configured stage and record verified output hashes
python -m scripts.research run --protocol /path/to/protocol.json --stage inference

# Compare matched baseline and adapter per-sequence CSV files
python -m scripts.ablate --baseline-protocol base.json --candidate-protocol adapter.json \
  --baseline-csv base.csv --candidate-csv adapter.csv --output comparison.json

# Run contract tests
python -m pytest -q tests/test_research_protocol.py
```

**Research status:** this is a validated *contract and orchestration layer*, not
a claim that arbitrary model-specific training or full-video benchmarking is
already reproducible. The example protocol is intentionally not executable
until the real dataset/checkpoint hashes and stage commands are supplied.
Current model-specific CLI implementations and checkpoint-compatible source
paths are preserved; see [remaining work](docs/RESEARCH_LIFECYCLE.md#not-yet-implemented).

## MedSAM2 controlled video inference (new, requires real checkpoint)

A matched native-versus-adapter **oracle-GT-initialized** video path is
available through `scripts.infer_med_video`, followed by
`scripts.evaluate_med_video`. Its initial mask comes from the **first
foreground GT frame**; the prompt frame is excluded from evaluation and never
counted as a successful prediction. The evaluator separates Dice on
foreground-present frames, false alarms on GT-empty frames, and equal-sequence
means. See [training/README.md](training/README.md).

```bash
python -m scripts.infer_med_video --help
python -m scripts.evaluate_med_video --help
```

This is a real model-specific execution path, but GPU-backed runs and
numerical parity remain **unverified**. It must not be described as automatic
detection or as reproduction of the historical Lean-SAM2 score.

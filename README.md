# AdeSEG — Medical Video Segmentation Research

A focused research repository with **two separate proposed model contributions**,
reproducible comparison baselines, and explicit clinical/proxy-data limitations.

## Scientific block architecture

The codebase is grouped into **six independent blocks**, rather than putting backbones, adapters, losses, and memory controls into one large model class. See [block registry and flow diagrams](modeling/blocks/README.md).

| Block | Scientific purpose |
|---|---|
| [MedSAM2 backbone](modeling/blocks/medsam2_backbone/) | Frozen medical Hiera-T/512 model |
| [High-resolution adapter](modeling/blocks/high_resolution_adapter/) | Learned early/deep decoder-skip fusion |
| [Teacher-preserving KL](modeling/blocks/teacher_preserving_kl/) | Detached-native-teacher loss **during training only** |
| [Lean-SAM2 backbone](modeling/blocks/lean_sam2_backbone/) | Pinned Hiera-B+ / spatial RoPE corrected model |
| [Multi-scale feature aggregation](modeling/blocks/multiscale_feature_aggregation/) | Six-stage Q/K/V aggregation and residual mask prediction |
| [Temporal memory](modeling/blocks/temporal_memory/) | Optional anchor, memory-position and RoPE ablations |

## Contributions

### 1. MedSAM2 high-resolution refinement with native-teacher preservation — primary

- **Backbone:** MedSAM2-latest **Hiera-T at 512 pixels** (SHA256 verified in the saved report).
- **Learned modification:** early/deep feature projections, spatial fusion and a bounded
  residual injected into the native mask decoder's first high-resolution skip.
- **Training:** supervised mask loss with optional detached native-teacher KL
  and positive-confidence preservation. No ground-truth information is used
  by the adapter at inference.
- **Code:** [medical adapter blocks](modeling/medical_highres/),
  [training](training/train_medical_highres_adapter.py),
  [architecture](docs/MEDICAL_HIGHRES_ARCHITECTURE.md),
  [experiment evidence](experiments/medical_high_resolution_adapter/README.md).

**Provenance warning:** This is a **new reconstruction** from the checkpoint's
five-layer geometry (12,736 parameters), saved metadata and methodology.
The original full forward source, raw-video trainer and runner are absent from
the Git backup. One genuine adapter checkpoint was loaded strictly and CPU
tests passed; that does **not** establish numerical equivalence to the
historical four-patient study.

### 2. Lean-SAM2 multi-stage channel-value feature aggregation

- **Backbone:** original Lean-SAM2/SAM2.1 Hiera-B+ (separate from MedSAM2).
- **Learned modification:** six explicit blocks for early/deep projection,
  compact spatial context, region-aware Q/K/V channel attention, feature fusion,
  foreground/background contrast and residual mask decoding.
- **Code:** [Lean structured aggregator](modeling/structured_feature_aggregation.py),
  [stage blocks](modeling/aggregation_blocks/),
  [training](training/train_feature_aggregator.py),
  [architecture](docs/MODEL_ARCHITECTURE.md).

The refactored Lean model keeps original weight names and checkpoint source-hash
checks. The historical fixed-read feature-fusion study **did not demonstrate
a reliable gain**; see [the research ledger](experiments/README.md).

## Verified comparator — corrected author Lean-SAM2

An independent historical 21-sequence automatic-initialization comparator
achieved **0.742071 patient-macro present-frame Dice**, using Hiera-B+, an
indexed MPS spatial RoPE correction and a blocking pointer-position transfer.
It used 1,683 GT-present frames and had 188/336 labeled-empty false alarms.
This run **did not use** either learned contribution above.
[Original score report](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view) ·
[Comparison protocol](experiments/modified_lean_742/README.md).

CMSA-Net, SAM2Long, SAMURAI, native SAM2 and Lean-SAM2 are retained as
comparison backbones. Historical prompt policies and frame denominators
must be matched before method-level interpretation.

## Repository layout

```text
AdeSEG/
├── modeling/
│   ├── blocks/                   # Six scientific-facing independent block APIs
│   ├── medical_highres/         # Original medical implementation and hook
│   ├── aggregation_blocks/       # Contribution 2: Lean six-stage refinement
│   ├── structured_feature_aggregation.py
│   ├── medsam2_latest.py        # MedSAM2 pretrained checkpoint identity guard
│   └── ...                      # Original compatibility modules
├── training/                     # Separate MedSAM2 and Lean trainers
├── datasets/                     # Video/mask and source-split utilities
├── scripts/                      # Baselines, video inference & audit wrappers
├── evaluation/                   # Segmentation/temporal metrics
├── protocols/                    # Frozen historical + versioned new protocols
├── experiments/                  # Method-specific results and limitations
├── docs/                         # Scientific model and checkpoint diagrams
├── tests/                        # CPU regression checks
├── external/                     # MedSAM2 vendor + five comparison submodules
└── archive/                      # Links to immutable full-history backups
```

## Getting started

```bash
git clone --recurse-submodules https://github.com/aCoderChild/AdeSEG.git
cd AdeSEG
python -m pip install -r requirements.txt
python -m pytest -q tests/

# Train a reconstructed medical adapter from TRAIN-only frozen decoder captures:
python -m training.train_medical_highres_adapter --help

# Train Lean-SAM2 multi-stage aggregation on its own frozen feature captures:
python -m training.train_feature_aggregator --help

# Inspect matched baselines and comparators:
python -m scripts.evaluate_reference_backbones --help
python -m scripts.evaluate_cmsa_net --help
```

Full model evaluation requires authentic source-identified pretrained weights,
sequence manifests, correctly constructed prompts and annotations that are
never opened during inference. The medical adapter currently has no complete
raw-video reproduction runner for the lost historical forward implementation.

## Historical medical results (4 exploratory patients)

| Method | Equal-patient present Dice | Advancement |
|---|---:|---|
| Native MedSAM2 Hiera-T/512 | 0.5111 | Baseline |
| Learned high-res skip | 0.5223 | Failed seq8/seq11 preservation criteria |
| Teacher-preserving high-res | 0.5144 | Failed multi-patient advancement criteria |

These are **archived historical scores only**, not newly reproduced results
from this reconstructed package. Neither candidate advanced to a verified
21-patient efficacy run. Detailed evidence:
[medical experiment](experiments/medical_high_resolution_adapter/README.md).

**Checkpoint identities:**
[MedSAM2 Hiera-T/512 vs Lean Hiera-B+](docs/CHECKPOINT_PROVENANCE.md).

**Validation status:** The medical adapter's checkpoint schema and isolated CPU smoke tests were checked during development; the new block-level integration tests and repository-wide test suite still require execution on a working runner. GitHub Actions has been failing *before job steps
start*, so repository-wide runner verification and full video Dice reproduction
are still outstanding.

**Scientific limitation:** PolypGen is a colonoscopy proxy, not a human
nasopharyngoscopy dataset. The cohorts were previously inspected, and
clinical/independent-test efficacy cannot be claimed from these experiments.

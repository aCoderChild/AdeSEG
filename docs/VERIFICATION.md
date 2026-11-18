# AdeSEG verification gates

This file distinguishes **test code in GitHub** from **tests that actually ran**.
Never mark a gate green until a successful run or receipt exists.

## Gate 1: full CPU suite on GitHub Actions

- Workflow: [Verify AdeSEG research core](../.github/workflows/verify.yml)
- Runs `python -m compileall` and `python -m pytest -q tests` under Python 3.11.
- Exports JUnit XML and checks five pinned reference gitlinks.
- Contract workflow: [Research lifecycle contracts](../.github/workflows/research-contracts.yml).
- Independent diagnostic: [Runner availability diagnostic](../.github/workflows/runner-diagnostic.yml).

**Current blocker (2026-11-19):** recent GitHub Actions jobs finish as failed with
zero executed steps (no runner assigned). Retrying one CPU job did not resolve
this. There is no evidence that the Python test commands were executed.
The repository alone cannot resolve account-level Actions settings,
hosted-runner eligibility, quota or billing restrictions. Repository owners
should inspect Settings → Actions → General, the Actions usage/billing
information, and the failed run's annotation; run the minimal diagnostic
workflow manually. Do not call this a code regression without test logs.

## Gate 2: real MedSAM2 checkpoint smoke

Workflow: [MedSAM2 GPU smoke and parity](../.github/workflows/gpu-smoke.yml).

**Manual only.** This workflow requires:
1. A trusted Linux x64 **self-hosted CUDA GPU runner** labeled `adeseg-gpu`
   with Python, PyTorch/CUDA, NumPy, SciPy, Pillow, OpenCV, pytest, Hydra,
   iopath and the vendored MedSAM2 dependencies installed.
2. The authorized **MedSAM2_latest.pt** checkpoint at an absolute path on
   that runner. Its SHA256 must equal the pinned MedSAM2 digest in
   `modeling/medsam2_latest.py`; a filename alone is not sufficient.
3. A **curated short development manifest**. Exactly one DEV video with
   2–24 ordered JPEG frames and corresponding indexed 0/1 mask images.
   Its first GT-positive frame must have later frames, and at least one later
   frame must also be positive.
4. GitHub repository **Actions variables** (Settings → Secrets and variables →
   Actions → Variables), set to absolute paths visible on the GPU runner:

   | Variable | Meaning |
   |---|---|
   | `ADESEG_MED_SMOKE_MANIFEST` | Path to a short, approved DEV JSONL manifest |
   | `ADESEG_MED_CHECKPOINT` | Path to pinned MedSAM2_latest.pt |
   | `ADESEG_MED_MODEL_CONFIG` | Optional, defaults to sam2.1_hiera_t512.yaml |
   | `ADESEG_MED_ADAPTER` | Optional path to **trained** reconstructed adapter checkpoint |
   | `ADESEG_MED_ADAPTER_SHA256` | Required if adapter is set; full lowercase SHA256 |

Only enable the job using Actions → "MedSAM2 GPU smoke and parity" →
Run workflow → select `RUN_GPU`. `DISABLED` is the default.
No external checkpoint or dataset is automatically downloaded. Do **not**
attach private patient data to a public Git repository or unrestricted
runner. The workflow uploads metrics-only JSON, not video, masks or prompts.
Use a trusted runner and review your organization's security policy.

The preflight rejects missing CUDA, incorrect checkpoint digest, wrong
configuration, non-DEV manifest, multiple videos, too many frames or missing
valid prompt/target frames.

## Gate 3: numerical identity control

The GPU workflow executes the same short video twice:

- `native`: pinned native MedSAM2 video predictor.
- `zero_initialized_control`: same model with the new adapter initialized
  to an **exact zero residual** and attached to the decoder.

`scripts.compare_med_runs --require-logit-identity` then verifies both
outputs against original image/mask SHA256 and checks that the binary masks
**and the float32 decoded logit tensors** have identical SHA256 on all
post-prompt frames. Mismatches fail the job. This is an implementation-parity
check, not proof of improvement. Bitwise equality across separate GPU runs
can also be affected by non-determinism; investigate the CUDA runtime and
repeat native runs before attributing a mismatch to the adapter.

## Gate 4: trained-adapter short-video study

If both optional adapter variables are configured, the workflow validates the
adapter against its pinned backbone, source and checkpoint hash, runs its
full-video inference and compares **exactly matched post-prompt frames**.
It exports per-sequence present-frame Dice differences and empty-frame false
alarms. This is a smoke/consistency study, **not** evidence of generalization.

## Gate 5: full benchmark reproduction

**Not automatically enabled.** Only proceed after the previous gates pass:

- Freeze and independently audit patient/source splits and the exact
  benchmark manifest.
- Run the native and *trained* adapter on the same frozen split, prompt
  positions, model config, checkpoint, postprocessing, hardware settings,
  frames and annotation hashes.
- Score separately: GT-present frame Dice, equal-sequence Dice,
  empty-frame false alarms and temporal consistency.
- Keep the provenance `run.json`, evaluation outputs, environment lockfile,
  checkpoint digests, git revision and all failure records.
- Report confidence intervals over independent source/patient groups, not
  treating individual video frames as independent samples.

`scripts.infer_med_video`, `scripts.evaluate_med_video`, and
`scripts.compare_med_runs` support this controlled paired workflow when
real assets are available. The former uses **first GT-mask initialization**,
not automatic detection. It is not a reproduction of the automatic
Lean-SAM2 baseline, and Lean-SAM2 adapter-to-video integration remains a
distinct unfinished task.

## CPU-only local checks

```bash
python -m compileall -q adeseg datasets evaluation modeling scripts training tests
python -m pytest -q tests
python -m scripts.infer_med_video --help
python -m scripts.evaluate_med_video --help
python -m scripts.compare_med_runs --help
```

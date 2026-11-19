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
2. The authorized SHA256-pinned MedSAM2 pretrained checkpoint and a curated
   **one-video DEV-only** JPEG/mask JSONL dataset on the trusted runner.
3. Generate a sealed contract using
   `python -m scripts.prepare_med_video_protocol --manifest /path/to/dev.jsonl
   --checkpoint /path/to/MedSAM2_latest.pt --output /path/to/med_contract
   --split dev --label-id 1`. It creates an authenticated
   `protocol.json` and frozen `prompt_frames.jsonl`.
4. Set GitHub Actions variables (Settings → Secrets and variables → Actions):
   
   | Variable | Meaning |
   |---|---|
   | `ADESEG_MED_PROTOCOL` | Absolute path to the sealed `protocol.json` on the trusted GPU runner |
   | `ADESEG_MED_ADAPTER` | Optional path to trained reconstructed adapter |
   | `ADESEG_MED_ADAPTER_SHA256` | Required if an adapter is set; its exact lowercase SHA256 |

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

## Strict shape and implementation-source contracts (2026-11-19)

The vendored MedSAM2 checkpoint loader now **rejects any mismatched tensor
shape** before modifying the target model. It uses explicit shape
`assert` statements for auditable invariants, backed by unconditional
`RuntimeError` checks (so `python -O` cannot disable the protection).
The sole exception is the unused, zero-sized `maskmem_tpos_enc` tensor
when `num_maskmem == 0`. State-dict keys must match exactly.

New reconstructed MedSAM2 training checkpoints also include
`implementation_sha256` over adapter, blocks and decoder integration source.
New structured Lean-SAM2 training checkpoints include an analogous digest
covering the aggregation blocks and memory-routing implementation. Loading
older source-only adapter checkpoints requires an **explicit**
`allow_legacy_source_only=True` opt-in. This compatibility path is not
evidence of historical forward parity.

The GPU-smoke preflight now verifies **GT-positive frames after the actual
first-positive prompt**, one DEV-only short video, safe source IDs, contiguous
source frame indexes and JPEG frame encoding. The video runner uses the
same strict continuity and ID checks.

For 0/255 binary or already-indexed anatomical masks, use the new
non-destructive converter:

```bash
python -m scripts.build_mask_manifest --manifest /path/to/input.jsonl \
  --output /path/to/converted --kind binary
# Or preserve anatomical labels 0=background, 1=adenoid, 2=airway:
python -m scripts.build_mask_manifest --manifest /path/to/input.jsonl \
  --output /path/to/converted --kind semantic --labels 0,1,2
```

RGB color-coded semantic masks are rejected until a reviewed explicit color
palette is supplied. This tool creates indexed PNGs and a new JSONL manifest;
it never overwrites the original annotations.

## Source-of-truth rule

The MedSAM2 oracle-video commands `scripts.infer_med_video`,
`scripts.evaluate_med_video`, `scripts.compare_med_runs`, and
`scripts.check_med_gpu_smoke` all require **one** `--protocol` file.
The contract authenticates original image/mask prompt selections, file SHA256,
512×512 backbone configuration, the 0.5 sigmoid mask threshold, no model
postprocessing, post-prompt present-frame Dice, and equal-sequence averaging.
These are fixed for the controlled research path, not inferred from CLI flags.
The legacy generic research lifecycle schema remains available for other tracks.

## Capture-training restart integrity

Both trained adapters can save authenticated **completed-epoch** snapshots
(`--save-epoch-snapshots`) and resume with `--resume-checkpoint` plus
`--resume-sha256`. Epoch count means the *total* desired count. Source-file
fingerprints include the active trainer and its loss implementation, as well
as the adapter implementation. A synthetic CPU AdamW regression test checks
that uninterrupted and epoch-restarted parameter values match exactly.
Arbitrary mid-epoch recovery and cross-device deterministic parity are NOT
established.

## Sealed evaluation and numerical evidence

`python -m scripts.seal_med_video_study --protocol <protocol.json> --native
<native_run> --zero <zero_run> --output <new_receipt.json>` re-reads every
prediction and independently saved float32 logit artifact, demands exact
native/zero-adapter parity, checks a clean Git revision and hashes critical
source files. Use `--trained <trained_run>` to include a trained candidate;
its presence does not imply improvement. The GPU workflow uploads the
**metrics-only** receipt and summary, never raw private images or logit tensors.
All three descriptive temporal measures (adjacent mask IoU, relative area
change and normalized centroid displacement) are reported separately from
foreground-present Dice and empty-frame false alarms.

`adeseg.research.lifecycle` accepts a new six-stage schema version 2; schema
version 1 remains compatible with its original four stages. Any missing stage
is explicitly incomplete and therefore cannot be reported as reproduced.

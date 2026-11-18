# CMSA-Net PolypGen comparison

This is a standalone comparison using the official CMSA-Net Res2Net-50 model,
not an integration of CMSA into SAM2. The source is `external/CMSA-Net` at
commit `4e4602e93211b5d0006a8a7415e8477de413b79d`; the released Res2Net
`ckpt_epoch_007.pth` was downloaded from the authors' OneDrive link and loaded
with strict state-dict matching. Its SHA-256 is
`d391d9b6a6eb08eb4bc80651861a38ef52cc30e35d942d529312c226cff96f32`.
The checked-in implementation constructs Res2Net; the upstream test script's
default PVT checkpoint path does not match that implementation.

`scripts/run_cmsa_polypgen.py` adapts the PolypGen chronological manifest to
the upstream `CMSAVideoDataset` sample and calls the upstream test method,
including its semantic/confidence reference updates. It uses the upstream
352×352 PIL resize, transforms, normalization statistics, four-frame windows,
first-save deduplication, probability quantization, and mask resize. The
upstream normalization indexes the last tensor dimension (`img[:, :, i]`),
so it affects the first three image columns rather than normalizing three
channels. The first cohort used that released behavior. The runner defaults to
the released behavior; `--normalization channel` selects corrected channel-wise
normalization. The pinned statistics file has SHA-256
`fede962cb17631cb59cabd0e96a95495b13f49f970fbc3ecdfb47b7d9b113091`.
The released training loader also uses the column-wise operation,
so inference-only correction may not match the checkpoint's training inputs.
Neither mode is established as the checkpoint's original training behavior.

Upstream's first window includes frames 0–3 and saves all four at once. The
initial references include frame 1, so frame 0 can depend on a later image.
All matched post-prompt targets exclude frame 0. The model's attention is
causal across clip positions, and subsequent reference updates take effect
only on later windows. The earlier blanket warning that multiple evaluated
frames necessarily use future images was too broad. No PolypGen ground-truth mask is passed as model input: labels are used
by the upstream data transform and to resize/score outputs. SAM2's native
reference receives a ground-truth mask prompt, so the two methods have
different inputs and this is a descriptive comparison, not an isolated
architectural effect.

The default cohort is the 21 sequences used by the native reference (seq2–23,
excluding seq7). Dice/IoU use the repository's PolypGen label decoder and
threshold the author's saved 8-bit probability map at 128 after resizing.
Scores are aggregated per sequence on ground-truth-present frames after
SAM2's first labeled prompt, then averaged equally across sequences. False
alarms are counted separately on ground-truth-empty target frames. Seq6/8 were
used previously for research development; patient independence is unverified.

Before the full run, the official checkpoint loaded with no missing/unexpected
keys. On seq3, all 12 input clips matched the original author's dataset exactly,
and two independent CPU runs produced identical 15 prediction hashes and
scores. The independent audit rescored those predictions and verified all 14
post-prompt target indices and presence labels against hash-checked archived
native records. A focused clip-coverage test is `tests/test_cmsa_polypgen.py`.
These checks do not establish equivalence to the authors' original CUDA
runtime, training preprocessing, or reported SUN-SEG scores.

Outputs and checkpoint are in `AdeSEG/outputs/experiment/cmsa_net_polypgen_v1`
on the configured Drive mount. `setup.json`, `frames.json`, `results.json`, and
`matched_native_audit.json` record the setup, frame scores, summaries, and
independently rescored comparison when each stage completes. A missing results
or audit file means that stage has not completed.

To reproduce with an existing author checkpoint, choose a new Drive output
directory for each attempt:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/run_cmsa_polypgen.py \
  --checkpoint '/path/on/Drive/ckpt_epoch_007.pth' \
  --output '/path/on/Drive/new_attempt'
python scripts/audit_cmsa_polypgen.py --run '/path/on/Drive/new_attempt' \
  --native '/path/on/Drive/143_official_native_feature_audit_v1/full/native_results.json'
```

A shorter gate can be run with `--sequence seq3`. Add `--normalization channel`
for the corrected-normalization sensitivity run. The full runner refuses to
overwrite an existing output directory. Inference uses CPU by default; a
non-CPU device is not covered by the reported repeatability check.

## Implementation audit and corrected-normalization confirmation

The released training and evaluation data paths both apply `toTensor_video`
(CHW output) before the same `Normalize_video`, confirming the indexing error
in the published training path as well as the test path. The published source
alone does not prove the exact preprocessing that produced the downloaded
checkpoint. The corrected mode applies the stored SUN-SEG mean and standard
deviation across the whole channel, and changes no model parameters, data,
reference-update policy, probability conversion, output resize, or scoring.

Other checks: the adapter loads the exact upstream model with strict checkpoint
matching, uses upstream sample ordering and dynamic-reference implementation,
and decodes ground truth with the same project functions as the archived SAM2
reference. The original upstream test uses `strict=False`; the adapter's strict
load avoids silently accepting missing parameters. Upstream SUN-SEG evaluation
min-max normalizes each prediction, excludes the first and last frame, and
aggregates Dice over thresholds. Our fixed-threshold, post-prompt PolypGen Dice
is intentionally a different metric; it is not directly comparable with the
paper's published Dice. Dynamic reference selection and decoder computations
were not rewritten: no independent evidence establishes that altering them
would correct this pretrained model.

Two corrected-normalization seq3 runs were byte-identical for all 15 predicted
frames; a perturbation test checks the attention's causal clip-position mask.
The first window's frame-zero output has access to the frame-one reference,
but frame zero is excluded from all matched post-prompt target sets. The full
cohort is an exploratory sensitivity comparison against the original run, not
a claim of improvement or independent external confirmation. The CPU runtime
and library versions differ from the authors' reported training environment.

The paired full-cohort audit found that the channel-wise inference-only change
reduced Dice and increased false alarms for this checkpoint on the matched
PolypGen frames. Therefore the runner defaults to released preprocessing for
checkpoint compatibility, while retaining the mathematically corrected mode
for explicit investigation. This result does not establish the checkpoint's
training provenance. The saved comparison and the source snapshot used by the
corrected cohort are under `channel_normalization_full_attempt01` on Drive.

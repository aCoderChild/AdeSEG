# Contrast-refinement experiments: status and sequence

> **Historical snapshot:** The table below describes the state on 2026-11-14
> and is not the current experiment status. Later fixed-read controls,
> full-video factorials, additional seeds, and channel-value experiments have
> been completed. For their audited results and current limitations, see the
> Drive-backed `outputs/experiment/contrast_refinement_v1/progress.md` tracker.
> The patient-grouped cross-fit and extended comparator studies are ongoing;
> no all-23-sequence matched ranking has been established.

Updated 2026-11-14. The detailed design and evaluation requirements are in
[MICCAI_2027_EXPERIMENT_PROTOCOL.md](MICCAI_2027_EXPERIMENT_PROTOCOL.md).
This is a research plan, not a claim that the method improves segmentation.

| Stage | Experiment | Status |
| --- | --- | --- |
| 1 | Unit checks of shapes, numerical behavior, checkpoint provenance and independent output/memory routes | Passed focused tests; does not establish end-to-end correctness. |
| 2 | Actual Lean/SAM2.1 predictor with identity and forced-logit stimuli, comparing output and stored state | Passed on a synthetic two-frame clip for A/B/C/D; long-window pruning and real-video trajectories still need checks. |
| 3 | Check alignment and provenance of previously saved frozen-backbone reads | File hashes and geometry passed for 25 fixed frames. The seq3 gate and runtime identity match the project's recent-token-window variant, not original author Lean. Full live-runtime binding and patient independence remain unverified. |
| 4 | Train and freeze a reproducible segmentation-refiner feasibility pilot; compare held-out fixed reads against unmodified masks | Completed for one five-epoch seed and two models on previously inspected samples; see results below. This does not establish generalization. |
| 5 | Fixed-frame controls: mask-only, early-only, deep-only, early+deep without contrast, foreground-only, foreground/background, RGB and shuffled features | The mask-only control completed for the exploratory pilot; remaining controls are not complete. Match training, capacity and inputs as closely as possible and report differences. |
| 6 | Real-video no-op and routing verification, including nonzero prompt, seq23 crop and active Lean pruning | Not complete; required before propagation experiments. |
| 7 | Matched automatic-prompt video factorial A/B/C/D on development sequences, including seq11/20/23 | Not complete. Conditional on engineering checks and predeclared fixed-frame criteria. |
| 8 | Baselines and comparator audit: native SAM2, original Lean, recent-window Lean, SAMURAI, SAM2Long and CMSA-Net | Prior runs require prompt, crop, checkpoint, sequence and temporal-access audit before a fair comparison. GT-prompt and automatic-prompt results are separate tracks; do not rerun historical GT-prompt native/Lean without reason. |
| 9 | Frozen-model independent patient/center-disjoint evaluation and repeated training seeds | Not complete. Patient identity and external data availability need verification. The previously inspected difficult sequences are exploratory. |

The factorial holds the original automatic YOLO prompt, Lean pruning, presence
logic and object pointers fixed. The initial conditioning output stays native.
A routes original logits to output and memory; B refines only output; C refines
only the memory write; D refines both. Corrected logits pass through the native
memory-encoder transform. Do not interpret a two-frame identity check as
evidence of nonzero-correction accuracy or a benefit from feedback.

The real-predictor smoke also checks that a deliberately forced nonzero test
logit takes the intended output and memory paths. It is an engineering
stimulus, not a trained refiner. During review, a pilot-runner resize mismatch
was identified and fixed before training: video output uses the predictor's
direct low-resolution to video-resolution interpolation, rather than an extra
intermediate resize. A regression test covers the corrected path.

## Exploratory fixed-read pilot, 2026-11-14

Training used 15 saved reads from seq2/3/4. Five epochs and one seed were used
for both models. Validation used 10 saved reads from seq9/10: seven labeled
present and three labeled empty. These reads use an experimental 16,384-token
window with indexed RoPE and a multiscale anchor, initialized from archived
automatic box-plus-feature-point prompts. They are not original author Lean or
the historical ground-truth-prompt Lean comparison. The seq3 gate and runtime
receipts match the project's recent-token-window variant; a live video runtime
still needs verification before reusing the checkpoint. Patient
independence and representativeness are unverified; no no-object-logit cases
are represented. Values below pool validation frames rather than videos.
All 15 training labels contain foreground, so empty-frame behavior received no
direct training supervision. Seq9/10 have now been inspected and must be
treated as development data for subsequent iterations.

| Method | Present-frame mean Dice, 7 frames | Empty-frame false alarms | Mean predicted pixels on 3 empty frames |
| --- | ---: | ---: | ---: |
| Original captured logits | 0.921285 | 3/3 | 1,162,747 |
| Multilevel contrast refiner | 0.920192 | 3/3 | 1,173,450 |
| Mask-only control | 0.921293 | 3/3 | 1,163,685 |

The candidate did not beat the baseline or mask-only control and increased mean
empty-frame predicted area in this pilot. The sample is far too small to support
an efficacy claim or infer why it failed. In particular, none of these reads is
from the priority sequences, and the checkpoint's feature pipeline has not
been verified against a live intended baseline. Do not advance it to full-video
efficacy testing under the declared selection rule. Revise using development
data and stronger controls before considering propagation.

Report per-video present-frame Dice, empty-frame false alarms and false-positive
area, boundary metrics, prompt coverage and computational cost. Store manifests,
checkpoints, logs and results under the user's Google Drive experiment directory.

## Additional audit and MPS development run

Independent replay verified the original pilot's saved predictions and metrics.
The author Lean CPU/MPS frequency-repeat order differs from its CUDA path; a
scoped correction and tests are available. Preserving compressed-token spatial
coordinates is a separate issue. The historical SAMURAI and SAM2Long runs used
ground-truth prompts; SAM2Long selects its final path after video processing.
Details are in [COMPARATOR_AUDIT_2026_11.md](COMPARATOR_AUDIT_2026_11.md).

The 17/10 exploratory development pilot added two label-selected false-positive
empty seq4 frames to training; the validation set was reused and inspected.
On MPS, the baseline, multilevel and mask-only mean Dice on the seven present
validation frames were 0.921285, 0.920678 and 0.921256 respectively. All
three produced false alarms on all three empty validation frames. Mean predicted
areas on the empty frames were 1,162,747, 1,171,126 and 1,163,870 pixels.
These results do not support a candidate improvement or independent claim.

A subsequent 30-epoch development follow-up, chosen after inspecting the
five-epoch results, yielded present-frame Dice 0.921285 for the captured
baseline, 0.859409 for the multilevel model and 0.915619 for mask-only. All
had 3/3 empty-frame false alarms. It uses the same small, reused validation set
and is not independent confirmation; full-video candidate efficacy remains untested.

Fixed-read CPU/MPS masks matched in the parity checks. In contrast, fresh-process
MPS video runs varied even with fixed seeds, deterministic algorithms and MPS
fast math disabled. A temporary CPU-to-MPS asynchronous object-pointer position
transfer was identified as a cause of observed positional mismatch; a scoped
blocking workaround gave bit-identical results in two short repeat runs.
A subsequent short replay used the actual token-window fork with the archived
prompt and matched CPU captures: both MPS runs agreed, and three of twelve
CPU/MPS masks differed by one pixel each. Both devices respected the 16,384
token budget and produced five evictions. The check was not a complete video,
and other transfer sites, longer trajectories and learned-refiner binding
remain to be verified.

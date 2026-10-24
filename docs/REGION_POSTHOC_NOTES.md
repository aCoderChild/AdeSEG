# Post-hoc regional follow-up — 2026-10-24

All analyses and the proposed propagation rule below are **post hoc and
exploratory** on the already inspected PolypGen cohort. The completed frozen
exclusion decision is unchanged: no region passes the all-definition claim rule.

## Control interpretation

The background control rigidly moves/rotates/reflects a large background-shaped
exclusion mask. For a full-grid bounding box this acts as a mirror/rotation swap
of the small retained object-shaped complement; it can also remove polyp-token
positions. It is **not** a pure background-versus-background comparison and is
not necessarily a mirror reflection in every case. Count, shape and connectivity
matching do not match anatomy. Large background excess loss must not be labeled
the causal contribution of semantically pure background features.

## Output location and protection

The user renamed `outputs/16:10:2026` to **`outputs/experiment`**; the new mounted
path was verified on 2026-10-24. Historical recorded absolute paths are retained
as provenance, not rewritten. New scripts take explicit paths.

The requested independent backup and drive.google.com confirmation are **not
yet verified**. No external disk is mounted; the user requests Drive-only
storage. A second folder in the same Drive account is only a snapshot copy,
not independent account/device protection. Browser control failed and the
connected Drive search did not expose this project. W&B delivery receipts are
separate evidence, not proof of the requested Drive-web upload.

No new model inference is authorized by a falsely completed protection check.
The propagation runner requires a verified copy manifest and a Drive-web receipt.
The first snapshot attempt stalled enumerating regenerable frame staging. The
second snapshot excludes that staging and upload caches; it remains incomplete
while mounted cloud-only files hydrate. Neither attempt has produced a verified
protection receipt. No scientific output was deleted or overwritten.

## Offline analyses

`scripts/analyze_posthoc_regions.py` reproduces sequence-level boundary TOST
equivalence (post-hoc margin ±0.02), exact absent-target transition denominators,
query-region/layer attention and frozen polyp-size strata from saved outputs.
The supplied 85%/42% claims must distinguish residual, induced and cleared false
alarms; they are not automatically accepted as verified findings.

The saved assignment tensors contain native mask logits and region schemes;
native memory vectors were hashed but not saved in the current attention audit.
Per-token attention weights, mask logits and hashes cannot recover vector norms.
Consequently the proposed no-inference norm comparison is unavailable from those
files; it must not be fabricated or replaced by a mask-logit norm. Even genuine
raw-vector norm differences would be an association, not proof of an attention
sink; projected keys/queries, position and softmax also matter.

## Causal full propagation

The user approved a **previous-frame native object-score** gate: score≤0 excludes
only foreground-interior memory keys; positive score restores the next native
read. Background, boundary and pointers remain. Current-frame scores cannot
control their own earlier memory read without a different two-pass algorithm.

Reuse: pinned official SAM2 and SAMURAI repositories (already cloned; pinned
commits fetched again), existing project key-alignment hooks, Efficient-SAM2
occupancy and native PyTorch SDPA. No replacement backbone, decoder, memory
writer or attention implementation is introduced. The conditional masking
policy is explicitly project experimental glue, not attributed to an author.

Verification at this implementation freeze: **171 tests passed**. These are
unit/regression checks, not completed real-video propagation gates. Native and
gated correctness/efficacy are not claimed before those separate runs complete.

The no-inference [post-hoc report](REGION_POSTHOC_RESULTS.md) is now reproduced,
including exact conditional false-alarm denominators. A second same-account
Drive snapshot is being attempted; it is not yet a verified independent backup
or a verified web upload. [Execution guide](FULL_PROPAGATION_FALSE_ALARM.md).

Dataset audit: 323 GT-empty postprompt targets,57 empty stretches,51 observed
returns and6 terminal empty tails. Seq4/seq9 exercise absence and return;
seq2/seq3 do not. Latency is in sampled frames because source image spacing is
irregular. GT-empty means annotation-defined absence. Separate image-only
negative sequences are not silently appended to the current cohort.

[Frozen propagation design](../protocols/full_propagation_false_alarm_v1.json).
Native code references:
[SAM2 object-score and memory path](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py),
[SAMURAI motion-aware memory](https://github.com/yangchris11/samurai/tree/76ba195984892b0d1e3db5d9c9f90bb62175680a).

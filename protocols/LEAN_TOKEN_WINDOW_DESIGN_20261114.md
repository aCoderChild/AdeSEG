# Lean spatial-token window: design and validation requirements

Status: **experimental integration under evaluation** in the isolated
`external/Lean-SAM2-token-window` worktree. The pure selection policy in
`modeling/token_memory_window.py` is connected there through
`modeling/lean_token_window_storage.py`; the original Lean checkout is
unchanged. Two no-op replays and two candidate engineering replays passed,
including an independent gate audit. Four exploratory video comparisons are
running; no performance gain has been established. The enforced represented
spatial-feature budget is not a bound on total RAM.

## Budget definition

Count resident spatial feature-token entries from conditioning memory,
nonconditioning memories and the insurance bank against one fixed capacity.
For each selected feature entry retain its paired positional entry and original
grid coordinates. Position tensors may be shared; report their actual bytes
separately. Object pointers, anchors, cached masks, other predictor state and
temporary compute tensors are **outside** this spatial-feature budget and must
be separately measured. Do not describe a spatial budget as a bound on total
RAM. The complete prompt memory stays protected at 4,096 tokens unless the user
explicitly chooses otherwise. Protect the latest full memory and current
insurance entries; fail rather than silently exceed the cap when protected
entries do not fit.

A provisional candidate budget is 16,384 spatial feature entries: this equals
13,312 represented feature entries plus 3,072 insurance entries in the common
pre-read state seen in the sequence-3 engineering replay and the three
completed sequence-8/20/23 observational replays. These are representation
counts, not deduplicated bytes, and do not establish an upper bound at writes.
Fix and record the exact experimental budget before validation. Do not tune it
against the target labels.

Use chronological contiguous recency to fill the remaining capacity by
actually resident feature-token counts, preserving the selected feature and
position indices together. Do not replace a missing entry with an older one to
simulate a fixed number of frames. Any pointer-retention policy must preserve
the author's inputs for supported forward-only inference and be accounted for
separately. The author currently reads up to its configured recent object
pointers, while old pointer records may remain stored.

## Integration hazards

Lean marks a dense 4,096-token entry `pending` before physically reducing it to
1,024. Its profiling helper can report the latter logical count even though the
full tensor remains resident; the window must count the actual representation.
It also fuses older history and may prune its resident visual entries. A
frame-index-only cache wrapper would leave the original fixed `num_maskmem`
read window in place and therefore would not implement the request. Integration
must explicitly change both admission/eviction and memory-read construction,
and define how the author's fused-history behavior is handled. The trained
temporal-position table has only the original number of slots: do not silently
index beyond it or pretend that duplicated temporal embeddings represent
learned long-range positions. Preserve object-pointer order and indexed rotary
position pairing. Any change to author behavior must be disclosed and compared
with the exact indexed reference, rather than called an unchanged Lean model.

Enforce the budget at every persistent memory write and after pending entries
are materialized, not just at the start of a read. Explain transient allocation
peaks separately. Evict only spatial memory and safely unneeded pointer/state
entries; do not destroy prediction outputs or externally owned artifacts. The
existing CMSA-inspired pooled-anchor scoring and early-feature prompting stay
fixed. No ground truth, re-detection, additional inference confidence gate or
new inference threshold is allowed.

## Checks before claiming success

1. Unit-test variable-sized entries, reserved insurance, simultaneous pending
   dense representations, index/position pairing, impossible protected
   allocations and chronological eviction. Observe actual buffers rather than
   relying on Lean's logical profiling count.
2. Verify a no-op path against the exact indexed CPU reference on two independent
   label-free replays, including all masks, token indices, rotary positions,
   object-pointer order, model weights and bank contents. Verify candidate
   repeatability and the cap after every persistent write, and record actual
   bytes, pointers and transient peaks separately. Fail closed on violations.
3. Compare the fixed candidate on sequences 8, 11, 20 and 23 with identical
   input preprocessing and automatic prompt schedule. In particular, the
   sequence-23 no-inset preprocessing must be present on both candidate and
   corresponding baseline. Native and historical Lean GT-initialized results
   are descriptive references, not matched-prompt comparisons.
4. Independently rescore present-frame Dice and empty-frame false alarms,
   report per-sequence results and actual stored token/byte traces, and assess
   both accuracy and compute. The overall objective is not met unless the
   required per-sequence targets are achieved and verified with their prompt
   differences made explicit.

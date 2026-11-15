# Short MPS allocator-memory diagnostic with setup milestones

This separately versioned diagnostic addresses a guarded seq6 run that stopped
when system-available RAM dropped below 4 GiB. It makes no change to the frozen
capture, cross-fit training or inference policy. The stop did not persist
per-frame MPS allocation readings, so its precise cause remains unknown.

Two fresh processes run the same seven-frame seq11 prefix, checkpoint, bounded
loader and archived automatic box→points→mask warm initialization. After each
of the five prompt/post-prompt outputs, both synchronize and record system
available RAM, current process RSS, MPS live tensor bytes and Metal-driver
allocated bytes. In `clear_idle_cache` only, the process calls
`torch.mps.empty_cache()` between the before/after samples. This requests
release of unoccupied allocator cache; it does not promise to release live
model tensors or MPS/MPSGraph allocations. `observe` uses identical measurement
and synchronization points without cache clearing. Per-frame records are
flushed during the run so a guarded stop preserves collected samples.

Both arms additionally synchronize and immediately record the same memory fields before and after model load, video-state initialization, and warm prompting. These checkpoints are instrumented measurements, and each is subject to the same frozen RAM and RSS limits. An early stopped run retains its milestones. They do not identify all intermediate peaks.

The independent same-device audit compares recorded decoder input features,
logits, compact state tensors and named memory traces under the established
numerical tolerance and reports any output mask sign flips and their magnitude.
It reports per-frame driver changes and system memory, including increases or
no change; it must not assume the driver delta is all reclaimable memory or
attribute system-wide changes solely to this process. A cache-clear diagnostic
pass is not evidence of full-video compatibility or segmentation improvement.
No ground-truth labels, redetection, new prompt or inference-time gating is
used. Cache clearing is a fixed experimental operation, not a conditional
model decision; no production run will adopt it without separate validation.

This instrumented diagnostic saves CPU copies, observes decoder features, and synchronizes MPS. It records five outputs on seq11, while the interrupted full run stopped after six seq6 outputs; it cannot reproduce or explain the entire failed seq6 trajectory.

Each before/after frame sample immediately stops on exceeding its frozen RAM or RSS limit and preserves the violating sample. The supervisor uses the shared GPU lock and monitors disk free (at least
26 GiB), system-available RAM (at least 4 GiB), and child RSS (at most 6 GiB),
retaining partial logs on failure. Sampling cannot guarantee avoidance of an
abrupt memory-pressure event. Run no GPU diagnostic until code/test review,
a fresh resource preflight and a safe I/O handoff. A passing seven-frame run
must not be represented as a full seq6 or seq11 result.

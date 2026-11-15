# Bounded SAM2 frame-loader engineering gate

This is a separate opt-in diagnostic, not a change to the frozen warm capture
or the six-arm training study. The existing eager loader retains every
float32 1024×1024 RGB frame. Its asynchronous alternative also retains every
frame and normalizes its float64 decoded values before an eager-style cast;
it is not a drop-in solution to the RAM issue.

The proposed on-demand loader invokes the same pinned author decoder, casts
to float32 before the same mean/std normalization, and keeps a bounded CPU
LRU of float32 images. It returns a copy of a cached image so callers cannot
mutate retained data. The recorded limit excludes returned copies, decode
buffers, model state, MPS allocations and other process memory. The hook is
installed only while initializing the pinned predictor in a single-threaded
Python process under the shared GPU lock, then restored. Its pinned source is
checked for the expected integer-index read followed by transfer to MPS.
Input order, path and SHA are frozen, checked at decode and rehashed before
accepting outputs. This does not claim continuous filesystem immutability.

The engineering gate runs eager and bounded initialization/propagation in two
fresh, isolated processes with the same real seq11 prefix, warm box/points and
archived automatic mask. It checks source, checkpoint, prompt and clip
identity and compares output logits and recorded compact state tensors using
same-device tolerances, early/deep decoder inputs, plus binary masks and named selection/position/anchor
and memory-event traces. It records actual per-process high-water RSS, sampled
MPS allocator/driver memory and parent-observed minimum system available RAM
and free disk. No target label is opened. A short seven-frame clip limits the
eager memory requirement; passing is not proof of full-video equivalence,
long-sequence efficacy, total-memory bounds or CPU/MPS parity.

Each subprocess is stopped if configured minimum system available RAM or disk
space is crossed, or if process RSS exceeds its cap; partial files/logs remain
for diagnosis. Such checks reduce but cannot eliminate risk of a sudden system
memory-pressure event. Do not execute the gate until the shared GPU lock is
free, required disk/RAM headroom has been reviewed, and source/code review has
passed. Even after a passing short gate, integrating this loader into a new
capture requires a separately versioned runtime and an appropriate longer
same-device trajectory check. Never overwrite the existing frozen study.

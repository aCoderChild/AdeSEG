# Expanded development captures: seq6 and seq13

Research-only, specified before capture. Add two existing **development**
sequences as possible training inputs, never as independent evaluation. Keep
sequences 8, 11, 20 and 23 as previously inspected priority cases, and keep
seq9/10 out of this capture. The source manifest does not identify patients or
centers; sequence separation and absent exact-image duplicates cannot prove
patient independence or lack of detector-training exposure.

Use the archived original automatic YOLO box and feature-selected points,
one initialization, the existing multi-scale anchor, project token-window
fork, indexed MPS RoPE, blocking pointer-position transfer, FP32, a 16,384
represented-token capacity and 3,072 insurance reserve. Preserve original
frame order, no GT prompts, no later redetection or new inference gate. Run
the *entire* post-prompt sequence twice in separate processes: first a
reference with no feature/head capture, then the same run with read-only
feature/head observers. Compare every mask, token-window event, selection and
position trace before using the captures. Track all failures and budget totals.

Select target frames without looking at labels or predictions: ten evenly
spaced integer positions between first prompt + 1 and last frame inclusive:

- seq6, prompt 23: 24, 31, 39, 46, 53, 61, 68, 75, 83, 90.
- seq13, prompt 7: 8, 35, 62, 88, 115, 142, 169, 195, 222, 249.

Freeze targets and actual image hashes before inference. Observers capture the
32-channel early FPN feature, prompt-derived foreground/background prototypes,
native head inputs and outputs, object score and original geometry. They do not
change any tensors; verify observed early features equal the decoder's own
high-resolution feature input. Samples do not contain complete memory-attention
inputs and must not be labeled as full frozen memory reads. Original source
labels may be opened only **after both full trajectories are frozen and
independently audited**; they are for development training or retrospective
classification, not prompt/target selection or model inference.

Existing seq2/3/4 captures use CPU; these new captures are on MPS. The saved
seq4 fork replay provides a limited CPU/MPS distribution-shift check (12 frames,
features on two frames and full seq4 mask replay), not general equivalence.
Record device, dependencies and measured discrepancies; do not silently
equate these MPS features with the historical CPU pipeline or claim patient-
independent performance. Further device-aware training and provenance changes
need a new manifest and new parity check; do not reuse the old CPU pipeline
hash as if MPS were identical.

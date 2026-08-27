# Frozen MedSAM2 with a training-free dynamic probability state

This repository evaluates a narrow, training-free hypothesis: whether a
single causal decoder-logit state, optionally transported by optical flow, can
help a frozen MedSAM2 video segmenter without a hand-designed reliability
score. This is **not** proxy tuning and does not modify MedSAM2 weights.

All reported `Dice` and `IoU` values are averages over video-level scores, so
each sequence contributes equally regardless of its number of frames.

## Structure

- `reliability_method/reliability_gated_video_memory_experiment.py` runs the
  frozen MedSAM2 state conditions: direct, fixed blending, and the legacy
  adaptive-reliability baseline.
- `experiments/summarize_state_ablation.py` calculates equal-weighted
  per-sequence Dice/IoU and paired bootstrap intervals.
- `experiments/native_single_state_memory.py` tests full native memory against
  an initial anchor plus one frozen-encoded mutable memory item.
- `experiments/summarize_native_single_state.py` compares those conditions to
  full native memory with paired bootstrap intervals.
- `modeling/causal_prompt_adapter.py` contains the separate *trained* single
  causal-state adapter. It disables MedSAM2 native memory and turns the single
  state into a standard dense prompt; `experiments/train_causal_prompt_adapter.py`
  trains and evaluates it against stateless and neutral-prompt controls.
- `research/loop.py` applies fixed evidence gates and generates the paper
  draft and verification plan from the saved result.
- `tests/` covers the state-update and research-decision logic.

## Current evidence

The completed controlled 23-sequence ablation is in
`results/state_ablation/comparison.json`. Direct state plus flow is the best
reliability-free candidate at sparse prompts, but its Dice improvements are
+0.006 (stride 5) and +0.011 (stride 10), with 95% paired bootstrap intervals
that include zero. It is therefore a useful negative/early ablation, not a
paper-quality result. Run `python research/loop.py` to regenerate the full,
evidence-gated decision and editable research draft.

The separate native-memory single-state experiment is in
`results/native_single_state/comparison.json`. It rejects replacing the full
native memory bank with one mutable state for this frozen checkpoint and
PolypGen protocol; see `research/native_single_state_verdict.md`.

The newer learned no-native-memory adapter is implemented for the stricter
ablation, but is not yet a viable seed-only method: the frozen object-presence
gate can suppress the first unprompted frame and eliminate its learning signal.
See `research/causal_prompt_adapter_method.md`.

## Paper-grade protocol before a new model claim

1. Repeat the direct-state condition with native MedSAM2 memory enabled.
2. Separate foreground-positive sequences from all-sequence results and use
   the same prompt policy across every method.
3. Validate on an external dataset without tuning on its test labels and with a
   prompt source not trained on PolypGen.
4. Report paired per-sequence Dice/IoU intervals, failure cases, runtime, and
   memory use. Advance only when the predefined gates in
   `research/verification_plan.md` pass.

## Commands

```bash
python experiments/summarize_state_ablation.py
python research/loop.py --comparison results/state_ablation/comparison.json
python experiments/native_single_state_memory.py \
  --variant single_state_flow --prompt-stride 5
python experiments/summarize_native_single_state.py
python experiments/train_causal_prompt_adapter.py train --device mps --epochs 10
python research/loop.py --comparison results/state_ablation/comparison.json \
  --native-comparison results/native_single_state/comparison.json
python -m pytest tests -q
```

# Early-FPN-enriched temporal memory on Lean-SAM2 (experimental)

## Why this is a distinct experiment

This module tests whether *early backbone/FPN information can enrich the content of
compressed temporal memory*, rather than only reranking memory tokens.

The implementation and architecture notes were checked against
the diverged branch **backup/experiment-full-20261118**, notably
reports/prior_MODEL_ARCHITECTURE.md, reports/prior_ARCHITECTURE.md,
reports/prior_CONTRIBUTIONS.md, and reports/REPORT_FRAMEWORK_2026-11.md.
The backup is a reference; do not merge it into main to recover this work.

Distinguish three separate mechanisms:

- RMFA: learned early/deep feature aggregation changes current mask logits,
  optionally forwarding *refined masks* to the unchanged memory encoder.
- Pooled-anchor scoring: changes token importance rankings; no new memory content.
- This experiment: changes the native **64-channel memory feature vectors**,
  without increasing their number or changing their positional encodings.

The historical 0.742071 Dice Lean result did NOT evaluate this fusion.

## Source-compatible mechanism

The pinned Lean-SAM2 Hiera-B+ backbone has 256-channel FPN features.
Its memory encoder outputs 64-channel, mask-conditioned memory. It consumes
the lowest-resolution backbone features plus the predicted mask; the earlier
FPN maps do not become separate memory tokens.

The adapter runs the **native memory encoder first**, then projects and
downsamples the earlier FPN maps to the native memory grid. It combines the
projected maps, projected mask-conditioned memory and the native predicted
foreground probability with a lightweight convolutional mixer. Its last
convolution is initialized to zero, giving exact native-memory identity
at initialization. A bounded learned residual (default at most 0.25 times
per-entry memory standard deviation before dtype rounding) enriches
the memory; the optional native-no-object gate is ON by default.
All features have the exact same spatial coordinates as native memory.

    U_t = Mixer(Project(E_t^1), Project(E_t^2), Project(M_t), sigmoid(z_t))
    delta_t = 0.25 * std(M_t) * tanh(Head(U_t))
    M_t_enriched = M_t + presence_t * delta_t

No ground-truth mask or annotation enters inference. The pretrained decoder,
object pointers, TAMP/TCIM/TARR selectors, native spatial positional encodings,
and token dimensions remain unchanged. The runtime hook applies only to
non-conditioning track_step memory writes by default; directly invoked
_run_memory_encoder (including corrections) remains native.

The module DOES increase temporary feature computation and activations. Equal
token count does not establish equal latency, full GPU memory or cost of the
protected, conditioning and TCIM insurance banks. It does not claim CMSA-Net's
trained temporal cross-attention mechanism is reproduced.

## Files

- modeling/lean_memory_feature_fusion.py — trainable adapter and immutable config.
- adeseg/inference/lean_memory_feature_fusion.py — scoped native memory hook.
- tests/test_lean_memory_feature_fusion.py — CPU shape, parity, provenance,
  mixed-precision, no-object and causal routing tests.

The code is **opt-in only**. Existing baseline commands do not import or
enable this module; main's baseline outputs are unchanged by default.

For an untrained identity smoke:

~~~python
from modeling.lean_memory_feature_fusion import EarlyFeatureMemoryFusion
from adeseg.inference.lean_memory_feature_fusion import route_memory_feature_fusion

adapter = EarlyFeatureMemoryFusion().to(predictor.device).eval()
events = []
with route_memory_feature_fusion(predictor, adapter,
                                 identity_control=True, telemetry=events):
    # Call the existing, authenticated author Lean video inference here.
    # The host experiment runner is not part of this patch.
    ...
assert all(e['changed_elements'] == 0 for e in events)
~~~

Learned evaluation requires load_trained_state(payload, backbone_sha256=...,
train_manifest_sha256=...) before the hook can be installed. Check the
checkpoint *file* SHA256 independently before deserialization. The payload
requires exact config and finite tensor shapes, implementation digest over
the trainable + hook sources, verified backbone SHA256, frozen training
manifest SHA256 and a positive training-step count. Metadata verification
is not proof that a model was trained correctly.

**Important unfinished components:** There is no pretrained fusion checkpoint,
differentiable video training runner, full-video evaluator or completed
Lean-SAM2 GPU integration test in this patch. Author Lean inference is
decorated with inference_mode and cannot simply be used as a backpropagating
video-training loop. Do not report a training or Dice result from these tests.

## Predeclared ablations

Hold fixed the same backbone, automatic prompt source and first frame,
frames/resolution, state-reset rules, TAMP/TCIM/TARR parameters, source
group splits, seed, and **total measured memory-attention budget**.

| Arm | Memory representation | Token policy | Purpose |
| --- | --- | --- | --- |
| B0 | Native | Native | Lean reference |
| B1 | Zero-initialized fusion | Native | Identity/parity |
| B2 | Native | Pooled multiscale scoring | Token-ranking-only |
| B3 | **Trained early-FPN fusion** | Native | **Primary hypothesis** |
| B4 | Native + RMFA output-only | Native | Historical output-control |
| B5 | RMFA mask-feedback to native encoder | Native | Historical feedback-control |
| B6 | Trained early-FPN fusion | Pooled scoring | Interaction, only after B3 |

Mechanism ablations for B3:
(1) learned native-memory-only mixer, parameter-matched as far as possible;
(2) first early FPN only, second only, both;
(3) spatially shuffled early features;
(4) 0.1 vs 0.25 residual bound;
(5) native no-object gate ON vs OFF;
(6) conditioning memory preserved vs explicitly enriched.

B2 historical fixed-prompt results changed 578/578 retained-index records but
showed essentially no segmentation change (mean Dice ~0.49556 in both arms).
The historical 4-video RMFA memory-mask-only arm rose from 0.3882 to 0.3979
patient-macro Dice, with 108/120 rather than 107/120 empty-frame alarms.
Both records are exploratory, on previously examined sequences. Neither
is a result for B3; neither proves an improvement from the new module.

## Verification and promotion gates

1. Run Python syntax and CPU tests. Prohibit untrained learned evaluation,
   mismatch checkpoints, conflicting hooks, non-finite and wrong-shape inputs.
2. On a frozen short **DEV** video, check native vs B1 frame-level masks,
   logits, obj-pointers, spatial memory and positions, TAMP indices,
   TCIM protected bank, retained token counts and output hashes.
   Any discrepancy blocks further testing.
3. Implement source-authenticated *differentiable* causal training using
   TRAIN-only labels and frozen pretrained SAM2 weights. Define loss and
   teacher-preservation weight before DEV tuning. Do not use GT at inference.
4. Train at least 3 predetermined seeds, select hyperparameters on a
   group-disjoint DEV set, then freeze weights, prompts and code SHA256.
5. Evaluate B0, B2, B3, B4, B5 on exactly matched automatic prompts and
   patient/source groups. Do not compare oracle-initialized baselines with
   automatic variants as if matched. Previously examined PolypGen videos
   are development evidence, not pristine external TEST.
6. Primary metric: post-prompt per-sequence mean of GT-present Dice.
   Report per-sequence and frame-weighted Dice, boundary F, precision,
   recall, GT-empty false alarms, and temporal stability separately.
   Report confidence intervals over independent source groups, not frames.
7. Log actual memory_attention_input_tokens/bytes, protected and conditioning
   tokens, TCIM insurance tokens, token-score changes, peak GPU memory,
   p50/p95 latency and fusion overhead. Test on the target hardware.
8. Only after matched-budget positive DEV evidence, test untouched external
   adenoid/airway videos; PolypGen is a colonoscopy proxy, NOT clinical
   evidence of adenoid segmentation.

## Test commands

~~~bash
python -m compileall -q modeling/lean_memory_feature_fusion.py adeseg/inference/lean_memory_feature_fusion.py tests/test_lean_memory_feature_fusion.py
python -m pytest -q tests/test_lean_memory_feature_fusion.py
python -m pytest -q tests
~~~

Passing synthetic CPU tests does not prove a real video, GPU or segmentation result.

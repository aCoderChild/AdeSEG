# Token-window Lean: CPU and MPS replay audit

This is an engineering check on seq4 frames 9–20, not a segmentation-quality
comparison or an authorization to install a learned refiner. The automatic
box and two feature points are reused from the archived prompt. No ground-truth
labels are read to choose the prompt, frames, memories, or outputs. CPU and MPS
trajectories use the original 48-frame sequence length, even though this check
stops at frame 20.

The replay uses the `external/Lean-SAM2-token-window` fork at author commit
`8d1cf2618519d291905d22fe859b8302476d7f50` with its two recorded patches,
the full released Lean base-plus config and bypass weights, a 16,384-feature-
token spatial bank with 3,072 tokens reserved for the insurance bank, and
multiscale anchor strength 0.5. As in the archived capture,
`ema_history_replace_old_tail_in_attn` and `ema_history_resident_prune` are
disabled. These choices form a project variant and must not be called original
Lean. The original model's TAMP/TCIM/TARR machinery is enabled.

The CPU path uses `modeling/lean_indexed_spatial_rope.py`. The MPS path uses
the separate, experimental `modeling/lean_indexed_spatial_rope_mps.py` for
exact-checked spatial position reconstruction and
`modeling/lean_mps_blocking_pointer_position.py` for the demonstrated
temporary CPU-to-MPS pointer-position transfer race. The author checkout is
not edited by either scoped adapter. Tensor copies for engineering receipts
synchronize execution, and temporary frame staging uses symlinks; all saved
results and captured feature tensors reside on the user's Google Drive. The MPS
adapter has only been tested with FP32, a single object, unmerged attention
keys and the configurations described here. Blocking one pointer transfer does
not validate every upstream nonblocking transfer or CPU-offloaded state.

## Exact identities

| Input or component | SHA-256 |
| --- | --- |
| SAM2.1 base-plus checkpoint | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |
| Archived seq4 prompt | `67e1360bd0c2b59387334c217ae7af7d197ba07aea83749733aa959300e4e16f` |
| Archived capture receipt | `7d8038e5659a4bb09d192b38dc9008ae55ac00607997ea8d4a630d644164802e` |
| Fork `sam2/modeling/sam2_base.py` | `a5794b585c6949fcac640787d4935ad1dcb7fd6049dc75f02916aadf7685df01` |
| Fork `sam2/sam2_video_predictor_topk_insurance.py` | `406854c686f47498ddb144f398cf0ca0a1c6cc850b0127e25b1be0d2f938d5c8` |
| Lean config | `1c153a6c974d79c14e1ee4a56e5f3ee5dc263e46f7c965edd288ca719bb4d84d` |
| Lean base-plus bypass checkpoint | `5e415e3d207f09ffe94216b1472a83518de5c54425ab55bdef8f331761af1c65` |
| Multiscale-anchor adapter | `017cb250608d13da8c5520558d0b80f21a7274e2e01577ad2a449d5d3929ef19` |
| Token-window policy | `a6c399b1543dee54c92a3f05d1e809c61455cce48b27ccb7be5e70bcc126755b` |
| Token-window runtime | `2ae8692cd345c4657907954721cb2ee878a4f941d8c3c3f2f2e9649f885563f2` |
| CPU indexed-RoPE adapter | `f1e691538ffc8da3512a7eaf181010f81cba24d2ed9bcd8e292792510742823d` |
| MPS indexed-RoPE adapter | `43da1f1e8cecca7382305f4d053a8c106030c0c8cb706011b06bd7aad7cc0721` |
| MPS blocking pointer-position adapter | `9ef35d07f0fb743025348a9650c1f9bbf27e79674ad9ed06bf258b87e87a08cc` |
| Replay runner | `a36df17c7419b2297da2501141979fe89600385b931e26ff3063b2f8a5f018d4` |
| Comparison runner | `c87db3b35d09a9129813c5bccd093b1b7c66e33f25ccf3bb3250a268285f69f1` |

The tested environment was macOS 15.8.3, PyTorch 2.12.0, FP32 and two CPU
threads. Both MPS trials used `PYTHONHASHSEED=0`, `PYTORCH_MPS_FAST_MATH=0`,
`PYTORCH_ENABLE_MPS_FALLBACK=0`, deterministic algorithms and fixed Torch/MPS
seed 20261021. Input images are checked against the archived image hashes.

## Reproduction

From the repository root, use a new output directory name for each invocation:

```bash
BASE='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment'
PRE="$BASE/lean_channel_attention_training_pilot_v1/split_preflight_attempt03"
REF="$BASE/lean_utility_training_capture_v1/seq4_capture_attempt01"
OUT="$BASE/contrast_refinement_v1/window_reproduction_NEW"
for RUN in cpu mps01 mps02; do
  DEVICE=mps
  if [ "$RUN" = cpu ]; then DEVICE=cpu; fi
  PYTHONHASHSEED=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
    .venv/bin/python scripts/verify_lean_window_mps_replay.py \
    --checkpoint "$BASE/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" \
    --capture-reference "$REF" --preflight "$PRE" --device "$DEVICE" \
    --end-frame 20 --output "$OUT/$RUN"
done
.venv/bin/python scripts/audit_lean_window_mps_replay.py \
  --cpu "$OUT/cpu" --mps-a "$OUT/mps01" --mps-b "$OUT/mps02" \
  --reference "$REF" --output "$OUT/comparison"
```

The focused suite completed with 27 passing tests:

```bash
PYTORCH_MPS_FAST_MATH=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python -m pytest -q \
  tests/test_lean_mps_blocking_pointer_position.py \
  tests/test_lean_indexed_spatial_rope_mps.py tests/test_lean_rope_mps_repeat_fix.py \
  tests/test_lean_indexed_spatial_rope.py tests/test_lean_rope_cpu_repeat_fix.py \
  tests/test_lean_contrast_refinement_integration.py \
  tests/test_lean_token_window_storage.py tests/test_token_memory_window.py
```

## Result and interpretation

The CPU replay matched all 12 archived masks, 66 token-window events, and
early/deep/logit tensors at frames 10 and 19 exactly. Two fresh-process MPS
runs matched each other for all recorded masks, the sampled tensors, sorted
selection events, token-window events and position-index hashes. Each run
reached 16,384 total stored spatial/insurance tokens and made five eviction
decisions. The stated budget does not include pointers, positions, caches,
auxiliary tensors or transient allocations.

CPU versus MPS had one binary-mask pixel difference in each of frames 15, 16
and 19; the other nine masks matched. Minimum per-frame IoU was 0.9999532 and
minimum Dice was 0.9999766. Across frames 10 and 19, the largest mean/max
absolute differences were 7.1e-8/1.7e-6 for early features,
9.5e-6/0.00165 for deep features and 8.8e-5/0.00537 for logits. Selected token
sets and token-window events matched, but cross-device indexed-key order hashes
did not always match. No equivalence tolerance was prespecified, so these
measurements must not be relabeled as a formal equivalence test.

The recorded comparison is at Google Drive
`contrast_refinement_v1/lean_window_replay_comparison_attempt01/comparison.json`;
individual replay receipts and sampled tensors are in the corresponding
`lean_window_replay_cpu`, `lean_window_replay_mps01` and `lean_window_replay_mps02`
directories. The test samples only one video and two feature frames. It does not
prove all MPS operators, all clips, all source-memory layouts, learned-refiner
routing, full-length repeatability, absence handling or segmentation efficacy.

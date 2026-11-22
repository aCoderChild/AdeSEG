# Literature-guided corrections to the Lean-SAM2 multiscale memory study

**Code status:** implementation-level changes; NO held-out segmentation or GPU
speedup claims. **Papers motivate the research choices; they do not validate
this adapter or guarantee a Dice improvement.** Use the pinned Lean source and
source-bound checkpoints when comparing outcomes.

## Literature grounding (what each paper actually supports)

1. **SAM 2: Segment Anything in Images and Videos (Ravi et al., 2024).**
   https://arxiv.org/abs/2408.00714.
   A promptable image/video architecture with a streaming memory bank.
   Its direct **box/point** initialization runs a prompt encoder and mask
   decoder. A mask-input initialization is a distinct model path, and the
   pinned Lean Hiera-B+ configuration uses
   \`use_mask_input_as_output_without_sam=true\`. Thus a predicted binary mask
   is not representation-equivalent to the original detector box+points.
   **Correction:** save BOTH prompt modalities, use \`--initialization-mode
   direct_prompt\` for the primary end-to-end RQ1/RQ2 track, and retain
   \`mask_reprompt\` as an explicit separate control.

2. **CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source
   Reference for Video Polyp Segmentation (MICCAI 2026).**
   https://papers.miccai.org/miccai-2026/0176-Paper1232.html.
   Motivates causal use of complementary spatial and temporal
   representations. The current adapter uses **current-frame early-FPN
   multiscale pooling** to *write* memory, NOT CMSA-Net's cross-frame
   multi-scale attention. This distinction is intentional and should be
   isolated using \`no_early\` and two single-scale controls. Spatial
   pooling may still discard thin boundaries; this requires measurement.

3. **Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2
   (2026).** https://arxiv.org/abs/2607.19811.
   TAMP uses prompt-derived foreground anchors plus attention usage;
   TCIM and TARR are additional native mechanisms. Our study keeps these
   components intact. A bounded spatial residual does NOT guarantee stable
   Top-K scores. Check retained indices and an explicit cosine-drift report.

4. **Putting the Object Back into Video Object Segmentation / Cutie
   (Cheng et al., CVPR 2024).**
   https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html.
   Establishes that object-level representation and memory *reading*
   matter in VOS. It motivates testing whether the written multiscale
   correction actually reaches future attention, not just whether its
   BF16 bytes differ. Cutie's object-level reader is NOT adopted here.

5. **XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin
   Memory Model (Cheng & Schwing et al., ECCV 2022).**
   https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/568_ECCV_2022_paper.php.
   Supports distinguishing bank write, read, compression, and long-term
   retention. It does not establish that our enriched SAM2 bank is superior.

## Corrected initialization: same pinned Lean model, two testable paths

Primary direct mode:

\`\`\`text
image -> YOLO box -> frozen FGPI points from Lean early FPN
                      | 
                add_new_points_or_box(...)
                      |
             native SAM2 decoder + conditioning state
                      |
           native or enriched future memory
\`\`\`

Control (legacy mask-reprompt handoff):

\`\`\`text
image -> YOLO box + FGPI points -> initial mask logits
           -> thresholded binary PNG
           -> add_new_mask(...) in a NEW video session
           -> independent conditioning state
\`\`\`

They share detector boxes, frames, backbone, frozen export and GT-free
initialization. They do NOT have equivalent learned object pointers.
The frozen binary initial mask is retained as an integrity witness:
replaying the direct prompt must reproduce it bit-for-bit in the pinned
environment or the run fails. The saved original-resolution direct box
and point coordinates are validated against the source JPEG.

**Important:** training capture and DEV/TEST propagation MUST have the
same \`--initialization-mode\`. Checkpoints authenticate the mode. Run both
direct and mask modes as independently trained controls, not by switching
an adapter after training.

## Corrected TRAIN teacher: GT-consistent object presence

Let \(\widehat z_t\) be Lean's predicted object-presence score and
\(y_t\) the TRAIN-only GT binary mask. Native mask-conditioned memory
may include an explicit learned no-object embedding \(e_\varnothing\):

\`\`\`math
M_t^{native} = E_{mem}(D_t,\widehat Y_t,\widehat z_t),
\qquad
M_t^{teacher,GT} = E_{mem}(D_t,Y_t,z_t^{GT}),
\qquad
z_t^{GT} =
\begin{cases}
+10 & \text{if } \sum Y_t > 0,\\
-10 & \text{otherwise.}
\end{cases}
\`\`\`

An incorrect negative \(\widehat z_t\) must not be silently used as the
GT-presence teacher on a GT-present frame. The default TRAIN-only
\`--teacher-presence-mode gt_mask\` corrects this. The former behavior
remains \`--teacher-presence-mode predicted\` as an explicitly named
supervision control. The teacher is never written into the live bank.

The adapter still contains a conservative hard predicted-absence gate.
On false-negative frames, this gate produces zero residual and therefore
cannot learn to correct absence. The existing
\`--allow-no-object-memory-fusion\` control can test this failure mode,
with GT-empty false alarms monitored. Do NOT claim false-negative recovery
from the default hard-gated model.

The clipped teacher is the coordinate-wise Euclidean projection into
the **bound** \(|\Delta_{t,j}|\le .95\alpha \sigma(M_t)\). This does not
prove that the CNN can realize every target inside the box. It also does
not address the teacher-forcing / rolled-out-history distribution shift.

## Training and evaluation objectives

Current objective (same-frame memory distillation):

\`\`\`math
\mathcal L_{memory}
= \operatorname{SmoothL1}\big(
\widetilde M_t,\Pi_\alpha(M_t^{teacher,GT})
\big)
+ \beta \operatorname{SmoothL1}(\widetilde M_t,M_t).
\`\`\`

A future-frame segmentation objective is conceptually more aligned with
the clinical goal:

\`\`\`math
\mathcal L_{future} =
\sum_{k=1}^K w_k \left(
1-\operatorname{Dice}(
\operatorname{SAM2}_{t+k}(\widetilde{\mathcal B}_{\le t+k-1}),
Y_{t+k})
\right).
\`\`\`

**The future-frame objective is NOT implemented.** Its formula is a
possible follow-up study, not a claim about current trained checkpoints.
It needs differentiable unrolling and careful handling of discrete native
TAMP/TCIM decisions. The native upstream components must not be silently
replaced to implement it.

## Efficiency and mechanistic diagnostics

The original accuracy runner includes per-frame mask PNG I/O, repeated
device synchronization for telemetry, video hashes, memory bank snapshots,
and optional TAMP index writing. Its \`propagation_seconds\` and GPU peak
are **diagnostic observations**, not fair method-speedup evidence.

A new independent CUDA-only
\`scripts.benchmark_lean_memory_latency\` runner measures synchronized
repeated propagation **without** per-write telemetry or PNG output.
Use identical DEV video IDs, backbone, prompt mode, device, warmups, and
repeats for native/identity/full, with interleaved arm order for a stronger
publication benchmark. It excludes prompt production cost, which must
be reported separately for a full-system end-to-end speed claim.

Even this clean benchmark is not an externally validated speed result.

## Supported executable commands

After generating GT-free YOLO detections and frozen direct-prompt metadata:

\`\`\`bash
# Primary RQ1/RQ2 track: native prompt-decoder semantics, no GT in inference.
python -m scripts.run_lean_memory_fusion_ablation \
  --arm native --initialization-mode direct_prompt \
  --require-verified-prompt-producer \
  --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --split dev --device cuda --output "$OUT/native_direct"

# TRAIN capture must use direct_prompt and GT-consistent teacher presence.
python -m scripts.capture_lean_memory_train \
  --manifest "$TRAIN_MANIFEST" --prompt-manifest "$TRAIN_PROMPTS" \
  --group-map "$GROUP_MAP" --split-file "$SPLIT_FILE" \
  --checkpoint "$BACKBONE" --backbone-sha256 "$BACKBONE_SHA" \
  --initialization-mode direct_prompt \
  --teacher-presence-mode gt_mask \
  --device cuda --output "$OUT/direct_gt_captures"

# Retrain all learned modes using the NEW capture manifest and source digests.
python -m training.lean_memory_fusion \
  --manifest "$OUT/direct_gt_captures/train_memory_captures.json" \
  --split-file "$SPLIT_FILE" --input-mode full \
  --teacher-target-policy feasible_clipped \
  --backbone-sha256 "$BACKBONE_SHA" --device cuda \
  --output "$OUT/direct_full.pt"

# Use the matching direct-prompt learned checkpoint in --arm fusion.
\`\`\`

For latency, hold source and frame cohort fixed:

\`\`\`bash
python -m scripts.benchmark_lean_memory_latency \
  --arm native --initialization-mode direct_prompt \
  --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --split dev --warmups 1 --repeats 3 \
  --output "$OUT/native_direct_latency.json"
\`\`\`

## Hard boundaries

- No claim of experimentally improved Dice, temporal stability, TAMP
  relevance, runtime or VRAM without GPU and held-out results.
- No GT masks enter automatic prompt export or video inference.
- No new temporal memory tokens are created by the adapter.
- No claim that same-frame oracle-memory distillation learns optimal
  future-frame representations.
- The native initial memory remains unmodified by default.
- Prior adapter checkpoints/captures that lack new presence/initialization
  provenance must be regenerated instead of bypassing checks.

# CMSA-Net idea for SAM2 ToMe — review, 2026-11-13

## Decision

Use CMA only as literature motivation for causal temporal context. The official CMSA-Net code is now cloned and inspected, but its module is not a drop-in SAM2 token method. Do not transplant CMA or DMR into this frozen SAM2 study.

## Terminology

- **CMA** in CMSA-Net is a learned causal temporal-attention module. It aggregates multi-scale features from historical video frames.
- **DMR** is a separate dynamic reference-frame selection strategy. It chooses reference frames using semantic-separability and confidence criteria. This conflicts with the current protocol, which retains SAM2's predetermined temporal memory schedule and does not adaptively choose good or bad frames.
- **SAM2 already has multiple references.** Its spatial memory attention concatenates memory maps from prior frames selected by its fixed temporal schedule. In the Hiera-T config, `num_maskmem: 7` means the current frame plus up to six past-frame memories.
- **Object pointers are additional reference tokens, not the reference frames or a multi-scale pyramid.** SAM2 extracts each pointer as a `[B, C]` vector from a mask-decoder output token and adds temporal position information before appending it to memory attention. Keep these pointer tokens intact.

## Implication for this project

Pointer-conditioned ToMe has already completed a 21-sequence comparison and failed the predeclared improvement screen: native mean present-frame Dice `0.710224`, pointer-conditioned ToMe `0.710213`, matched generic ToMe `0.710073`. Do not repeat that idea under a new name.

SAM2 already performs **within-frame** multi-scale fusion: in the Hiera-T config, FPN levels 2 and 3 correspond to Hiera stages 3 and 4; the stage-3 map receives an upsampled stage-4 feature. `scalp: 1` then drops the standalone lowest-resolution stage-4 output, while retaining its contribution inside stage 3. Stage-1 and stage-2 maps remain high-resolution inputs to the mask decoder. The memory-attended image feature is the retained stage-3 map. This is distinct from CMA's temporal aggregation across historical frames: SAM2's memory bank stores one spatial map per memory frame, not the separate multiscale historical maps used by CMA.

A distinct candidate may use **scale context to choose spatial ToMe pairs while retaining the reference frames selected by SAM2's predetermined schedule**. The selected past frames naturally advance as video progresses; they are not one immutable set for the whole video. To remain compatible with the current goal, the method must leave SAM2 weights, memory writes, decoder, frame schedule, current-frame queries, and object pointers unchanged; merge spatial K/V only; and use a matched generic ToMe control with identical per-frame token budgets. Deriving scale context from existing spatial memory is an adaptation, not a reproduction of CMA.

Before coding, specify exactly how multi-scale descriptors affect pair matching and confirm that positional and temporal encodings remain aligned. After coding, require native/no-op parity, pointer and memory-slot preservation, K/V alignment, unchanged weights/banks, and independent-process replay. Only consider efficacy evaluation after the active cohort finishes and the design is frozen. Do not add DMR-style frame selection or confidence gates.

## Adaptive token utilisation — prior-experiment overlap

The user's intended distinction is a native predetermined frame schedule with token decisions that can depend on the current frame. The schedule advances through the video; this does not require treating each frame as wholly useful or harmful.

However, generic query-conditioned token use is not a new, untested direction in this repository:

| Existing experiment | Current-query signal | Evidence and limit |
|---|---|---|
| Run170 | First-read attention, top-k retrieval, fixed prompt anchors and soft role affinity | 73 fixed-native, GT-present targets on seq2/seq3; regional selection was −0.003618 Dice vs native. No full propagation or empty-frame safety denominator. |
| Attention-aware MCTF Seq5 diagnostic | Mean incoming attention over current queries guides the author MCTF matching operation; merged values are restored to dense spatial slots | Attention-aware Dice 0.541826 vs native 0.541987 and uniform-importance MCTF 0.541966. It did not alter the 85 GT-present zero-Dice frames. This is one post-hoc sequence, not a general rejection of query conditioning. |

Therefore the broad recommendation “try query-conditioned merging next” is insufficient. Before any new implementation, specify how its matching objective differs from mean-attention importance and prior top-k protection, identify the failure it addresses, and justify a matched fixed-budget control. Attention relevance alone does not certify beneficial segmentation information. No efficacy process is active after the calibration-control defect was found; do not relaunch MCTF or Run170 under another name.

Code/evidence: `modeling/mctf_memory_value.py::_mean_attention_over_queries`, `protocols/PRIOR_EXPERIMENTS.md`, and the completed Seq5 MCTF record in `protocols/ACTIVE_RESEARCH_GOAL.md`.

## Sources

- Wang et al., [CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation](https://arxiv.org/abs/2602.22821), arXiv:2602.22821 (2026); [official implementation](https://github.com/wangtong627/CMSA-Net).
- SAM2 source: [`sam2_base.py`](../external/sam2/sam2/modeling/sam2_base.py), [`memory_attention.py`](../external/sam2/sam2/modeling/memory_attention.py), and [`sam2.1_hiera_t.yaml`](../external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml).
- Existing pointer-conditioned ToMe cohort record: [`REGION_TOKENS_RESULTS.md`](../docs/REGION_TOKENS_RESULTS.md).

## Official repository audit — 2026-11-13

- Repository: [`wangtong627/CMSA-Net`](https://github.com/wangtong627/CMSA-Net), cloned to `external/CMSA-Net` at commit `4e4602e93211b5d0006a8a7415e8477de413b79d`. The checkout is clean.
- `cmsa_project/lib/model_arch/cmsa_net.py::X2RoleMultiScaleTemporalCrossAttn_AnchorsCausalVarT` aligns three Res2Net feature maps (`x2`, `x1`, `x0`) with learned convolution blocks, uses learned Q/K/V/out projections and a learned softmax `scale_gate`, and updates the `x2` feature map through temporal attention plus residual/FFN paths. CMSA-Net then uses the modified features for its mask guidance and decoder. This is a trained, image-encoder/decoder-side architecture, not a training-free merger over SAM2's stored K/V tokens.
- Correction after inspecting the full released test path: the video dataloader initializes two references (`f0`, `f1`), and `scripts/my_test.py` subsequently updates them using semantic and confidence criteria. The earlier statement that the revision lacked a dynamic selector was incorrect. This establishes what the released test code does, not an independent validation of the full published DMR results.
- CMSA reference frames are image/feature references. SAM2 object pointers are separate decoder-output vectors appended to memory attention with temporal position encoding; a pointer is not itself a selected reference frame.
- **Integration decision:** no CMSA code was copied into SAM2 and no inference was run. The source's learned feature projections, 32-channel pyramid, temporal attention, mask-guidance path, and decoder depend on CMSA-trained weights and differ from SAM2's post-RoPE spatial K/V interface. Porting its idea would require a new adaptation/training protocol and would violate the present training-free, frozen-module objective. The user-requested upstream-first rule is satisfied by cloning and auditing the official source; it does not justify forcing an incompatible module into SAM2.

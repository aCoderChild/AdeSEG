# Efficient-SAM2 SMR source audit — 2026-11-13

## Finding

The local author checkout contains a complete top-k memory-token retrieval path. My initial source search was too narrow: it stopped at `sam2/` and missed the consumer in `build_model/window_bypass.py`. The earlier claim that masks were never applied was incorrect and is superseded by this audit.

## Verified path

1. `tools/vos_inference_main.py` initializes the predictor's memory bookkeeping and calls `build_WB_model` when `--apply_WB` is enabled. The inference script exposes `--prune_memory`, `--topk_mask`, and `--set_drop_ratio`.
2. `sam2/modeling/sam2_base.py` captures memory-attention activations. With top-k mode it averages attention over query positions and creates a fixed-count spatial mask with `get_topk_mask`; the mask is saved in the per-object FIFO `mem_masks_stack`.
3. `sam2/sam2_video_predictor_with_bypass.py` assembles the masks for the memory-attention layers and exposes them through `mem_info['mem_masks']`.
4. `build_model/window_bypass.py::WB_RoPEAttention_ca.forward` applies the mask to spatial K and V tokens before attention. `apply_rotary_enc_prune` uses the retained original positions for rotary encoding, rather than assigning compressed tokens new grid positions.
5. The inference command in the repository README enables SMR with `--prune_memory --topk_mask --set_drop_ratio=0.95`.

Relevant source locations in the pinned local checkout:

- `sam2/modeling/sam2_base.py`: `get_topk_mask`, mask capture and FIFO append.
- `sam2/sam2_video_predictor_with_bypass.py`: mask-queue assembly.
- `build_model/window_bypass.py`: K/V filtering and positional-frequency indexing.
- `tools/vos_inference_main.py`: predictor setup and SMR arguments.

Local repository revision: `8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5`. `sam2/sam2_video_predictor.py` had pre-existing local modifications and was not edited.

## What it means for this project

SMR is a real token-level memory-attention method, but it is **hard top-k pruning**: with `set_drop_ratio=0.95`, the selector retains 5% of the scored 4096-token spatial map. It uses attention-derived saliency and reuses masks over time. It does not merge tokens, and it is not a Seq5 reappearance method. The official paper presents it as an efficiency method and reports 1.68× speedup with a 1.0% accuracy drop on SA-V test; it does not establish a Dice gain over native SAM2.

This is therefore useful as a literature/code reference, not a validated repair for Seq5. It conflicts with the project's standing no-gating constraint because it removes tokens by a hard top-k mask. No local adaptation or experiment was launched. If that constraint changes, the upstream `tools/vos_inference_main.py` path can isolate SMR with `--apply_WB --disable_WB --prune_memory --topk_mask`; leave `--Mem_Frame_Prune` off. First run an isolated source-parity and no-op gate; do not implement a second pruning operator from scratch.

Paper: [Efficient-SAM2 (ICLR 2026)](https://arxiv.org/abs/2602.08224). Official code: [jingjing0419/Efficient-SAM2](https://github.com/jingjing0419/Efficient-SAM2).

## Closely related 2026 work

Puspitasari et al., [“Accelerating SAM2 With Efficient Memory Attention Module via Spatiotemporal Token Pruning”](https://doi.org/10.1109/TPAMI.2026.3723878), directly studies SAM2 memory-attention token pruning. Its abstract reports pruning up to 87% of memory tokens and 1.7× per-frame acceleration with negligible performance disturbance across its evaluated benchmarks. The described method scores tokens for boundary delineation and temporal trackability. This supports the premise that memory tokens have unequal task utility, but it remains pruning (hard exclusion) aimed at speed, not ToMe merging or a demonstrated PolypGen Dice improvement. I found no official repository for this paper in the source search performed on 2026-11-13, so its code cannot be reused or independently inspected here.

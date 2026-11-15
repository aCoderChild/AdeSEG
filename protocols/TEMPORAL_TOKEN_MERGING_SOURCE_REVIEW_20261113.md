# Temporal token-merging source review — 2026-11-13

## Decision

Do not implement either reviewed method as a Seq5 repair or launch an efficacy run from this review. Neither is a faithful, no-threshold SAM2 memory-attention method, and the Seq5 evidence does not show that merging caused the prolonged loss.

## Source findings

| Source | What the official method does | Fit to this project |
|---|---|---|
| [Multi-Granular Spatio-Temporal Token Merging (STTM), ICCV 2025](https://arxiv.org/abs/2507.07990), official code at [`HYUNJS/STTM`](https://github.com/HYUNJS/STTM), inspected at commit `336bf360118b1cde3caa8e0a826f05e15ab91a83` | Builds multi-granular spatial nodes, then merges spatially corresponding nodes across adjacent frames when cosine similarity exceeds a configured temporal threshold. It can aggregate feature and positional components. Evaluated for video-language QA. | Temporal summaries are conceptually relevant, but the thresholded merge violates this project's no-threshold rule. The code and results do not establish dense video-segmentation gains, nor compatibility with SAM2's post-RoPE K/V and per-frame temporal positions. Do not port the threshold rule or present it as validated for VOS. |
| [Video, How Do Your Tokens Merge?, CVPR 2025 Workshop](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Pollard_Video_How_Do_Your_Tokens_Merge_CVPRW_2025_paper.html), official code at [`sjpollard/video-how-do-your-tokens-merge`](https://github.com/sjpollard/video-how-do-your-tokens-merge), inspected at commit `267fb4915c0dcb55f87f555fd72dbb6df4a618b7` | Applies fixed-budget ToMe-style reduction inside video-transformer encoder layers (including ViViT/TimeSformer variants), with weighted merging and unmerging support. Evaluated for action recognition. | This is closer to the fixed-budget constraint but is still the same broad ToMe family already tested in this repository. It does not merge SAM2's stored memory bank or test dense VOS. Adapting its encoder patch to SAM2 memory would be new experimental glue, not a reproduction, and its results do not support a Dice claim. |

## Seq5 evidence that bounds any proposed fix

Seq5's 85 zero-Dice frames are shared by native and tested ToMe/value arms. On frame 157, the raw decoder mask has Dice `0.602104`, but the object-presence logit is `−0.8830`; the native no-object rule suppresses it. On frame 184, the raw mask has zero GT overlap and about `10.9×` the GT area, with presence logit `−1.9156`; suppression prevents a large false positive. The pinned SAM2 code applies this presence decision before memory writing and inserts a no-object spatial embedding for absent predictions.

Already completed checks include generic/regional ToMe, same-frame and prompt-pointer ToMe, MCTF attention-aware value fusion, fixed-budget wider history, and 160 fixed-bank single-tile removals at frames 157/184. None restores the long loss; the removal diagnostic is deterministic but post hoc and not a deployable selector. Therefore another Seq5-specific merge heuristic would be outcome-driven tuning without evidence that token merging is the cause.

## Next decision

Keep the Seq5 failure as a common-mode native limitation, not as a reason to bypass the presence head. Continue the primary study only with a distinct token-level method that fits the frozen schedule, no-gating constraint and fixed budget, and that has a source-backed mechanism plus a passing engineering gate. If a proposed temporal merge must average different SAM2 spatial or temporal positions, first demonstrate correct K/V, RoPE, mass and source-position semantics on a real-frame gate; do not infer segmentation utility from video-QA or action-recognition results.

No SAM2/model code was changed and no inference was run during this source review. Local pinned source snapshots are in `external/STTM/` and `external/video-token-merging/`; these are research references, not integrated dependencies.

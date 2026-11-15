# Memory-token accuracy candidates — upstream/source screen (2026-11-13)

## Decision

No new candidate is ready for implementation or an efficacy run. The closest recent papers either optimize speed with accuracy preservation, alter the fixed frame/write protocol, or do not expose usable upstream code. Do not recreate their operators from paper text.

## DTMFormer upstream audit

- Official source: `external/DTMFormer`, MIT license, commit `47d28fea4fa2e92ce39b5f521245b58a4540753b`; worktree clean at inspection.
- `models/model_utils.py::ATM` projects feature tokens with a learned convolution, scores them with a learned linear layer, then calls `cluster_dpc` and `merge_tokens`. The clustering uses feature distances plus stored attention-distribution distances, and performs weighted token averaging.
- `models/SETR.py` runs those modules inside a SETR encoder with learned attention/projection blocks. Its `LTR` recovery/fusion maps the sparse representation back toward the dense spatial grid before the segmentation decoder.
- This supports the broad medical-segmentation intuition that task-aware grouping and spatial reconstruction can coexist. It is not a frozen, training-free operation: its projections, scores and attention blocks are learned as part of SETR. A checkpoint trained for SETR/ACDC cannot be treated as SAM2 memory-attention weights.
- The inspected SETR forward contains hard-coded `.cuda()` device transfers, which would need correction before CPU/MPS use. Because this model is not being integrated into the project, the upstream source was left unmodified.
- Source SHA-256: `models/model_utils.py` `55f5e7c5f7e6aaef0ee4aac8504749e73d9278ced48f7bb8e14b639529d9c774`; `models/SETR.py` `6c21e1a9e44bd881d7eec8caba4d943a0915cfb7eeace51128ca8ebef1fba46f`; `LICENSE` `eb7d3cc18aafaa3594fb2983a5835875576b289866adfe35e2bf96142fec8c2c`.

## Candidate screen

| Candidate | Published mechanism and result | Upstream/code status | Fit with this study |
|---|---|---|---|
| Puspitasari et al., SAM2 spatiotemporal memory-token pruning, TPAMI 2026 | Layer-wise memory-token scores target boundary delineation and trackability; paper reports up to 87% token compression and 1.7× speed with negligible accuracy disturbance. | Targeted exact-title/DOI/author searches did not identify official code. | Directly operates on SAM2 memory tokens, but targets efficiency, not meaningful Dice gain. No reimplementation from prose. Details: `SAM2_SPATIOTEMPORAL_TOKEN_PRUNING_REVIEW_20261113.md`. |
| RS³-Prune, arXiv:2608.22526 | Training-free read-side spatial gating and write-side memory-token pruning. The prior propagates from the previous predicted mask. Its SAM2 DAVIS-17 J&F is 0.882 → 0.797; on SA-V, 0.465 → 0.506; on YouTube-VOS, 0.588 → 0.636; on MOSE, 0.398 → 0.502; on LVOS-v2, 0.769 → 0.660. | The arXiv source has no code link. | Not compatible: sparsifies current-frame queries and changes which tokens enter memory. Results are mixed across benchmarks and use J&F, not PolypGen Dice. |
| DTMFormer, AAAI 2024 | Attention-guided, feature/dependency-based clustering merges tokens into semantic tokens and reconstructs features to the dense grid; paper reports lower complexity and improved trained medical-image segmentation. | Official MIT repository cloned to `external/DTMFormer`, commit `47d28fea4fa2e92ce39b5f521245b58a4540753b`. | Useful conceptual evidence for segmentation-aware merging and dense-grid reconstruction. Its ATM contains learned projections/score and its SETR encoder is trained end to end; weights/module do not fit frozen SAM2's post-RoPE memory K/V. No direct transplantation or PolypGen claim. |
| Progressive Token Merging (PTM), 2026 | Two-stage medical-image token grouping/matching aims to avoid premature inter-class merges and records a map for dense-position reconstruction. | Publisher cites `https://github.com/wuwen1994/PTM`, but `git ls-remote` returns “Repository not found”; no usable source clone. | Useful caution about class-boundary mixing, but unavailable upstream code and single-image ViT scope. No recreation from prose. |
| ERMU, CVIU 2026 | Relevance–redundancy memory-bank updates plus smoothed, quantized Q/K attention; the paper reports improvements on multiple VOS benchmarks. | Publisher points to `https://github.com/Shine-Ray/ERMU`, but `git ls-remote` returns “Repository not found”; targeted searches found no replacement repository. | Not compatible with the fixed native memory schedule/write path; no usable upstream source to adapt. Do not recreate it. |
| Efficient-SAM2 SMR | Attention-derived hard top-k K/V pruning and temporal reuse of pruning masks; official local source is already audited. | Existing checkout: `external/Efficient-SAM2`; source audit: `EFFICIENT_SAM2_SMR_SOURCE_AUDIT_20261113.md`. | Overlaps prior Run170 first-read/top-k selection and conflicts with the user's no-gating constraint. Efficiency method; not evidence for a Dice gain. Do not repeat. |

## What this means for implementation

- The original hypothesis remains specific: parts of an imperfect historical frame may help a later segmentation even when the whole frame is unreliable.
- Current attention/top-k selectors, mask-region ToMe, pointer-conditioned ToMe, MCTF, and regional utility prediction have already been tested or screened here; do not relaunch them under new names.
- No newly screened paper provides both a usable official implementation and a mechanism that fits the frozen frame schedule, memory writes, dense queries, decoder, and meaningful-Dice target.
- Continue source review for a distinct, code-available method. Before adapting one, state how it differs from failed runs, retain a matched generic control, and gate native/no-op parity, K/V and positional alignment, pointer/object isolation, unchanged weights/writes, and independent replay.

## Sources

- Puspitasari et al., [SAM2 spatiotemporal token pruning](https://pubmed.ncbi.nlm.nih.gov/42599858/), TPAMI 2026, DOI [10.1109/TPAMI.2026.3723878](https://doi.org/10.1109/TPAMI.2026.3723878).
- Mandal and Shashikumar, [RS³-Prune: Read-Sparse, Store-Sparse Token Pruning for Video Object Segmentation](https://arxiv.org/abs/2608.22526), arXiv:2608.22526.
- Wang et al., [DTMFormer: Dynamic Token Merging for Boosting Transformer-Based Medical Image Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/28394), AAAI 2024, DOI [10.1609/AAAI.V38I6.28394](https://doi.org/10.1609/AAAI.V38I6.28394); official code [iam-nacl/DTMFormer](https://github.com/iam-nacl/DTMFormer), MIT license.
- Wu et al., [Efficient medical image segmentation via self-similarity decoupling](https://www.sciencedirect.com/science/article/pii/S1047320326002233), 2026.
- Shi et al., [Redundancy-aware memory update for improved video object segmentation](https://www.sciencedirect.com/science/article/pii/S1077314226002249), *Computer Vision and Image Understanding* 270 (2026), article 104857, DOI [10.1016/j.cviu.2026.104857](https://doi.org/10.1016/j.cviu.2026.104857).
- Official Efficient-SAM2 repository: [jingjing0419/Efficient-SAM2](https://github.com/jingjing0419/Efficient-SAM2).

# SAM2 spatiotemporal token pruning — source review (2026-11-13)

## Finding

Puspitasari et al. propose training-free pruning inside SAM2 memory attention. Their scores target boundary delineation and temporal trackability, and the paper reports up to 87% token compression and 1.7× inference speedup with negligible performance disturbance on several benchmarks. This is relevant evidence that token-level memory use is technically plausible for video segmentation. It does **not** establish a Dice improvement; the stated contribution is efficiency while preserving segmentation quality.

## Upstream implementation status

Targeted searches by exact title, paper DOI, acronym, and author did not identify an official source repository or code link. No implementation was cloned or recreated from the paper. If the authors release or identify their repository, inspect and pin that source before considering integration.

## Fit with this project

- The method is a close architectural precedent because it evaluates spatial memory tokens inside SAM2 memory attention.
- Its objective is token pruning and compute reduction. Hard pruning/retention budgets do not by themselves test whether token use can improve PolypGen Dice over native SAM2.
- Its results are not a matched PolypGen comparison and cannot be used as evidence for our accuracy target.
- Do not reimplement the scoring functions from the paper or launch an efficacy run without the official code, a frozen protocol, a matched generic control, and a passing integration gate.
- An official implementation could be evaluated as an efficiency baseline if it can be integrated without changing SAM2's weights, memory writes, frame schedule, dense queries, or decoder. Any accuracy claim still requires full-video PolypGen propagation and paired sequence-level analysis.

## Sources

- Puspitasari et al., [Accelerating SAM2 With Efficient Memory Attention Module via Spatiotemporal Token Pruning](https://pubmed.ncbi.nlm.nih.gov/42599858/), *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2026, DOI [10.1109/TPAMI.2026.3723878](https://doi.org/10.1109/TPAMI.2026.3723878).
- SAM2 official implementation: [facebookresearch/sam2](https://github.com/facebookresearch/sam2), especially `sam2/modeling/memory_attention.py`.

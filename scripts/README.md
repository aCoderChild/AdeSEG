# Scripts

| File | Purpose |
|---|---|
| `audit_memory_token_regions.py` | Native memory grouping, reference/predicted overlays, coverage and width sensitivity |
| `audit_region_definitions.py` | CPU native-mask audit: seq2/seq3 replay gate, then all-sequence assignment/contour statistics with matching `--validation-dir`; author-code comparators and read-only primary attention |
| `region_definition_statistics.py` | Contour coverage, fixed size-bin assignment and descriptive aggregation; not a region selector |
| `run_token_attention.py` | Native read-only FP32 per-key/query-region capture; committed source gate and fresh-model replay |
| `token_attention_statistics.py` | Footprint precision and fixed sequence-pooled/log-median summaries, size and outcome strata |
| `queue_token_attention.py` | Gate then full attention; fails closed and stops before unregistered exclusions |
| `run_contiguous_exclusions.py` | Committed CPU FP32 one-target exclusions, six definitions, exact replay/bank/layout checks and rigid shape-matched controls |
| `contiguous_exclusion_statistics.py` | Sequence-level excess loss, fixed strata, bootstrap intervals and Holm18 inference |
| `queue_contiguous_exclusions.py` | Integrity/delivery gate then complete cohort, without effect-dependent stopping |
| `diagnose_contiguous_exclusions.py` | Isolated/history replay of failed seq12 target; separates replay equality from buffer equality; diagnostic evidence, not a full-cohort result |
| `relabel_region_trimaps.py` | Offline amended kernels 15/21/29 from completed saved-logit sequences; isolated author adapter, counts/coverage/GT/size statistics, no inference or attention |
| `measure_region_attention.py` | Verify all saved overlays; observe native attention mass and spatial enrichment using Efficient-SAM2 author attention |
| `run_region_exclusions.py` | Native one-frame regional key exclusions with three per-memory-frame count-matched random controls; no memory writeback |
| `run_author_baseline.py` | Isolated official SAM2/SAM2Long/SAMURAI/Lean predictors; exact-replay preflight and signature-gated full inference |
| `run_author_baseline_batch.py` | Complete seq2/seq3 validation of authors and fixed regional/zero/shuffled controls, then serial full runs and paired comparison; stops on failed checks/uploads |
| `calibrate_native_enrichment.py` | Read-only official CPU SAM2 attention calibration; existing author-attention observer, exact baseline predictions and complete replay checks |
| `measure_author_regions.py` | Read-only SAMURAI/SAM2Long regional attention; author candidate alignment, exact native prediction parity and replay-gated full inference |
| `queue_author_region_measurements.py` | Serial author measurement validation/full runs, existing-job dependency, per-layer/sequence comparison and confirmed W&B uploads |
| `run_enrichment_readout.py` | Native SDPA with frozen all-cohort proportional/inverse/random weights; strict source/recipe/runtime checks |
| `queue_enrichment_readout.py` | Validates calibration, estimates ratios on all eligible sequences, validates three readouts, then full evaluations and paired statistics |
| `run_efficient_smr.py` | Official SMR-only with dense encoder; device-allocation patch, observed key reduction, dense-encoder parity and signature-gated replay/full inference |
| `memory_token_io.py` | Shared native rollout capture, hashing, output checks and overlays |
| `build_polypgen_manifest.py` | Numerically ordered original PolypGen manifest |
| `build_polypgen_kaggle_manifest.py` | Kaggle mirror manifest with chronology safeguards |
| `verify_wandb_connection.py` | Credential/project preflight |

The entry points support `python scripts/<file>.py --help`;
`memory_token_io.py` is an internal helper, not a CLI.
Use `audit_memory_token_regions.py --all-sequences` for complete all-frame
assignment validation. No frame sampling or classifier training is used.
Unpromptable/pre-prompt frames receive reference-only labels, with no fake
memory vectors. See the methods for coverage, storage and exactness checks.
Only the native-memory workflows are active. Old study runners, detector
comparators, image-feature training and notebook orchestration were removed.

See [methods](../docs/REGION_TOKENS.md) and
[retained results](../docs/REGION_TOKENS_RESULTS.md). Outputs, caches and
credentials do not belong in this folder.

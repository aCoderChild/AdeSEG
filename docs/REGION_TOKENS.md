# Region-token experiment

Active branch: `experiment`. Frozen preregistration: `protocols/region_tokens_v1.json`.
This study tests region-level memory contributions; it does not reuse the old
latent-capability conclusions as evidence.

## Entry points

- `modeling/region_tokens.py`: category maps, matched sampling, attention aggregation and native persistent-propagation wrapper.
- `modeling/utility_memory.py`: existing additive attention hook, extended to token masks/biases.
- `scripts/run_region_token_pilot.py`: independently restored counterfactuals, E0 and E1.
- `scripts/analyze_region_token_pilot.py`: sequence-level pilot decision.
- `kaggle/region-tokens/region_tokens.ipynb`: pinned GitHub revision bootstrap.
- `kaggle/region-tokens/runner.py`: credentials, data checks, E0 → E1 orchestration.
- `scripts/collect_region_results.py`: terminal-status polling, direct Drive export and checksum receipt.

The notebook requires the **attached** Kaggle secret `WANDB_API_KEY`. It checks
Forge access before inference. It creates online orchestrator/E0/E1 runs in
`adenoid-hypertrophy/ade-seg`; raw results are uploaded as versioned artifacts.
Kaggle retains outputs under `/kaggle/working/region_tokens_v1`. The Mac exports
these directly to the existing Drive directory, without a second local results
directory. The local Drive client must sync them to the cloud. Kaggle working
storage is remote scratch, not the user's local disk. Drive delivery occurs
after Kaggle publishes the terminal version's output; W&B provides live progress.

```sh
kaggle kernels push -p kaggle/region-tokens --accelerator NvidiaTeslaT4
.venv/bin/python scripts/collect_region_results.py --wait --output-root \
  '/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026'
```

## Scientific status

Implemented: E0/E1 pilot, all four categories, two banks, five uniform and
distance-matched random controls, pointer/spatial bounds, per-layer/query-region
saliency, Dice/IoU/boundary F/HD95, low-resolution output and logit changes.
E1 samples ten fixed frame indices in each of ten dev sequences. Categories with
fewer than k available tokens are skipped and counted. The distance-matched
control can overlap the reference; overlap is reported.

**Not yet implemented/validated:** E2–E6 orchestration, GPU validation of persistent propagation, value-only
substitution and cross-sequence swap, CholecSeg8k adapter and source-video split,
RAFT condition analysis, complete external-baseline adapters, and method tuning.
An E1 pass permits developing/running those stages; it does not make this pilot
a full study. An E0 failure stops inference; an E1 failure stops/pivots the study.

PolypGen mirror equivalence to Synapse remains unverified. The runner records
per-file checksums, names and counts and explicitly retains that limitation.
The existing illumination helper uses luminance, not the HSV rule described in
the supplied design; the preregistration records the actual reused detector.

The MA-SAM2 submodule is pinned to its public repository, which currently ships
one predictor source file. It is not yet a validated runnable baseline. Existing
SAM2Long/Lean-SAM2 repositories remain the source for their adapters.

Historical v2 protocols and the exploratory Phase A report retain their original
provenance. Old development notebooks are not the active region-study workflow.

## Current deployment

Kaggle notebook: `bunnoob2005/adeseg-region-memory-tokens`. Version 1 failed
at Kaggle secret retrieval before creating a W&B run. The secret was verified
attached in the editor. A CPU-only interactive execution of the existing
`verify_wandb_connection.py --check-only` printed `connected` for the Forge
project. Version 2 was cancelled after 14 minutes queued; an interactive T4
session then started the bootstrap at revision `12919339aa4e`. No GPU E0 pass
is claimed yet.
Kaggle's available accelerator selector offered T4 x2 and TPU; P100 was not
available. No duplicate GPU job was launched to bypass that queue.

`scripts/collect_region_live.py --watch --group region-tokens-v1-12919339aa4e
--output-root <Drive output directory>` delivers stage artifacts while the
interactive session runs. Artifact downloads use `skip_cache=True`; SDK staging
paths also point into Drive. The resulting receipt distinguishes W&B artifact
delivery from a saved Kaggle notebook version export.

The collector recomputes E1 with the reviewed analysis code (matching feasible
frames before averaging), retains the original notebook decision, uploads
`decision_verified.json` to the same W&B orchestrator run, and records both
analysis and inference revisions. A passing pilot never sets
`full_study_complete` to true.

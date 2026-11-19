# Reproducible research lifecycle (v1)

Pipeline: **Data → Training → Inference → Evaluation → Ablation → Reproduction**.

The `adeseg.research` package adds a fail-closed **experiment contract and stage executor**, not a replacement for the model-specific runners. Existing baseline and training entrypoints remain intact. Each experiment JSON pins the dataset manifest and backbone checkpoint by SHA256. Stage commands are explicit argv arrays, not shell strings; executable stages must declare at least one artifact and output files are hashed into a receipt.

## Use

1. Produce an immutable JSONL dataset manifest with fields `patient_id`, `video_id`, `frame_index`, `split`, `image`, `mask`. Paths are relative to the JSONL file. Validate it:

   `python -m adeseg.research.data --manifest /path/to/frames.jsonl`

   This checks file existence, duplicate frames, and cross-split patient/video leakage. Patient IDs are trusted metadata; they must be independently audited against source provenance.

2. Copy `configs/research/protocol.example.json` and fill every actual path and SHA256. The example contains placeholders and is **not executable as-is**. Every stage has `command` and `artifacts`. Leave a stage command empty only when unimplemented; this is reported, never silently treated as success. In the command, `{python}`, `{workdir}`, `{protocol_dir}` and `{repo_root}` are substituted without a shell. Commands execute from the repository root.

3. Validate pinned inputs:

   `python -m scripts.research verify --protocol /path/to/protocol.json`

4. Preview an experiment:

   `python -m scripts.research run --protocol /path/to/protocol.json --stage all --dry-run`

5. Execute actual stage commands:

   `python -m scripts.research run --protocol /path/to/protocol.json --stage inference --workdir runs`

   The output root is `runs/<experiment_id>/` with distinct `receipt_<stage>.json` files (or `receipt_all.json`). A stage must return zero and create every declared artifact; otherwise the receipt is incomplete. To keep outputs immutable, existing declared outputs and existing receipts are never overwritten.

6. Paired ablation: export a CSV from each run containing unique `sequence` and a metric column, e.g. `dice`; compare them with

   `python -m scripts.ablate --baseline-protocol base.json --candidate-protocol adapter.json --baseline-csv base.csv --candidate-csv adapter.csv --output comparison.json`

   The comparison requires an explicit `evaluation` object in BOTH protocols: input resolution, probability threshold, postprocessing, frame policy, aggregation and the SHA256 of a frozen prompt-frame manifest. It also requires matching dataset and backbone hashes, split, initial prompt source and prompt budget, research track, and identical sequence sets. Run-specific adapter settings can differ. A missing evaluation object makes comparison fail closed. It produces paired per-sequence deltas and bootstrap intervals. It does NOT independently prove patient independence.

7. Reproduce: use the same sealed protocol and all-stage execution on a clean environment, compare artifact and receipt hashes against the release record, and document expected nondeterminism.

## Supported integration boundaries

- Existing training commands: `python -m scripts.train_med` and `python -m scripts.train_lean`, with mandatory `--split-file` guards.
- Existing baseline commands: `python -m scripts.compare_models`, `python -m scripts.evaluate_cmsa`, `python -m scripts.evaluate_lean`.
- Evaluation implementation: `evaluation/segmentation.py`, `evaluation/temporal.py`, and `evaluation/statistics.py`.
- The lifecycle wrapper executes **specified real commands**. It does not generate unimplemented raw-video training, model-specific inference or clinical annotation capabilities.

## Schema

```json
{
  "schema_version": 1,
  "experiment_id": "unique-run-id",
  "track": "medsam2_highres",
  "dataset": {"name": "PolypGen", "manifest": {"path": "frames.jsonl", "sha256": "<actual sha256>"}},
  "split": "test",
  "initialization": {"source": "oracle_mask", "prompt_budget": 1},
  "checkpoint": {"path": "backbone.pt", "sha256": "<actual sha256>"},
  "evaluation": {"resolution_hw": [512,512], "mask_threshold": 0.5, "postprocessing": "none", "frame_policy": "post_prompt_present", "aggregation": "sequence_macro", "prompt_frame_manifest_sha256": "<actual sha256>"},
  "stages": {
    "data": {"command": [], "artifacts": []},
    "training": {"command": [], "artifacts": []},
    "inference": {"command": [], "artifacts": []},
    "evaluation": {"command": [], "artifacts": []}
  }
}
```

The schema intentionally requires all four stages: missing implementations must be visible. Model-specific stage configuration should state which outputs represent detector-generated versus oracle prompts.

## Not yet implemented

- Universal image preprocessing and clinical adenoid dataset adapters.
- Raw-video end-to-end training for both contribution tracks.
- One universal model inference CLI (model-specific runners are retained).
- Full GPU benchmark CI and independent numerical forward parity for historical adapters.
- Automatic verification of all six lifecycle phases across the full benchmark.

Until these are addressed the repository has a reproducible *orchestration foundation*, not demonstrated end-to-end scientific reproduction.

## Evaluation conventions (2026-11 correction)

- Binary masks treat all nonzero values as foreground; both-empty Dice/IoU
  equal 1. A zero predicted foreground with nonempty GT has precision 0.
- Semantic mode rejects unknown class IDs; it never silently drops predictions.
- `evaluation.temporal_consistency.adjacent_mask_metrics` computes raw mask
  temporal IoU, absolute relative area change, and centroid shift normalized
  by the image diagonal. Missing predicted masks invalidate adjacent pairs;
  empty/empty centroid shift is undefined. These measures are NOT optical-flow
  compensation and can reflect legitimate camera motion.
- The temporal evaluator retains its historical frame-weighted segmentation
  aggregates and now also exports temporal values in frame/sequence CSVs.
  For paper claims, specify whether frame-weighted or equal-sequence estimates
  are used and whether GT-empty frames are included. Previously published
  results must be reported with their original evaluation version.

## MedSAM2 checkpoint inference

The reconstructed medical adapter has a strict new-run loader:
`modeling.medical_highres.load_reconstructed_medical_adapter`. It validates
the serialized checkpoint SHA256, exact backbone identity and config, current
adapter source hash, positive training step count and the saved residual
scale. The method **does not** establish historical-forward equivalence.
Full-video integration and GPU testing remain separate verification gates.

## Aggregate reporting

The evaluator retains top-level frame-weighted metrics for compatibility, and
adds `summary["sequence_macro"]` for equal-sequence means. These are distinct
estimands and must never be presented as interchangeable. The paired Wilcoxon
helper now rejects mismatched sequence sets rather than silently intersecting.

Paired ablations now independently authenticate declared dataset and backbone
file bytes via `verify_protocol` before reading result CSVs, and include
SHA256 checksums for the two result CSVs in the output report. Comparing two
matching declarative hash strings alone does not establish integrity.

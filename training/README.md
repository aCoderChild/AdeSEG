# Training

Two independent training entry points share one validated capture loader and
source-group integrity guard.

| Command | Model trained | Backbone | Loss |
|---|---|---|---|
| `python -m scripts.train_med` | Medical high-resolution skip adapter | Frozen MedSAM2 Hiera-T/512 | BCE + soft Dice + optional native-teacher KL / positive preservation |
| `python -m scripts.train_lean` | Lean feature aggregation adapter | Frozen Lean-SAM2 Hiera-B+ captured features | Boundary-weighted segmentation + optional positive preservation |

Implementation lives in `training/medical.py` and `training/lean.py`;
`training/safe_train.py` checks input provenance before/after training;
`training/medical_highres_training.py` contains the frozen decoder step.

Every public run requires `--split-file groups.json`. Example:

```json
{"train":["source_A","source_B"],"dev":["source_C"],"test":["source_D"]}
```

Every manifest row must contain `split: "train"`, `source_group`, a
`capture_sha256` and a `.pt` file reference. The medical manifest uses
`file`; the Lean manifest uses `tensor_file`. A saved checkpoint gets a
`<checkpoint>.split.json` receipt with the digest of the split, manifest,
capture bytes and trained weights.

**These are integrity declarations**, not proof of independent patient
identities; verify them against the actual dataset source. Neither trainer
accepts raw videos. The medical trainer uses *reconstructed* decoder-capture
training, not the missing historical clip trainer.

Run `python -m scripts.train_med --help` or
`python -m scripts.train_lean --help` for model-specific flags.

## Reconstructed MedSAM2 inference checkpoint contract

Use `modeling.medical_highres.load_reconstructed_medical_adapter(path,
expected_sha256=..., backbone_sha256=..., model_config=...)` to load newly
trained adapters. It rejects a mismatched binary file, backbone, model config,
source implementation, training-step count or residual scale. This returns a
verified adapter in eval mode for use with
`attach_medical_highres_adapter(predictor, adapter)`.

Both training files now also include optimizer and CPU RNG state in their
training checkpoint payloads. **Exact resume remains unimplemented**: an
optimizer state saved to a file does not by itself reproduce the interrupted
data order, epoch position, device RNG, libraries, or GPU kernels. Do not
claim resumable training until a tested resume command is added.

## Native MedSAM2 oracle-initialized video evaluation

A model-specific real predictor runner now exists:

```bash
python -m scripts.infer_med_video \
  --manifest /path/to/frames.jsonl --split test \
  --checkpoint /path/to/MedSAM2_latest.pt \
  --model-config sam2.1_hiera_t512.yaml \
  --output /path/to/native_run --device cuda

# Run adapter under the SAME input frames/first-GT-mask prompt:
python -m scripts.infer_med_video \
  --manifest /path/to/frames.jsonl --split test \
  --checkpoint /path/to/MedSAM2_latest.pt \
  --model-config sam2.1_hiera_t512.yaml \
  --adapter-checkpoint /path/to/adapter.pt \
  --adapter-sha256 <exact-file-sha256> \
  --output /path/to/adapter_run --device cuda
```

`predictions/<video_id>/<image_stem>.png` excludes the oracle prompt;
`run.json` records the prompt's source frame and image-hash provenance.
This runner is strictly **GT-mask oracle initialized** (first annotated
foreground), not automatic; do not conflate it with detector-driven results.
Actual GPU checkpoint execution remains to be verified in a supported runtime.

The matching evaluator for the oracle first-mask protocol is
`python -m scripts.evaluate_med_video --manifest /path/to/frames.jsonl
--run /path/to/native_run --output /path/to/native_evaluation`.
Do **not** use the generic all-frames temporal evaluator to score this
post-prompt-only prediction folder: it would otherwise count the excluded
oracle frame as missing. Repeat the same command for the adapter run,
then pair their `per_sequence.csv` reports for analysis.

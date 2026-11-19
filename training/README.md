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

## Sealed MedSAM2 oracle-video evaluation

Generate a protocol from a curated manifest and the SHA256-pinned checkpoint:

```bash
python -m scripts.prepare_med_video_protocol \
  --manifest /path/to/frames.jsonl \
  --checkpoint /path/to/MedSAM2_latest.pt \
  --output /path/to/experiment_protocol \
  --split dev --label-id 1
```

This produces `protocol.json` and `prompt_frames.jsonl` with authenticated
first-GT-positive oracle prompt positions and original image/mask hashes.
Every controlled runner must use the **same** sealed protocol:

```bash
python -m scripts.infer_med_video --protocol /path/to/experiment_protocol/protocol.json --device cuda --output /path/to/native
python -m scripts.evaluate_med_video --protocol /path/to/experiment_protocol/protocol.json --run /path/to/native --output /path/to/eval_native
python -m scripts.infer_med_video --protocol /path/to/experiment_protocol/protocol.json --device cuda --zero-initialized-adapter --output /path/to/zero
python -m scripts.compare_med_runs --protocol /path/to/experiment_protocol/protocol.json --native /path/to/native --candidate /path/to/zero --require-logit-identity --output /path/to/parity.json
```

For a trained adapter supply `--adapter-checkpoint` and its
`--adapter-sha256` instead of `--zero-initialized-adapter`.
This protocol uses **oracle GT-mask initialization** and scores post-prompt
frames, with GT-present Dice and GT-empty false alarms distinguished.
It is not automatic detection and does not reproduce historical Lean scores.

The single-protocol agreement is enforced across inference, evaluation,
comparison and CUDA preflight. The capture-based **training** process remains
a separate TRAIN-only protocol and is not yet an end-to-end video trainer.

# Public AdeSEG package

Use `adeseg.` for **new** experiments. The underlying source-bound components
and pretrained model vendors remain frozen at their historical paths until
their provenance hashes and exact forward parity can be verified.

## Research method A — MedSAM2 high-resolution refinement

1. `adeseg.backbones.medsam2` — verify/load **MedSAM2-latest Hiera-T/512**.
2. `adeseg.adapters.high_resolution_skip` — five-layer trainable skip adapter.
3. `adeseg.inference.medical_decoder_integration` — install the decoder hook.
4. `adeseg.losses.teacher_kl` — detached teacher KL **only in training**.
5. `adeseg.losses.segmentation` — GT-supervised optimization **only in training**.

The medical adapter is a reconstruction from checkpoint tensor geometry;
its lost original forward equation remains unavailable. Historical study
scores do not automatically transfer to this implementation.

## Research method B — Lean-SAM2 channel-value aggregation

1. `adeseg.backbones.lean_sam2` — pinned author **SAM2.1 Hiera-B+** backbone.
2. `adeseg.adapters.multiscale_channel_attention` — six-stage Q/K/V refiner.
3. `adeseg.inference.lean_memory_routing` — output/memory ablation controls.
4. `adeseg.memory.temporal_anchors` and `spatial_position_encoding` —
   separate, explicitly enabled memory/position experiments.

The verified 21-sequence 0.742071 Dice run used an indexed-RoPE Lean
modification **without** either of the above learned adapters.

## Ground-truth and model boundaries

- Inference modules must never accept target masks as hidden features.
- Dataset splits and source identities must be checked before training.
- Pretrained MedSAM2 Hiera-T and Lean Hiera-B+ checkpoint hashes differ.
- Historical classes retain original registration and source-hash checking.
- A source import or CPU smoke test is not a 21-video reproduction.

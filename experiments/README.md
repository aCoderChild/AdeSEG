# Research experiments

One experiment folder per scientific question. This is not a single
leaderboard: checkpoint family, initialization, frame selection and data
source all affect a comparison.

| Experiment | Question | Status |
|---|---|---|
| [MedSAM2 high resolution](medsam2_high_resolution/README.md) | Can high-res feature refinement plus teacher KL improve masks? | Reconstructed code; four-sequence historical results are mixed |
| [Lean feature aggregation](lean_feature_aggregation/README.md) | Does learned early/deep Q/K/V channel mixing improve propagation? | Not validated on 21 sequences |
| [Reference baselines](reference_baselines/README.md) | What do the original or corrected backbone methods achieve? | Includes corrected-Lean **0.742071** Dice |
| [Exploratory ablations](exploratory_ablations/README.md) | Which memory, RoPE, region and loss controls matter? | Not independent validation |

Always report initialization policy, patient/sequence IDs, mask availability,
per-sequence target counts, false alarms on empty frames, checkpoint SHA-256,
source revision and retrospectively chosen preprocessing rules. File labels
like `split=train` are not proof that an external patient holdout was sealed.

Historical protocols and scripts remain source-bound under their original
names; the entire old tree is preserved in
[the full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

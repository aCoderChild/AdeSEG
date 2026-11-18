# Experiments and retained evidence

These links are to **saved original artifacts**, not newly rerun scores. They use different initialization and target-frame protocols; do not rank them without matching prompts and frames.

| Experiment | Saved evidence | Interpretation |
|---|---|---|
| Lean-SAM2 baseline (0.73446 GT-prompt Dice) | [audited attempt03](https://drive.google.com/drive/folders/1DmkKRrGmEcJQUp0Pb0fp5u5CS0DSo0B5) | 21 eligible sequences; originally inspected; 177/323 GT-empty false alarms |
| Automatic pooled-multiscale Lean (0.71721 all-frame Dice) | [cohort results](https://drive.google.com/drive/folders/15IxxQ8WnWv_9rLcQlCwYxAd7x_R6Zamt) | First YOLO detection; different from the GT-prompt baseline |
| Fixed-prompt memory-anchor ablation | [independent eight-sequence audit](https://drive.google.com/drive/folders/19zIHmT8x9pRfsN6WojWIjrrDwldhHhah) | Difference in Dice ~0.000006; does not support a useful gain |
| Indexed spatial RoPE (0.7353 matched Dice, 0.7318 all-frame) | [cohort indexed-RoPE report](https://drive.google.com/file/d/13gm6M61y2zKyJSBS5Vxxok5AY1lePhpt/view) | Same automatic prompts, not the GT-mask Lean baseline |
| Teacher-preserving training | [study protocol](../protocols/SOURCE_GROUP_PRESERVATION_V1.md) | Training-only loss; no final 21-video efficacy claim |
| Learned multiscale channel-value aggregation | [research protocol](../protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md) | Separate project adapter; do not attribute the Lean baseline Dice to it |

For historical runs and any modified-Lean result not yet tied to a verified score record (including the requested ~0.742 figure), **the complete experiment branch was backed up unchanged** in [backup/experiment-full-20261118](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

The source branch contains additional experiments (Kalman, utility probes, high-resolution adapters) intentionally omitted from the focused active tree. Their historical commits remain accessible in that backup, not silently deleted from Git history.

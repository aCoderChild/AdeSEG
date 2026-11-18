# Research method: learned multi-stage channel-value aggregation

**New, descriptive implementation:** `modeling/feature_aggregation.py`, class `MultistageFeatureAggregator`.

The model projects early SAM2 FPN features and deep memory-conditioned features, computes learned Q/K channel correlations from predicted foreground/background probabilities, multiplies the channel attention by V to aggregate values, then predicts residual mask-logit corrections.

The original `modeling/multiscale_channel_value_refiner_v1.py` is retained as a source-bound historical implementation. Video routing remains `modeling/lean_channel_value_routing_v1.py` with verifier-owned runtime leases. The active convenience API is `modeling/refinement_routing.py`.

Train offline on frozen feature captures using `python -m training.train_feature_aggregator --help`; record new source/feature hashes. Original frozen protocol: `protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md`.

**Scientific limitation:** prior small pilots were mixed. No verified 21-video mean Dice of 0.742 is attributable to this learned aggregator.

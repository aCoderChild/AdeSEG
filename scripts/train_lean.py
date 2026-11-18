"""Train the learned Lean-SAM2 multiscale channel-value refiner on TRAIN-only captures.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from training.train_feature_aggregator import main

if __name__ == "__main__":
    main()

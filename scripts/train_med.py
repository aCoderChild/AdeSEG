"""Train a reconstructed MedSAM2 high-resolution skip adapter on TRAIN-only captures.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from training.train_medical_highres_adapter import main

if __name__ == "__main__":
    main()

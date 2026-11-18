"""Train MedSAM2 high-resolution refinement with sealed capture-group checks."""
from training.medical import main as trainer
from training.safe_train import run_training


def main():
    run_training(trainer, mode="med")


if __name__ == "__main__":
    main()

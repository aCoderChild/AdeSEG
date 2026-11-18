"""Train Lean-SAM2 learned feature aggregation with sealed capture-group checks."""
from training.lean import main as trainer
from training.safe_train import run_training


def main():
    run_training(trainer, mode="lean")


if __name__ == "__main__":
    main()

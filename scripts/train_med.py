"""Train MedSAM2 with a mandatory, frozen source-group partition.

Use --split-file train_groups.json alongside original training arguments.
"""
from training.safe_train import checked_arguments, save_receipt
from training.train_medical_highres_adapter import main as training_main


def main():
    with checked_arguments(mode="med") as proof:
        training_main()
        if proof is not None:
            save_receipt(proof)


if __name__ == "__main__":
    main()

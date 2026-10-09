"""Select the fixed-gain control on validation JSONs produced by evaluate_polypgen.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Contains gain_*/kalman_fixed_gain.json runs.")
    parser.add_argument("--fp-weight", type=float, default=0.5)
    args = parser.parse_args()
    candidates = []
    for path in sorted(args.results_root.glob("gain_*/kalman_fixed_gain.json")):
        result = json.loads(path.read_text())
        run = json.loads((path.parent / "run.json").read_text())
        metrics = result["summary"]["per_sequence"]
        candidates.append({
            "gain": run["fixed_gain"], "directory": str(path.parent),
            "dice": metrics["polyp_dice"], "empty_fp": metrics["empty_fp"], "lost": metrics["lost"],
            "selection_j": metrics["polyp_dice"] - args.fp_weight * metrics["empty_fp"],
        })
    if not candidates:
        raise ValueError("No gain_*/kalman_fixed_gain.json files found.")
    selected = max(candidates, key=lambda row: row["selection_j"])
    output = {
        "selection_split": json.loads(Path(selected["directory"], "run.json").read_text())["split"],
        "selection_rule": f"max per-sequence polyp Dice - {args.fp_weight} x empty-frame FP",
        "candidates": candidates,
        "selected": selected,
        "warning": "Use the selected gain unchanged on a separately locked test set.",
    }
    (args.results_root / "selection.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()

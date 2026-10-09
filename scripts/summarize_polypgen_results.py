"""Create summary and paired tests from one evaluate_polypgen.py result directory."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def metrics(result):
    per_video = result["summary"]["per_video"]
    return {
        video: {
            "dice": row["polyp_dice"],
            "fp": row["empty_fp"],
            "lost": row["lost"],
            "j": row["polyp_dice"] - 0.5 * row["empty_fp"],
        }
        for video, row in per_video.items()
    }


def compare(first, second):
    common = sorted(set(first) & set(second))
    answer = {}
    for name in ("dice", "fp", "j"):
        delta = np.array([first[video][name] - second[video][name] for video in common])
        answer[name] = {
            "mean_difference": float(delta.mean()),
            "higher": int((delta > 0).sum()), "lower": int((delta < 0).sum()),
            "p_wilcoxon": float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0,
        }
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    raw = {
        path.stem: json.loads(path.read_text())
        for path in args.results_dir.glob("*.json") if path.name not in {"run.json", "summary.json"}
    }
    by_method = {name: metrics(result) for name, result in raw.items()}
    summary = {
        "protocol": json.loads((args.results_dir / "run.json").read_text()),
        "methods": {
            name: {
                "per_sequence": result["summary"]["per_sequence"],
                "j": result["summary"]["per_sequence"]["polyp_dice"] - 0.5 * result["summary"]["per_sequence"]["empty_fp"],
            }
            for name, result in raw.items()
        },
        "paired": {},
    }
    for baseline in ("native", "native_redetect", "rde_redetect", "detector_only", "kalman_untrained",
                     "presence_only", "kalman_fixed_gain", "rde_trained", "rde_trained_redetect"):
        if baseline in by_method and "kalman_full" in by_method:
            summary["paired"][f"kalman_full_minus_{baseline}"] = compare(by_method["kalman_full"], by_method[baseline])
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, values in summary["methods"].items():
        row = values["per_sequence"]
        print(f"{name}: Dice={row['polyp_dice']:.3f}, FP={row['empty_fp']:.3f}, "
              f"lost={row['lost']:.3f}, J={values['j']:.3f}")


if __name__ == "__main__":
    main()

"""Read-only recheck of a saved native-region gate; no model rerun or changes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from modeling.region_definitions import region_label_schemes  # noqa: E402
from modeling.region_tokens import sam2_low_res_mask  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import tensor_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.validation_dir
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    setup = json.loads((source / "setup.json").read_text())
    if not all(
        digest(ROOT / name) == value
        for name, value in setup["definition_sources"].items()
    ):
        raise ValueError(
            "Gate source changed; do not recheck with a different definition"
        )
    if digest(args.manifest) != setup["manifest_sha256"]:
        raise ValueError("Gate manifest changed")
    videos = {
        **load_manifest(args.manifest, "dev"),
        **load_manifest(args.manifest, "test"),
    }
    rows, prompts = [], []
    for sequence in ("seq2", "seq3"):
        folder = source / sequence
        records = [
            json.loads(line)
            for line in (folder / "assignment_frames.jsonl").read_text().splitlines()
        ]
        for record in records:
            index = record["frame"]
            saved = torch.load(
                folder / f"assignments_{index:06d}.pt", weights_only=False
            )
            low = saved["native_low_res_logits"]
            if tensor_hash(low) != record["native_mask_sha256"]:
                raise ValueError("Saved native logits do not match capture hash")
            if record["conditioning"]:
                truth = load_label_mask(videos[sequence][index].mask_path, [1]) == 1
                exact = torch.equal(low, sam2_low_res_mask(truth))
                if not exact:
                    raise ValueError("Prompt low-res bitwise check failed")
                prompts.append({"sequence": sequence, "frame": index, "exact": exact})
                continue
            with Image.open(folder / f"native_mask_{index:06d}.png") as image:
                round_trip = region_label_schemes(
                    sam2_low_res_mask(np.asarray(image) > 0)
                )
            for name, native in saved["schemes"].items():
                changed = int((native["labels"] != round_trip[name]["labels"]).sum())
                rows.append(
                    {
                        "sequence": sequence,
                        "frame": index,
                        "definition": name,
                        "changed_tokens": changed,
                        "tokens": 4096,
                        "disagreement_fraction": changed / 4096,
                    }
                )
    summaries = {}
    for name in sorted({r["definition"] for r in rows}):
        summaries[name] = {}
        for sequence in ("seq2", "seq3"):
            selected = [
                r for r in rows if r["definition"] == name and r["sequence"] == sequence
            ]
            summaries[name][sequence] = {
                "frames": len(selected),
                "frames_with_changed_tokens": sum(
                    r["changed_tokens"] > 0 for r in selected
                ),
                "mean_disagreement_fraction": float(
                    np.mean([r["disagreement_fraction"] for r in selected])
                ),
                "max_disagreement_fraction": max(
                    r["disagreement_fraction"] for r in selected
                ),
            }
    result = {
        "scope": "seq2/seq3 gate only; post-launch verification, not preregistration",
        "prompt_checks": prompts,
        "native_input_hashes_match": True,
        "round_trip": summaries,
        "frames": rows,
        "decision": "Native masks are authoritative. PNG labels are not assumed identical; no PNG-derived labels enter the running audit.",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()

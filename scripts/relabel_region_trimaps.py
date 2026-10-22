"""Offline amended trimaps from saved native logits; no model or attention run."""

import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from modeling import region_definitions  # noqa: E402
from modeling.region_tokens import sam2_low_res_mask  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.region_definition_statistics import contour_coverage, summarize_frames  # noqa: E402

AMENDMENT = ROOT / "protocols/region_definition_trimap_amendment_2026_10_23.json"


@lru_cache(maxsize=1)
def amended_adapter():
    # Isolated configuration: never mutate the adapter used by the active audit.
    spec = importlib.util.spec_from_file_location(
        "amended_region_definitions", region_definitions.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRIMAP_WIDTHS = tuple(
        json.loads(AMENDMENT.read_text())["amended_widths_at_author640"]
    )
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    setup = json.loads((args.source_dir / "setup.json").read_text())
    if digest(args.manifest) != setup["manifest_sha256"]:
        raise ValueError("Source manifest mismatch")
    if any(
        digest(ROOT / name) != value
        for name, value in setup["definition_sources"].items()
    ):
        raise ValueError(
            "Executed source changed; cannot silently relabel with another adapter"
        )
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    videos = {
        **load_manifest(args.manifest, "dev"),
        **load_manifest(args.manifest, "test"),
    }
    records = []
    for folder in sorted(args.source_dir.glob("seq*")):
        if not folder.is_dir() or not (folder / "summary.json").exists():
            continue  # Never read a partially written sequence.
        for line in (folder / "assignment_frames.jsonl").read_text().splitlines():
            old = json.loads(line)
            saved = torch.load(
                folder / f"assignments_{old['frame']:06d}.pt", weights_only=False
            )
            low = saved["native_low_res_logits"]
            schemes = amended_adapter().region_label_schemes(low)
            truth = (
                load_label_mask(videos[folder.name][old["frame"]].mask_path, [1]) == 1
            )
            reference_low = sam2_low_res_mask(truth)
            reference = amended_adapter().region_label_schemes(reference_low)
            row = {
                k: old[k]
                for k in ("frame", "conditioning", "size_bin", "gt_area_fraction")
            }
            row.update(sequence=folder.name, schemes={})
            labels_to_save = {}
            for width in (15, 21, 29):
                name = f"matteformer_k{width}"
                labels, expected = schemes[name]["labels"], reference[name]["labels"]
                row["schemes"][name] = {
                    "counts": np.bincount(labels.ravel(), minlength=3).tolist(),
                    "confusion": np.bincount(
                        (3 * expected + labels).ravel(), minlength=9
                    )
                    .reshape(3, 3)
                    .tolist(),
                    "self_contour": contour_coverage((low > 0).numpy(), labels),
                    "gt_contour": contour_coverage((reference_low > 0).numpy(), labels),
                }
                labels_to_save[name] = labels
            destination = output / folder.name
            destination.mkdir(exist_ok=True)
            torch.save(labels_to_save, destination / f"labels_{old['frame']:06d}.pt")
            records.append(row)
    if not records:
        raise ValueError("No completed native sequences to relabel")
    (output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    targets = [r for r in records if not r["conditioning"]]
    summary = {
        "source": str(args.source_dir),
        "inference_run": False,
        "attention_available": False,
        "frames_including_prompts": len(records),
        "postprompt_frames": len(targets),
        "statistics_postprompt": summarize_frames(targets),
        "by_size_postprompt": {
            group: summarize_frames([r for r in targets if r["size_bin"] == group])
            for group in ("absent", "Q1", "Q2", "Q3", "Q4")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "amendment.json").write_bytes(AMENDMENT.read_bytes())
    (output / "relabel_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("statistics_postprompt", "by_size_postprompt")
            }
        )
    )


if __name__ == "__main__":
    main()

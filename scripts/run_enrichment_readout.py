#!/usr/bin/env python3
"""Reuse the verified author runner with frozen all-cohort enrichment weights.

Process-local adapters leave the running author-baseline source untouched.
No replacement attention, encoder, decoder or memory-selection algorithm.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling import region_bias  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402


def weighted_bias(labels, pointers, mode, seed, weights):
    labels = torch.as_tensor(labels).long().flatten().cpu()
    values = torch.as_tensor(weights, dtype=torch.float32)
    if values.shape != (3,) or not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Exactly three finite positive weights required")
    if pointers < 0 or not ((labels >= 0) & (labels <= 2)).all():
        raise ValueError("Invalid labels or pointers")
    if mode not in ("zero", "boundary", "shuffled"):
        raise ValueError("Unknown integration mode")
    if mode == "shuffled":
        labels = labels[
            torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        ]
    spatial = torch.zeros(len(labels)) if mode == "zero" else values.log()[labels]
    return torch.cat([spatial, torch.zeros(pointers)])


def read_recipe(path, strategy, manifest):
    recipe = json.loads(path.read_text())
    if recipe["manifest_sha256"] != author.digest(manifest):
        raise ValueError("Recipe manifest differs")
    if recipe["regions"] != ["background", "foreground_interior", "boundary"]:
        raise ValueError("Unexpected region order")
    from scripts.queue_enrichment_readout import estimate_recipe

    if (
        recipe.get("schema_version") != 2
        or recipe.get("calibration_scope") != "all_eligible_sequences"
    ):
        raise ValueError("Expected all-cohort native CPU calibration")
    computed = estimate_recipe(Path(recipe["source_directory"]), manifest)
    if json.loads(json.dumps(computed)) != recipe:
        raise ValueError("Recipe weights, source hashes or cohort provenance changed")
    weights = recipe["mean_enrichment"]
    if len(weights) != 3 or any(not math.isfinite(v) or v <= 0 for v in weights):
        raise ValueError("Undefined/zero enrichment: do not invent reciprocal weights")
    return recipe, [1 / v for v in weights] if strategy == "inverse" else weights


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("proportional", "inverse", "random"), required=True
    )
    options, remaining = parser.parse_known_args()
    manifest = Path(remaining[remaining.index("--manifest") + 1])
    recipe, weights = read_recipe(options.recipe, options.strategy, manifest)
    expected = "sam2_shuffled" if options.strategy == "random" else "sam2_region"
    if remaining[remaining.index("--method") + 1] != expected:
        raise ValueError("Method/strategy mismatch")
    original_controller = region_bias.NativeRegionBias

    class WeightedController(original_controller):
        def capture(self, *args):
            record = super().capture(*args)
            record["weights_background_foreground_boundary"] = weights
            record["strategy"] = options.strategy
            return record

    region_bias.NativeRegionBias = WeightedController
    region_bias.region_log_bias = lambda labels, pointers, mode, seed: weighted_bias(
        labels, pointers, mode, seed, weights
    )
    original_signature = author.signature

    def signature(*args):
        identity = original_signature(*args)
        if any(
            identity[key] != value for key, value in recipe["runtime_identity"].items()
        ):
            raise ValueError("Calibration/inference runtime mismatch")
        return {
            **identity,
            "enrichment_recipe_sha256": author.digest(options.recipe),
            "enrichment_adapter_sha256": author.digest(Path(__file__)),
            "enrichment_strategy": options.strategy,
            "weights_background_foreground_boundary": weights,
        }

    author.signature = signature
    original_start = author.start_wandb

    def start(project, entity, name, mode, url, config, output, tags):
        config["region_modulation"].update(
            weights_background_foreground_boundary=weights,
            shuffled=options.strategy == "random",
            strategy=options.strategy,
            recipe=recipe,
        )
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(__file__, output / "enrichment_adapter_source.py")
        shutil.copy2(options.recipe, output / "enrichment_recipe.json")
        return original_start(
            project,
            entity,
            f"09 | enrichment {options.strategy} | {config['scope']}",
            mode,
            url,
            config,
            output,
            tags + ["enrichment-readout", options.strategy],
        )

    author.start_wandb = start
    original_finalize = author.finalize_tracking

    def finalize(run, output, summary, names):
        return original_finalize(
            run,
            output,
            summary,
            names
            + [
                "enrichment_adapter_source.py",
                "enrichment_recipe.json",
            ],
        )

    author.finalize_tracking = finalize
    sys.argv = [str(Path(author.__file__)), *remaining]
    author.main()


if __name__ == "__main__":
    main()

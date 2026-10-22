#!/usr/bin/env python3
"""Read-only regional attention for pinned author predictors.

Reuse the native observer, Boundary IoU and baseline runner. SAM2Long records
all executed branch reads, not an invented attention map for its final path.
"""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.region_bias import feature_key  # noqa: E402
from modeling.region_tokens import reference_token_regions  # noqa: E402
from scripts.calibrate_native_enrichment import NativeEnrichmentObserver  # noqa: E402
from scripts.measure_region_attention import sequence_report  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402


def selected_memory(output, frame, selection):
    """Read the author's candidate axis; never choose a hypothesis ourselves."""
    features, masks = output["maskmem_features"], output["pred_masks"]
    if features.ndim == 5:
        if not isinstance(selection, dict) or frame not in selection:
            raise RuntimeError("Missing author SAM2Long branch identity")
        branch = selection[frame]
        features, masks = features[..., branch], masks[..., branch]
    else:
        branch = None
    if features.shape != (1, 64, 64, 64) or masks.ndim != 4:
        raise RuntimeError("Unsupported author memory/mask layout")
    return features, masks, branch


class AuthorRegionObserver(NativeEnrichmentObserver):
    """Observe each native memory-read call, including SAM2Long pathways."""

    def __init__(self, model, sequence, prompt, state):
        super().__init__(model, "zero", sequence, prompt)
        self.state, self.labels_cache = state, {}
        self.calls = 0
        self.prepare = model._prepare_memory_conditioned_features
        self.arguments = inspect.signature(self.prepare)

    def prepare_read(self, *args, **kwargs):
        bound = self.arguments.bind(*args, **kwargs)
        bound.apply_defaults()
        context = bound.arguments
        frame = context["frame_idx"]
        if context["is_init_cond_frame"]:
            return self.prepare(*args, **kwargs)
        self.frame, self.cached, self.bank = frame, None, {}
        selection = context.get("mem_pick_index", 0)
        branches = {}
        for outputs in context["output_dict"].values():
            for stored, output in outputs.items():
                if stored >= frame or output.get("maskmem_features") is None:
                    continue
                features, masks, branch = selected_memory(output, stored, selection)
                cache_id = (stored, branch)
                if cache_id not in self.labels_cache:
                    _, original = self.model._get_orig_video_res_output(
                        self.state, masks
                    )
                    labels = reference_token_regions(
                        original[0, 0].detach().cpu().numpy() > 0, (64, 64), 0.02
                    )["labels"]
                    self.labels_cache[cache_id] = (
                        feature_key(features.flatten(2).transpose(1, 2)),
                        torch.from_numpy(np.asarray(labels)),
                    )
                key, labels = self.labels_cache[cache_id]
                if key in self.bank and self.bank[key][0] != stored:
                    raise RuntimeError("Ambiguous raw memory-frame identity")
                self.bank[key] = (stored, labels)
                branches[stored] = branch
        result = self.prepare(*args, **kwargs)
        if [r["layer"] for r in self.observer.rows] != list(range(4)):
            raise RuntimeError("Incomplete four-layer author memory read")
        for row in self.observer.rows:
            for unit in row["memory_units"]:
                unit["candidate"] = branches[unit["frame"]]
            self.measurements.append(
                {
                    **row,
                    "frame": frame,
                    "sequence": self.sequence,
                    "read_call": self.calls,
                }
            )
        self.calls += 1
        self.observer.rows.clear()
        self.trace.clear()
        return result

    def __enter__(self):
        super().__enter__()
        self.prepare_patch = patch.object(
            self.model, "_prepare_memory_conditioned_features", self.prepare_read
        )
        self.prepare_patch.start()
        return self

    def __exit__(self, *exc):
        self.prepare_patch.stop()
        super().__exit__(*exc)


def verify_predictions(rows, reference):
    expected = [json.loads(line) for line in reference.read_text().splitlines()]
    keys = ("source_frame_index", "prediction_sha256", "present")
    if [tuple(r[k] for k in keys) for r in rows] != [
        tuple(r[k] for k in keys) for r in expected
    ]:
        raise RuntimeError("Observer changed author predictions or source coverage")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    method = remaining[remaining.index("--method") + 1]
    if method not in ("sam2", "samurai", "sam2long"):
        raise ValueError("Only reviewed native author predictors are supported")
    reference = json.loads((options.reference_baseline / "setup.json").read_text())
    delivery = json.loads((options.reference_baseline / "delivery.json").read_text())
    summary = json.loads((options.reference_baseline / "summary.json").read_text())
    if not summary["completed"] or not delivery["wandb_artifact_upload_complete"]:
        raise ValueError("Native author reference is incomplete")
    original_signature = author.signature

    def signature(*args):
        identity = original_signature(*args)
        if identity != {key: reference[key] for key in identity}:
            raise ValueError("Native author reference/source/runtime mismatch")
        return {
            **identity,
            "regional_measurement_sha256": {
                name: author.digest(ROOT / name)
                for name in (
                    "scripts/measure_author_regions.py",
                    "scripts/calibrate_native_enrichment.py",
                    "modeling/region_attention.py",
                    "scripts/measure_region_attention.py",
                )
            },
            "reference_setup_sha256": author.digest(
                options.reference_baseline / "setup.json"
            ),
            "attention_scope": "all_executed_branch_reads"
            if method == "sam2long"
            else "online_reads",
        }

    author.signature = signature
    original_rollout, original_iterator = author.rollout, author.iter_predictions

    def rollout(
        model,
        method,
        frames,
        prompt,
        folder,
        device,
        run,
        repeat=False,
        capture_logits=False,
    ):
        holder = []

        def iterator(predictor, state, name, first):
            observer = AuthorRegionObserver(model, frames[0].video_id, prompt, state)
            holder.append(observer)
            with observer:
                yield from original_iterator(predictor, state, name, first)

        author.iter_predictions = iterator
        try:
            rows, seconds = original_rollout(
                model,
                method,
                frames,
                prompt,
                folder,
                device,
                run,
                repeat,
                capture_logits,
            )
        finally:
            author.iter_predictions = original_iterator
        verify_predictions(
            rows, options.reference_baseline / frames[0].video_id / "frames.jsonl"
        )
        measurements = holder[0].measurements
        targets = {r["frame"] for r in rows}
        if {r["frame"] for r in measurements} != targets:
            raise RuntimeError("Missing or unexpected attention targets")
        expected = len(rows) if method != "sam2long" else 1 + 3 * (len(rows) - 1)
        if len(measurements) != 4 * expected:
            raise RuntimeError("Incomplete author branch/layer coverage")
        if not repeat:
            (folder / "attention.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in measurements)
            )
            (folder / "attention_summary.json").write_text(
                json.dumps(
                    {
                        "sequence": frames[0].video_id,
                        "native_predictions_bit_identical": True,
                        "label_source": "author stored candidate predicted mask, original resolution; Boundary IoU 0.02",
                        "attention_scope": "all executed branch reads; NOT final-path attribution"
                        if method == "sam2long"
                        else "online reads",
                        "layers": sequence_report(measurements),
                    },
                    indent=2,
                )
                + "\n"
            )
        return rows, seconds

    author.rollout = rollout
    original_start, original_finalize = author.start_wandb, author.finalize_tracking

    def start(project, entity, name, mode, url, config, output, tags):
        config["region_modulation"]["enabled"] = False
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(__file__, output / "measurement_source.py")
        return original_start(
            project,
            entity,
            f"12 | {method} regional attention | {config['scope']}",
            mode,
            url,
            config,
            output,
            tags + ["read-only-regional-attention"],
        )

    author.start_wandb = start
    author.finalize_tracking = lambda run, output, summary, names: original_finalize(
        run, output, summary, names + ["measurement_source.py"]
    )
    sys.argv = [__file__, *remaining]
    author.main()


if __name__ == "__main__":
    main()

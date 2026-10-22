#!/usr/bin/env python3
"""Measure enrichment on the unchanged official SAM2 CPU rollout.

Reuse native key alignment, the verified author-attention observer and sequence
aggregation. Return native SDPA outputs unchanged; never write measurements.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling import region_bias  # noqa: E402
from modeling.region_attention import RegionAttentionObserver  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.measure_region_attention import sequence_report  # noqa: E402


class NativeEnrichmentObserver(region_bias.NativeRegionBias):
    def __init__(self, model, mode, sequence, prompt):
        super().__init__(model, "zero", sequence, prompt)
        self.observer = RegionAttentionObserver(model.memory_attention, None)
        self.measurements = []

    def _before(self, layer, module, args, kwargs):
        super()._before(layer, module, args, kwargs)
        frames = {frame: labels for frame, labels in self.bank.values()}
        self.observer.labels = torch.cat(
            [frames[u["frame"]].flatten() for u in self.units]
            + [torch.full((self.pointers,), -1)]
        )
        self.observer.units = self.units

    def _sdpa(self, q, k, v, *args, **kwargs):
        if not self.active:
            return self.native_sdpa(q, k, v, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None:
            raise RuntimeError("Calibration requires unmodified native attention")

        def native(q, k, v, attn_mask=None):
            return self.native_sdpa(q, k, v, **{**kwargs, "attn_mask": attn_mask})

        return self.observer._observe(native, self.trace[-1]["layer"], q, k, v)

    def capture(self, state, index, logits):
        trace = super().capture(state, index, logits)
        for row in self.observer.rows:
            self.measurements.append({**row, "frame": index, "sequence": self.sequence})
        self.observer.rows.clear()
        return trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    if remaining[remaining.index("--method") + 1] != "sam2_zero":
        raise ValueError("Calibration must use zero-bias native SAM2")
    reference = json.loads((options.reference_baseline / "setup.json").read_text())
    original_signature = author.signature

    def signature(*args):
        identity = original_signature(*args)
        for key in (
            "author_revision",
            "checkpoint_sha256",
            "manifest_sha256",
            "device",
            "torch",
            "precision",
            "config",
            "integration_sha256",
        ):
            if identity[key] != reference[key]:
                raise ValueError(f"Calibration/native reference mismatch: {key}")
        return {
            **identity,
            "calibration_source_sha256": {
                name: author.digest(ROOT / name)
                for name in (
                    "scripts/calibrate_native_enrichment.py",
                    "modeling/region_attention.py",
                    "scripts/measure_region_attention.py",
                )
            },
            "calibration_label_source": "predicted_0.02",
            "reference_baseline": str(options.reference_baseline),
        }

    author.signature = signature
    # Existing native/zero parity checks also cover all calibration smoke frames.
    # The reference has an extra calibration identity, so use independent per-frame
    # checks below rather than pretend its old validation receipt is identical.
    author.REPOSITORIES["sam2_calibration"] = author.REPOSITORIES["sam2"]
    remaining[remaining.index("--method") + 1] = "sam2_calibration"
    original_rollout = author.rollout
    original_iterator = author.iter_predictions

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
        controller = NativeEnrichmentObserver(model, "zero", frames[0].video_id, prompt)

        def iterator(predictor, state, name, first):
            with controller:
                for index, ids, logits in original_iterator(
                    predictor, state, name, first
                ):
                    controller.capture(state, index, logits)
                    yield index, ids, logits

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
        baseline = [
            json.loads(line)
            for line in (
                options.reference_baseline / frames[0].video_id / "frames.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        if [(r["source_frame_index"], r["prediction_sha256"]) for r in rows] != [
            (r["source_frame_index"], r["prediction_sha256"]) for r in baseline
        ]:
            raise RuntimeError(
                "Observer changed native baseline predictions or coverage"
            )
        if len(controller.measurements) != len(rows) * 4:
            raise RuntimeError("Incomplete native attention measurement")
        if not repeat:
            (folder / "attention.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in controller.measurements)
            )
            (folder / "attention_summary.json").write_text(
                json.dumps(
                    {
                        "sequence": frames[0].video_id,
                        "native_predictions_bit_identical": True,
                        "label_source": "predicted_0.02",
                        "layers": sequence_report(controller.measurements),
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
        shutil.copy2(__file__, output / "calibration_source.py")
        return original_start(
            project,
            entity,
            f"11 | All-sequence native enrichment | {config['scope']}",
            mode,
            url,
            config,
            output,
            tags + ["native-calibration", "all-sequence"],
        )

    author.start_wandb = start
    author.finalize_tracking = lambda run, output, summary, names: original_finalize(
        run, output, summary, names + ["calibration_source.py"]
    )
    sys.argv = [__file__, *remaining]
    author.main()


if __name__ == "__main__":
    main()

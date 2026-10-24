"""Full propagation through pinned author predictors, with a causal read mask.

The author runner owns frames, prompt, native writes, scoring, replay and W&B.
Only the iterator is wrapped; no author model/encoder/decoder is replaced.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.stale_foreground import StaleForegroundGate  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.false_alarm_statistics import paired_method_summary, propagation_summary  # noqa: E402

PROTOCOL = "protocols/full_propagation_false_alarm_v1.json"
SOURCES = [
    PROTOCOL, "scripts/run_stale_foreground.py", "modeling/stale_foreground.py",
    "scripts/false_alarm_statistics.py", "scripts/run_author_baseline.py",
    "modeling/region_bias.py", "modeling/region_definitions.py",
    "modeling/region_tokens.py", "scripts/memory_token_io.py",
    "training/tracking.py",
]
CONDITIONS = {
    "sam2": ("sam2", None),
    "samurai": ("samurai", None),
    "samurai_noop": ("samurai", False),
    "samurai_fg_gate": ("samurai", True),
}


def protection_check(path):
    record = json.loads(Path(path).read_text())
    if not record.get("backup_verified") or not record.get("drive_web_verified"):
        raise ValueError("Verified backup and Drive-web upload receipt required before inference")
    for field in ("backup_manifest", "source_root"):
        if not Path(record[field]).exists():
            raise ValueError(f"Protection evidence missing: {field}")
    if author.digest(Path(record["backup_manifest"])) != record["backup_manifest_sha256"]:
        raise ValueError("Backup manifest checksum mismatch")
    if not str(record.get("drive_folder_url", "")).startswith(
        "https://drive.google.com/drive/folders/"
    ):
        raise ValueError("Observed Drive folder URL required, not a local-folder assumption")
    return record


def gate_suite_check(root, sources, protection_sha):
    identities = []
    for condition in CONDITIONS:
        folder = Path(root) / condition
        record = json.loads((folder / "validation.json").read_text())
        delivered = json.loads((folder / "delivery.json").read_text())
        identity = record["signature"]
        if not (
            record["passed"] and delivered["completed"]
            and delivered["wandb_artifact_upload_complete"]
            and identity["propagation_condition"] == condition
            and identity["propagation_sources"] == sources
            and identity["protection_receipt_sha256"] == protection_sha
            and identity["device"] == "cpu"
            and record["propagation_gate_verified"]
            and record["gate_sequences"] == ["seq4", "seq9"]
        ):
            raise ValueError(f"Gate suite failed or changed: {condition}")
        identities.append((identity["checkpoint_sha256"], identity["manifest_sha256"]))
    if len(set(identities)) != 1:
        raise ValueError("Gate checkpoint/manifest identities differ across conditions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--protection-receipt", type=Path, required=True)
    parser.add_argument("--native-reference-dir", type=Path)
    parser.add_argument("--gate-suite-dir", type=Path)
    options, remaining = parser.parse_known_args()
    protection_check(options.protection_receipt)
    if "--method" in remaining or "--validation-sequence" in remaining:
        parser.error("Condition fixes author method and gate sequences")
    if options.condition == "samurai_noop" and options.native_reference_dir is None:
        parser.error("No-op gate requires the fresh native SAMURAI validation directory")
    if options.condition == "samurai_noop" and "--smoke" not in remaining:
        parser.error("No-op is a validation condition, not an extra full-cohort method")
    if "--device" in remaining:
        if remaining[remaining.index("--device") + 1] != "cpu":
            parser.error("This frozen experiment requires CPU FP32")
    else:
        remaining += ["--device", "cpu"]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT,
        check=True, capture_output=True,
    )
    freeze = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", *SOURCES], cwd=ROOT, text=True
    ).strip()
    sources = {s: author.digest(ROOT / s) for s in SOURCES}
    protection_sha = author.digest(options.protection_receipt)
    if "--smoke" not in remaining:
        if options.gate_suite_dir is None:
            parser.error("All four validations must pass before any full experiment")
        gate_suite_check(options.gate_suite_dir, sources, protection_sha)
    original_signature = author.signature

    def signature(*args):
        return {
            **original_signature(*args),
            "propagation_condition": options.condition,
            "propagation_sources": sources,
            "protection_receipt_sha256": protection_sha,
            "protocol_commit": freeze,
        }

    author.signature = signature
    original_iterator, original_rollout = author.iter_predictions, author.rollout
    method, enabled = CONDITIONS[options.condition]
    context = {}

    def predictions(model, state, name, prompt):
        controller = (
            StaleForegroundGate(model, context["sequence"], prompt, enabled=enabled)
            if enabled is not None else None
        )
        with controller if controller is not None else nullcontext():
            for index, ids, logits in original_iterator(model, state, name, prompt):
                if controller is not None:
                    trace = controller.capture(state, index, logits)
                    if not context["repeat"]:
                        with (context["folder"] / "foreground_gate.jsonl").open("a") as stream:
                            stream.write(json.dumps(trace) + "\n")
                yield index, ids, logits

    def rollout(model, name, frames, prompt, folder, device, run,
                repeat=False, capture_logits=False):
        context.update(sequence=frames[0].video_id, folder=folder, repeat=repeat)
        rows, seconds = original_rollout(
            model, name, frames, prompt, folder, device, run, repeat, capture_logits
        )
        if not repeat:
            (folder / "false_alarm_summary.json").write_text(
                json.dumps(propagation_summary(rows), indent=2) + "\n"
            )
            if options.condition == "samurai_noop":
                reference = options.native_reference_dir / frames[0].video_id / "frames.jsonl"
                native = [json.loads(line) for line in reference.read_text().splitlines()]
                if [(r["source_frame_index"], r["prediction_sha256"]) for r in rows] != [
                    (r["source_frame_index"], r["prediction_sha256"]) for r in native
                ]:
                    raise RuntimeError("Read-only/no-op hook changes native SAMURAI predictions")
        return rows, seconds

    author.iter_predictions, author.rollout = predictions, rollout
    original_start = author.start_wandb

    def start(project, entity, name, mode, url, config, output, tags):
        config["false_alarm_protocol"] = json.loads((ROOT / PROTOCOL).read_text())
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        return original_start(
            project, entity, f"19 | Full propagation | {options.condition} | {config['scope']}",
            mode, url, config, output, tags + ["full-propagation", "post-hoc"],
        )

    author.start_wandb = start
    original_finalize = author.finalize_tracking

    def finalize(run, output, summary, names):
        if summary["scope"] == "validation":
            if options.condition == "samurai_fg_gate":
                traces = [json.loads(line)
                          for p in output.glob("seq*/foreground_gate.jsonl")
                          for line in p.read_text().splitlines()]
                layers = [layer for trace in traces for layer in trace["layers"]]
                if not any(layer["excluded_tokens"] > 0 for layer in layers):
                    raise RuntimeError("Real-data gate never excluded a token; validation incomplete")
                if not any(t["previous_object_score"] is not None
                           and t["previous_object_score"] <= 0
                           and t["current_object_score"] > 0 for t in traces):
                    raise RuntimeError("No object-score recovery observed; validation incomplete")
            path = output / "validation.json"
            record = json.loads(path.read_text())
            record["propagation_gate_verified"] = True
            record["gate_sequences"] = ["seq4", "seq9"]
            path.write_text(json.dumps(record, indent=2) + "\n")
        elif options.condition == "samurai_fg_gate":
            methods = {}
            source_indices = {}
            for condition in ("sam2", "samurai", "samurai_fg_gate"):
                folder = output.parent / condition
                completed = json.loads((folder / "summary.json").read_text())
                if completed["completed_sequences"] != 21 or not completed["completed"]:
                    raise RuntimeError("Three complete cohort conditions required for comparison")
                methods[condition] = {}
                for frame_path in folder.glob("seq*/frames.jsonl"):
                    rows = [json.loads(line) for line in frame_path.read_text().splitlines()]
                    sequence = frame_path.parent.name
                    identities = [(r["source_frame_index"], r["present"]) for r in rows]
                    if sequence in source_indices and identities != source_indices[sequence]:
                        raise RuntimeError("Methods evaluated different source-frame/GT intervals")
                    source_indices[sequence] = identities
                    methods[condition][sequence] = propagation_summary(rows)
            comparison = paired_method_summary(methods)
            (output / "propagation_comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n"
            )
            names = [*names, "propagation_comparison.json"]
        return original_finalize(run, output, summary, names)

    author.finalize_tracking = finalize
    sys.argv = [sys.argv[0], *remaining, "--method", method,
                "--validation-sequence", "seq4", "--validation-sequence", "seq9"]
    author.main()


if __name__ == "__main__":
    main()

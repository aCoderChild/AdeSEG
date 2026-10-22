"""Committed per-token attention rerun built on the verified native audit."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from modeling.token_attention import TokenAttentionObserver  # noqa: E402
from modeling.region_tokens import sam2_low_res_mask  # noqa: E402
from scripts import audit_region_definitions as geometry  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.relabel_region_trimaps import amended_adapter  # noqa: E402
from scripts.token_attention_statistics import (  # noqa: E402
    footprint_precision,
    sequence_summary,
    cohort_summary,
)

PROTOCOL = "protocols/token_attention_rerun_v1.json"
SOURCES = [
    PROTOCOL,
    "protocols/region_definition_trimap_amendment_2026_10_23.json",
    "modeling/token_attention.py",
    "scripts/run_token_attention.py",
    "scripts/token_attention_statistics.py",
    "scripts/relabel_region_trimaps.py",
    "scripts/queue_token_attention.py",
]
CONTEXT = {}


class PerTokenController(geometry.DefinitionObserver):
    def __init__(self, *args):
        super().__init__(*args)
        self.observer = TokenAttentionObserver(self.model.memory_attention, self._store)
        self.query_cache = {}

    def _before(self, *args):
        super()._before(*args)
        if self.frame not in self.query_cache:
            truth = load_label_mask(CONTEXT["frames"][self.frame].mask_path, [1]) == 1
            self.query_cache[self.frame] = geometry.region_label_schemes(
                sam2_low_res_mask(truth)
            )["mixed_cells"]["labels"]
        self.observer.query_labels = self.query_cache[self.frame]

    def _store(self, layer, joint, counts, row):
        path = CONTEXT["folder"] / f"token_attention_{self.frame:06d}_{layer}.npz"
        if path.exists():
            raise RuntimeError("Duplicate target/layer token capture")
        np.savez_compressed(
            path,
            joint_mass=joint,
            query_counts=counts,
            memory_frames=np.array([unit["frame"] for unit in self.units], np.int32),
            pointer_count=np.int32(self.pointers),
            target=np.int32(self.frame),
            layer=np.int32(layer),
        )
        with np.load(path) as saved:
            if (
                not np.array_equal(saved["joint_mass"], joint)
                or abs(float(saved["joint_mass"].sum()) - 1) > 1e-4
            ):
                raise RuntimeError("Per-token storage round-trip failed")


def main():
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *SOURCES],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *SOURCES],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    protocol_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", *SOURCES], cwd=ROOT, text=True
    ).strip()
    base_signature = author.signature

    def signature(*args):
        return {
            **base_signature(*args),
            "token_protocol_commit": protocol_commit,
            "token_attention_sources": {
                name: author.digest(ROOT / name) for name in SOURCES
            },
        }

    author.signature = signature
    base_schemes = geometry.region_label_schemes

    def schemes(low):
        return {**base_schemes(low), **amended_adapter().region_label_schemes(low)}

    geometry.region_label_schemes = schemes
    geometry.DefinitionObserver = PerTokenController
    base_metrics = geometry.frame_metrics

    def metrics(frame, low, labels, index, prompt):
        record = base_metrics(frame, low, labels, index, prompt)
        truth = sam2_low_res_mask(load_label_mask(frame.mask_path, [1]) == 1) > 0
        for name, scheme in labels.items():
            record["schemes"][name]["self_precision"] = footprint_precision(
                (low > 0).numpy(), scheme["labels"]
            )
            record["schemes"][name]["gt_precision"] = footprint_precision(
                truth.numpy(), scheme["labels"]
            )
        return record

    geometry.frame_metrics = metrics
    base_rollout = author.rollout

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
        CONTEXT.update(folder=folder, frames=frames)
        try:
            result = base_rollout(
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
            if not repeat:
                records = [
                    json.loads(line)
                    for line in (folder / "assignment_frames.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for row in records:
                    if row["conditioning"]:
                        row["outcome"] = "conditioning"
                        continue
                    truth = load_label_mask(frames[row["frame"]].mask_path, [1]) == 1
                    with Image.open(folder / f"frame_{row['frame']:06d}.png") as image:
                        predicted = np.asarray(image) > 0
                    row["outcome"] = (
                        (
                            "overlap_present"
                            if (truth & predicted).any()
                            else "complete_miss"
                        )
                        if truth.any()
                        else ("false_positive" if predicted.any() else "correct_empty")
                    )
                (folder / "assignment_frames.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
                token_files = list(folder.glob("token_attention_*.npz"))
                if len(token_files) != len(result[0]) * 4:
                    raise RuntimeError("Missing token attention target/layer files")
                summary = sequence_summary(folder)
                summary["compressed_token_bytes"] = sum(
                    p.stat().st_size for p in token_files
                )
                (folder / "token_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
            return result
        finally:
            CONTEXT.clear()

    author.rollout = rollout
    base_start, base_finalize = author.start_wandb, author.finalize_tracking

    def start(project, entity, name, mode, url, config, output, tags):
        config["token_protocol"] = json.loads((ROOT / PROTOCOL).read_text())
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(ROOT / PROTOCOL, output / "token_protocol.json")
        with zipfile.ZipFile(
            output / "token_sources.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for source in SOURCES:
                archive.write(ROOT / source, source)
        return base_start(
            project,
            entity,
            f"16 | Per-token regional attention | {config['scope']}",
            mode,
            url,
            config,
            output,
            tags + ["per-token", "fixed-summary"],
        )

    def finalize(run, output, summary, names):
        cohort_summary(output)
        return base_finalize(
            run,
            output,
            summary,
            names
            + [
                "token_results.json",
                "token_results.md",
                "token_protocol.json",
                "token_sources.zip",
            ],
        )

    author.start_wandb, author.finalize_tracking = start, finalize
    geometry.main()


if __name__ == "__main__":
    main()

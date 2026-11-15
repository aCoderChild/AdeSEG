"""Run 169: engineering gate, then a fixed one-seed contrastive-loss ablation.

Reuse the verified native rollout, autoregressive trainer, evaluator and author
loss. This runner manages provenance and matched conditions, not a new VOS model.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from datasets.video import load_manifest
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust
from modeling.complementary_memory import ComplementaryMemoryReader
from modeling.complementary_memory.contrastive import ForegroundContextContrast, SUPCON_REVISION
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts import run_complementary_memory_pilot as base
from scripts.verify_complementary_memory_pilot import rollout, compare_exact

PROTOCOL = ROOT / "protocols/foreground_context_contrast_v1.json"
SEED = 20261106
ARMS = {"no_contrast": 0., "contrastive": .1}
EXTRA_SOURCES = (
    "protocols/foreground_context_contrast_v1.json",
    "scripts/run_foreground_context_contrast.py",
    "modeling/complementary_memory/contrastive.py",
    "tests/test_foreground_context_contrast.py",
    "external/SupContrast/losses.py", "external/SupContrast/LICENSE",
)


def status(out, phase, **values):
    base.atomic_json(out / "run_status.json", {
        "phase": phase, "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(), **values,
    })
    print(json.dumps({"phase": phase, **values}, default=str), flush=True)


def make_setup(args, videos, out):
    _, baseline, _, repo = base.make_setup(args, videos)
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/SupContrast"), "rev-parse", "HEAD"], text=True).strip()
    if revision != SUPCON_REVISION:
        raise RuntimeError("SupContrast revision changed")
    protocol = json.loads(PROTOCOL.read_text())
    committed = subprocess.check_output(
        ["git", "show", "HEAD:protocols/foreground_context_contrast_v1.json"], cwd=ROOT)
    if hashlib.sha256(committed).hexdigest() != digest(PROTOCOL):
        raise RuntimeError("Protocol must be committed before launch")
    if (protocol["training"]["auxiliary_weight"] != ARMS["contrastive"]
            or protocol["training"]["seeds"] != [SEED]):
        raise RuntimeError("Runner settings differ from frozen protocol")
    sources = baseline["sources"] | {name: digest(ROOT / name) for name in EXTRA_SOURCES}
    core = {"baseline_setup": baseline, "protocol": protocol, "sources": sources,
            "protocol_commit": base.git_text("log", "-1", "--format=%H", "--", str(PROTOCOL)),
            "supcontrast_revision": revision}
    core_hash = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    if args.resume:
        saved = json.loads((out / "setup.json").read_text())
        if saved["core_sha256"] != core_hash:
            raise RuntimeError("Resume code/data/protocol differs from original attempt")
        if digest(out / "source_snapshot.zip") != saved["snapshot_sha256"]:
            raise RuntimeError("Source snapshot changed")
    else:
        with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(sources):
                archive.write(ROOT / name, name)
        base.atomic_json(out / "setup.json", {
            "core": core, "core_sha256": core_hash, "command": sys.argv,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_sha256": digest(out / "source_snapshot.zip"),
            "torch": torch.__version__, "python": sys.version,
        })
    return core, core_hash, repo


def guard(core, videos, name=None):
    for rel, expected in core["sources"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"Code changed during experiment: {rel}")
    if name is not None:
        for item in core["baseline_setup"]["input_receipts"][name]["frames"]:
            frame = videos[name][item["local_frame"]]
            if (digest(frame.image_path) != item["image_sha256"]
                    or digest(frame.mask_path) != item["mask_sha256"]):
                raise RuntimeError(f"Input changed during experiment: {name}")


def factory(frames, weight):
    targets = {i: base.target_for(frame)[0] for i, frame in enumerate(frames)}
    return lambda reader, observer: ForegroundContextContrast(reader, observer, targets, weight)


def engineering_gate(args, videos, repo, out, core, core_hash):
    folder = out / "gate"
    folder.mkdir(exist_ok=True)
    receipt = folder / "gate.json"
    if args.resume and receipt.exists():
        gate = json.loads(receipt.read_text())
        if gate.get("passed") is not True or gate.get("core_sha256") != core_hash:
            raise RuntimeError("Invalid saved engineering gate")
        return
    parity, learning = [], []
    for name in ("seq2", "seq3"):
        guard(core, videos, name)
        status(out, "engineering_parity", sequence=name)
        native = rollout(args.checkpoint, repo, videos[name], folder)
        torch.manual_seed(SEED)
        reader = ComplementaryMemoryReader().eval()
        adapted = rollout(args.checkpoint, repo, videos[name], folder, reader)
        parity.append(compare_exact(native, adapted, name, "regional"))
        base.atomic_json(folder / f"{name}_parity.json", {"native": native, "reader": adapted})
    for arm, weight in ARMS.items():
        torch.manual_seed(SEED)
        reader = ComplementaryMemoryReader().train()
        initial = base.state_digest(reader.state_dict())
        optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-4, weight_decay=.01)
        model = base.init_predictor(args.checkpoint, repo)
        original = parameter_digest(model)
        for name in ("seq2", "seq3"):
            status(out, "engineering_training", arm=arm, sequence=name)
            row = base.train_video(model, reader, "regional", optimizer, videos[name], name,
                                   folder, out / "progress.json", SEED, arm, max_post_frames=8,
                                   auxiliary_factory=factory(videos[name], weight))
            if len(row["chunks"]) != 2 or any(
                len(chunk["auxiliary"]) != 4 for chunk in row["chunks"]
            ):
                raise RuntimeError("Engineering training did not exercise the auxiliary path")
            if parameter_digest(model) != original or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen SAM2 changed during training")
            if base.state_digest(reader.state_dict()) == initial:
                raise RuntimeError("Reader did not learn")
            learning.append({"arm": arm, **row})
        del model
        gc.collect()
        status(out, "engineering_active_propagation", arm=arm)
        active = rollout(args.checkpoint, repo, videos["seq3"], folder, reader.eval(), max_targets=3)
        base.atomic_json(folder / f"{arm}_active.json", active)
        if not any(r["reader_update_l1"] > 0 for r in active["reader_calls"]):
            raise RuntimeError("Trained reader has no effect on the actual forward")
    guard(core, videos)
    base.atomic_json(receipt, {"passed": True, "core_sha256": core_hash,
                              "parity": parity, "training": learning})
    status(out, "engineering_passed", post_prompt_parity_targets=73)


def train_arm(args, arm, weight, videos, repo, out, core, core_hash):
    folder = out / "training" / arm
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    reader = ComplementaryMemoryReader().train()
    initial = base.state_digest(reader.state_dict())
    optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-4, weight_decay=.01)
    model = base.init_predictor(args.checkpoint, repo)
    original = parameter_digest(model)
    checkpoint = folder / "checkpoint.pt"
    completed = []
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if (saved["core_sha256"] != core_hash or saved["arm"] != arm
                or saved["initial_sha256"] != initial):
            raise RuntimeError("Training checkpoint provenance mismatch")
        completed = saved["completed"]
        if [r["sequence"] for r in completed] != base.TRAIN_SEQUENCES[:len(completed)]:
            raise RuntimeError("Training checkpoint sequence coverage is invalid")
        for row in completed:
            if digest(folder / f"{row['sequence']}.json") != row["receipt_sha256"]:
                raise RuntimeError("Training receipt changed")
        reader.load_state_dict(saved["reader"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["rng"])
    for index in range(len(completed), len(base.TRAIN_SEQUENCES)):
        name = base.TRAIN_SEQUENCES[index]
        guard(core, videos, name)
        status(out, "training", arm=arm, sequence=name, completed_sequences=index,
               expected_sequences=13, expected_updates=415)
        row = base.train_video(model, reader, "regional", optimizer, videos[name], name,
                               out, out / "progress.json", SEED, arm,
                               auxiliary_factory=factory(videos[name], weight))
        if parameter_digest(model) != original:
            raise RuntimeError("Training changed frozen SAM2")
        row["arm"] = arm
        base.atomic_json(folder / f"{name}.json", row)
        completed.append({"sequence": name, "receipt_sha256": digest(folder / f"{name}.json"),
                          "frames": row["post_prompt_frames"], "updates": row["optimizer_updates"]})
        base.atomic_torch_save(checkpoint, {
            "core_sha256": core_hash, "arm": arm, "initial_sha256": initial,
            "reader": reader.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": torch.random.get_rng_state(), "completed": completed,
        })
    if sum(r["frames"] for r in completed) != 1640 or sum(r["updates"] for r in completed) != 415:
        raise RuntimeError("Training coverage differs from frozen protocol")
    base.atomic_json(folder / "completion.json", {
        "completed": completed, "initial_sha256": initial,
        "checkpoint_sha256": digest(checkpoint), "reader_sha256": base.state_digest(reader.state_dict()),
        "original_sam2_parameter_sha256": original, "core_sha256": core_hash,
    })
    del model
    gc.collect()
    return reader.eval()


def report(summaries, out):
    result = {"per_sequence": summaries, "means": {}, "contrasts": {}}
    names = base.TEST_SEQUENCES
    for arm, records in summaries.items():
        if set(records) != set(names) or sum(r["target_frames"] for r in records.values()) != 372:
            raise RuntimeError("Evaluation coverage is incomplete")
        result["means"][arm] = {
            "present_dice": float(np.mean([records[n]["present_dice"] for n in names])),
            "present_iou": float(np.mean([records[n]["present_iou"] for n in names])),
            "present_boundary_f": float(np.mean([records[n]["present_boundary_f"] for n in names])),
            "false_alarms": sum(r["gt_empty_false_alarm_count"] for r in records.values()),
            "gt_empty_frames": sum(r["gt_empty_frames"] for r in records.values()),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in records.values()),
        }
    passed, pvalues = True, {}
    for control, margin in (("native", .02), ("no_contrast", .01)):
        left = {n: summaries["contrastive"][n]["present_dice"] for n in names}
        right = {n: summaries[control][n]["present_dice"] for n in names}
        delta = {n: left[n] - right[n] for n in names}
        mean = float(np.mean(list(delta.values())))
        ci = sequence_bootstrap_interval(delta, seed=SEED)
        fa_delta = float(np.mean([summaries["contrastive"][n]["gt_empty_false_alarm_rate"]
                                  - summaries[control][n]["gt_empty_false_alarm_rate"] for n in names]))
        pvalues[control] = paired_wilcoxon(left, right)["p"]
        result["contrasts"][control] = {
            "mean_dice_difference": mean, "sequence_bootstrap_95_ci": ci,
            "per_sequence_delta": delta, "wins": sum(v > 0 for v in delta.values()),
            "ties": sum(v == 0 for v in delta.values()), "losses": sum(v < 0 for v in delta.values()),
            "equal_sequence_false_alarm_difference": fa_delta,
            "leave_one_sequence_out_mean": {n: float(np.mean([v for k, v in delta.items() if k != n])) for n in names},
        }
        passed &= mean >= margin and ci[0] > 0 and fa_delta <= .05
    result["holm_pvalues"] = holm_adjust(pvalues)
    result["pilot_screen_passed"] = bool(passed)
    result["decision"] = "promising exploratory signal; replication required" if passed else "criterion failed; stop this candidate"
    base.atomic_json(out / "results.json", result)
    lines = ["# Run 169: foreground/context contrastive learning", "",
             "One seed; exploratory seq16–23 evaluation after training on seq2–15 (excluding seq7).",
             "Identical regional reader and frozen SAM2; contrastive loss coefficient 0 versus 0.1.", "",
             "| Condition | Present Dice | Present IoU | Boundary F | False alarms / empty frames |",
             "|---|---:|---:|---:|---:|"]
    for arm, row in result["means"].items():
        lines.append(f"| {arm} | {row['present_dice']:.6f} | {row['present_iou']:.6f} | {row['present_boundary_f']:.6f} | {row['false_alarms']}/{row['gt_empty_frames']} |")
    lines += ["", "| Sequence | Native | No contrastive loss | Contrastive |", "|---|---:|---:|---:|"]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{summaries[a][n]['present_dice']:.6f}" for a in ("native", *ARMS)) + " |")
    for control, row in result["contrasts"].items():
        lines += ["", f"Contrastive minus {control}: {row['mean_dice_difference']:+.6f}; 95% sequence-bootstrap CI {row['sequence_bootstrap_95_ci']}; wins/ties/losses {row['wins']}/{row['ties']}/{row['losses']}."]
    lines += ["", "Decision: " + result["decision"] + ".", "",
              "This tests an auxiliary learning objective on regional memory features. It does not establish token semantic purity or identify illumination/occlusion causes. No additional datasets or seed sweeps are launched."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    return result


def sync_result_document(out):
    """Replace only Run169's owned block, preserving all other research text."""
    path = ROOT / "docs/REGION_TOKENS_RESULTS.md"
    start, stop = "<!-- RUN169_RESULTS_START -->", "<!-- RUN169_RESULTS_END -->"
    text = path.read_text()
    report_text = (out / "results.md").read_text().replace("# Run 169:", "## Run 169:", 1)
    block = start + "\n" + report_text + f"\nSource: `{out}`.\n" + stop
    if start in text:
        if text.count(start) != 1 or text.count(stop) != 1:
            raise RuntimeError("Ambiguous Run169 report markers")
        before, rest = text.split(start)
        _, after = rest.split(stop)
        text = before + block + after
    else:
        text += "\n" + block + "\n"
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    # All experiment temporary and cache files stay in the requested Drive tree.
    os.environ["TMPDIR"] = str(out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    try:
        videos = load_manifest(ROOT / base.MANIFEST_REL, "dev") | load_manifest(ROOT / base.MANIFEST_REL, "test")
        core, core_hash, repo = make_setup(args, videos, out)
        engineering_gate(args, videos, repo, out, core, core_hash)
        if args.gate_only:
            return
        summaries = {}
        for arm in ("native", *ARMS):
            guard(core, videos)
            reader = None if arm == "native" else train_arm(args, arm, ARMS[arm], videos, repo, out, core, core_hash)
            model = base.init_predictor(args.checkpoint, repo)
            summaries[arm] = {}
            for i, name in enumerate(base.TEST_SEQUENCES):
                guard(core, videos, name)
                status(out, "evaluation", arm=arm, sequence=name, completed_sequences=i, expected_sequences=8)
                summaries[arm][name] = base.evaluate_video(
                    model, videos[name], reader, "regional", out / "evaluation" / arm,
                    name, SEED, core_hash)
                base.atomic_json(out / "partial_results.json", summaries)
            del model, reader
            gc.collect()
        guard(core, videos)
        hashes = {r["original_sam2_parameter_sha256"] for rows in summaries.values() for r in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("Original SAM2 parameters differ between conditions")
        result = report(summaries, out)
        base.atomic_json(out / "completion.json", {
            "complete": True, "core_sha256": core_hash,
            "results_sha256": digest(out / "results.json"),
            "elapsed_seconds": time.monotonic() - started,
            "decision": result["decision"],
        })
        sync_result_document(out)
        status(out, "complete", decision=result["decision"])
    except BaseException:
        status(out, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

"""Gated E3 C0/C1 screen only. No regional efficacy or E4 is launched."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest  # noqa: E402
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval  # noqa: E402
from modeling.dense_long_memory import DenseLongMemory  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.bank_integrity import fingerprint  # noqa: E402
from scripts.false_alarm_statistics import propagation_summary  # noqa: E402
from scripts.memory_token_io import prompt_index, validate_output  # noqa: E402
from training.tracking import start_wandb, log_results_artifact, finalize_tracking  # noqa: E402

SOURCES = ["protocols/long_history_E3_screen_v1.json", "modeling/dense_long_memory.py",
           "scripts/run_long_history_screen.py", "scripts/false_alarm_statistics.py"]


def require_e2(path):
    validation = json.loads((path/"validation.json").read_text())
    delivery = json.loads((path/"delivery.json").read_text())
    if not (validation["passed"] and validation["all_token_native_bit_exact"]
            and validation["sparse_fresh_replay_bit_exact"] and validation["long_history_reads"] > 0
            and delivery["completed"] and delivery["wandb_artifact_upload_complete"]):
        raise ValueError("E2 has not passed with completed artifact delivery")
    for name, expected in validation["setup"]["sources"].items():
        if author.digest(ROOT/name) != expected:
            raise ValueError(f"E2 source changed since successful gate: {name}")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--e2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-for-e2", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    args = parser.parse_args()
    if args.wait_for_e2:
        started = time.monotonic()
        while not (args.e2_dir/"delivery.json").exists():
            if (args.e2_dir/"failure.json").exists():
                raise RuntimeError("E2 failed: do not launch E3")
            if time.monotonic()-started > 7200:
                raise RuntimeError("E2 queue timeout; no E3 launched")
            print("E3 queue waiting for complete E2 gate", flush=True)
            time.sleep(10)
    gate = require_e2(args.e2_dir)
    if author.digest(args.checkpoint) != gate["setup"]["checkpoint_sha256"] or author.digest(args.manifest) != gate["setup"]["manifest_sha256"]:
        raise ValueError("E3 checkpoint or manifest differs from E2")
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output/key.lower())
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    protocol = json.loads((ROOT/SOURCES[0]).read_text())
    setup = {"protocol": protocol, "E2_validation_sha256": author.digest(args.e2_dir/"validation.json"),
             "sources": {s: author.digest(ROOT/s) for s in SOURCES}, "E2_setup": gate["setup"],
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "23-E3 C0 C1 full-history screen - PolypGen", "online",
                     "https://forge.coreweave.com/api/wandb", setup, output, ["E3", "C0-C1", "exploratory"])
    original = author.iter_predictions
    try:
        # Preserve detailed consumed E2 receipts before new efficacy inference.
        archive = output/"consumed_E2_integrity.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in args.e2_dir.glob("seq*/*/*.json*"):
                zipped.write(path, arcname=path.relative_to(args.e2_dir))
            zipped.write(args.e2_dir/"validation.json", arcname="validation.json")
        log_results_artifact(run, output, [archive.name, "setup.json"])
        sys.path.insert(0, str(repo))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
        prompt_protocol = json.loads((ROOT/"protocols/memory_token_regions.json").read_text())

        def rollout(sequence, condition, folder, warmup_reference=None, replay_reference=None):
            frames = videos[sequence]
            prompt = prompt_index(frames, prompt_protocol)
            model = author.build_author_predictor(args, repo)
            parameters = author.parameter_fingerprints(model)
            folder.mkdir(parents=True)
            integrity, scores = {}, {}

            def predictions(predictor, state, method, initial):
                controller = DenseLongMemory(predictor, initial) if condition == "C1" else None
                native_reads = []
                handle = None
                if controller is None:
                    def observe_memory(module, args, kwargs):
                        pointers = int(kwargs.get("num_obj_ptr_tokens", 0))
                        native_reads.append({"selected_spatial_keys": kwargs["memory"].shape[0]-pointers,
                                             "pointer_keys": pointers})
                    handle = predictor.memory_attention.register_forward_pre_hook(observe_memory, with_kwargs=True)
                try:
                    with controller if controller else nullcontext():
                        for index, ids, logits in original(predictor, state, method, initial):
                            trace = controller.capture(state, index) if controller else {"native_reads": native_reads.copy()}
                            native_reads.clear()
                            outputs = state["output_dict_per_obj"][0]
                            own = outputs["cond_frame_outputs"].get(index, outputs["non_cond_frame_outputs"].get(index))
                            record = {"frame": index, "prediction_sha256": author.tensor_hash(logits),
                                      "low_res_sha256": author.tensor_hash(own["pred_masks"]),
                                      "bank_sha256": fingerprint(state["output_dict_per_obj"])["sha256"],
                                      "cache_sha256": fingerprint(state["cached_features"])["sha256"]}
                            if warmup_reference is not None and index-initial <= 7 and warmup_reference[index] != record:
                                raise RuntimeError("C1 native-horizon warmup parity failed")
                            if replay_reference is not None and replay_reference[index] != record:
                                raise RuntimeError("C1 fresh-model replay failed")
                            integrity[index] = record
                            score = float(own["object_score_logits"].item())
                            if not np.isfinite(score):
                                raise RuntimeError("Nonfinite native object score")
                            scores[index] = {"current_object_score": score, "prediction_nonempty": bool((logits > 0).any())}
                            with (folder/"integrity.jsonl").open("a") as stream:
                                stream.write(json.dumps({**record, "trace": trace, **scores[index]})+"\n")
                            yield index, ids, logits
                finally:
                    if handle is not None:
                        handle.remove()

            author.iter_predictions = predictions
            rows, seconds = author.rollout(model, "sam2", frames, prompt, folder, "cpu", run)
            author.assert_parameter_integrity(model, parameters, folder, "E3")
            for row in rows:
                row.update(scores[row["frame"]])
            result = {"sequence": sequence, "condition": condition, **propagation_summary(rows),
                      "iou": float(np.mean([r["scores"]["iou"] for r in rows if r["present"]])),
                      "boundary_f": float(np.mean([r["scores"]["boundary_f"] for r in rows if r["present"]])),
                      "wall_seconds": seconds}
            (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
            packed = output/("_".join(folder.relative_to(output).parts)+".zip")
            with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as zipped:
                for path in folder.iterdir():
                    zipped.write(path, arcname=path.name)
            log_results_artifact(run, output, [packed.name])
            del model
            gc.collect()
            return result, integrity

        for sequence in ("seq2", "seq3"):
            reference = {r["frame"]: {k: r[k] for k in ("frame", "prediction_sha256", "low_res_sha256", "bank_sha256", "cache_sha256")}
                         for r in map(json.loads, (args.e2_dir/sequence/"native_0/integrity.jsonl").read_text().splitlines())}
            _, first = rollout(sequence, "C1", output/"smoke"/sequence/"C1_first", warmup_reference=reference)
            rollout(sequence, "C1", output/"smoke"/sequence/"C1_repeat", replay_reference=first)
        (output/"engineering_smoke.json").write_text(json.dumps({"passed": True, "sequences": ["seq2", "seq3"], "sources": setup["sources"]}, indent=2)+"\n")
        results = {"C0": {}, "C1": {}, "unpromptable": []}
        for sequence in sorted(videos, key=lambda s: int(s[3:])):
            try:
                prompt_index(videos[sequence], prompt_protocol)
            except ValueError:
                results["unpromptable"].append(sequence)
                continue
            for condition in ("C0", "C1"):
                reference = None
                if condition == "C0" and sequence in ("seq2", "seq3"):
                    reference = {r["frame"]: {k: r[k] for k in ("frame", "prediction_sha256", "low_res_sha256", "bank_sha256", "cache_sha256")}
                                 for r in map(json.loads, (args.e2_dir/sequence/"native_0/integrity.jsonl").read_text().splitlines())}
                result, _ = rollout(sequence, condition, output/"full"/sequence/condition, replay_reference=reference)
                results[condition][sequence] = result
                (output/"progress.json").write_text(json.dumps({"state": "running", "results": results}, indent=2)+"\n")
        c0 = {s: r["present_dice"] for s, r in results["C0"].items()}
        c1 = {s: r["present_dice"] for s, r in results["C1"].items()}
        if len(c0) != 21 or set(c0) != set(c1):
            raise RuntimeError("Incomplete E3 matched cohort")
        differences = {s: c1[s]-c0[s] for s in c0}
        delta = float(np.mean(list(differences.values())))
        summary = {"completed": True, "stage": "E3 C0/C1 only", "results": results,
                   "C0_present_dice": float(np.mean(list(c0.values()))), "C1_present_dice": float(np.mean(list(c1.values()))),
                   "mean_paired_difference": delta, "ci95": sequence_bootstrap_interval(differences, seed=20261025),
                   "wilcoxon_p": paired_wilcoxon(c1, c0)["p"], "history_alone_screen_positive": delta > 0,
                   "C2_C3_C4_launched": False, "E4_launched": False}
        (output/"results.json").write_text(json.dumps(summary, indent=2)+"\n")
        lines = ["# E3 C0/C1 history-alone screen", "", "Exploratory PolypGen; equal-sequence present Dice. Higher-compute C1 is not the regional method.", "", "| Sequence | C0 Dice | C1 Dice | Difference |", "|---|---:|---:|---:|"]
        for seq in c0:
            lines.append(f"| {seq} | {c0[seq]:.5f} | {c1[seq]:.5f} | {differences[seq]:+.5f} |")
        lines += ["", f"Mean difference {delta:+.5f}; 95% sequence-bootstrap CI {summary['ci95']}; Wilcoxon p={summary['wilcoxon_p']:.6g}.", "", "This screen does not establish regional token utility. Negative dense history does not disprove selective regional history. C2/C3/C4 and E4 have not run."]
        (output/"results.md").write_text("\n".join(lines)+"\n")
        finalize_tracking(run, output, {k: v for k, v in summary.items() if k != "results"}, ["results.json", "results.md", "engineering_smoke.json", "setup.json"])
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"completed": False, "error": repr(error)}, indent=2)+"\n")
        run.summary.update({"completed": False, "failure": repr(error)})
        raise
    finally:
        author.iter_predictions = original
        run.finish()


if __name__ == "__main__":
    main()

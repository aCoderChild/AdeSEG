"""Gate, then compare native and region-sparse author predictors in isolation."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval, holm_adjust
from modeling.author_sparse_memory import AuthorSparseMemory
from scripts import run_author_baseline as author
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import prompt_index, validate_output
from training.tracking import start_wandb, log_results_artifact, finalize_tracking

SOURCES = ["protocols/author_region_sparse_v1.json", "modeling/author_sparse_memory.py",
           "scripts/run_author_sparse_experiment.py", "tests/test_author_sparse_memory.py",
           "modeling/sparse_memory.py", "modeling/region_definitions.py",
           "scripts/region_sparse_statistics.py", "scripts/run_author_baseline.py",
           "scripts/false_alarm_statistics.py", "evaluation/statistics.py",
           "protocols/memory_token_regions.json", "modeling/region_tokens.py",
           "scripts/bank_integrity.py", "scripts/memory_token_io.py",
           "datasets/io.py", "datasets/video.py", "evaluation/segmentation.py",
           "training/tracking.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("samurai", "sam2long"))
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if not args.method:
        args.output_dir = validate_output(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        # Different author repos export the same sam2 module: separate processes.
        for method in ("samurai", "sam2long"):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--method", method,
                       "--checkpoint", str(args.checkpoint), "--manifest", str(args.manifest),
                       "--output-dir", str(args.output_dir/method)]
            if args.gate_only:
                command.append("--gate-only")
            (args.output_dir/"queue_progress.json").write_text(json.dumps({"state": "running", "method": method})+"\n")
            subprocess.run(command, check=True)
        summaries = {m: json.loads((args.output_dir/m/"results.json").read_text()) for m in ("samurai", "sam2long")}
        adjusted = holm_adjust({m: r["wilcoxon_p"] for m, r in summaries.items()}) if not args.gate_only else {}
        (args.output_dir/"comparison.json").write_text(json.dumps({"completed": True, "gate_only": args.gate_only,
            "methods": summaries, "holm2_p": adjusted}, indent=2)+"\n")
        (args.output_dir/"queue_progress.json").write_text('{"state":"completed"}\n')
        return
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    args.device = "cpu"
    repo, revision = author.source_identity(args.method)
    sys.path.insert(0, str(repo/"sam2" if args.method == "samurai" else repo))
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output/key.lower())
    setup = {"method": args.method, "author_revision": revision,
             "protocol": json.loads((ROOT/SOURCES[0]).read_text()),
             "source_sha256": {n: author.digest(ROOT/n) for n in SOURCES},
             "checkpoint_sha256": author.digest(args.checkpoint),
             "manifest_sha256": author.digest(args.manifest), "torch": torch.__version__,
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    if setup["checkpoint_sha256"] != "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5":
        raise ValueError("Expected verified SAM2.1 B+ checkpoint")
    if setup["manifest_sha256"] != "80d83e6a4099fc727e90bf06757ae81b3292d24dc5651daf1457f4a54e248c7f":
        raise ValueError("Expected frozen PolypGen manifest")
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    with zipfile.ZipFile(output/"sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:
            z.write(ROOT/name, arcname=name)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", f"25 {args.method} region-sparse longer memory", "online",
                     "https://forge.coreweave.com/api/wandb", setup, output, ["author-region-sparse", "exploratory"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
    protocol = json.loads((ROOT/"protocols/memory_token_regions.json").read_text())
    original = author.iter_predictions

    def rollout(sequence, mode, folder):
        print(f"[author-sparse] start {args.method} {sequence} {mode}", flush=True)
        folder.mkdir(parents=True)
        frames = videos[sequence]
        prompt = prompt_index(frames, protocol)
        if args.method == "samurai":
            # Exact existing author-runner convention: author lookup starts0.
            # ManifestFrame.frame_index remains the original source identity.
            frames, prompt = frames[prompt:], 0
        model = author.build_author_predictor(args, repo)
        parameters = author.parameter_fingerprints(model)
        (folder/"inputs.json").write_text(json.dumps({"prompt_stage_index": prompt,
            "source_frame_indices": [f.frame_index for f in frames]}, indent=2)+"\n")
        holders, final = [], {}

        def predictions(predictor, state, method, initial):
            controller = AuthorSparseMemory(predictor, method, initial, identity=mode == "identity") if mode != "native" else None
            if controller:
                holders.append(controller)
            with controller if controller else nullcontext():
                for index, ids, logits in original(predictor, state, method, initial):
                    (output/"progress.json").write_text(json.dumps({"state": folder.relative_to(output).parts[0],
                        "sequence": sequence, "mode": mode, "frame": index, "last_frame": len(frames)-1})+"\n")
                    yield index, ids, logits
                final.update(bank_sha256=fingerprint(state["output_dict_per_obj"])["sha256"],
                             cache_sha256=fingerprint(state["cached_features"])["sha256"])

        author.iter_predictions = predictions
        rows, seconds = author.rollout(model, args.method, frames, prompt, folder, "cpu", run)
        author.assert_parameter_integrity(model, parameters, folder, mode)
        calls = holders[0].calls if holders else []
        if holders:
            expected = len(rows) if args.method == "samurai" else 1+3*(len(rows)-1)
            if len(calls) != expected:
                raise RuntimeError("Missing author branch reads")
        (folder/"reads.jsonl").write_text("".join(json.dumps(r)+"\n" for r in calls))
        summary = {"method": args.method, "mode": mode, "sequence": sequence,
                   **propagation_summary(rows), "wall_seconds": seconds,
                   "iou": float(np.mean([r["scores"]["iou"] for r in rows if r["present"]])),
                   "boundary_f": float(np.mean([r["scores"]["boundary_f"] for r in rows if r["present"]])),
                   "query_key_pairs": sum(layer["query_tokens"]*(layer["selected_spatial_keys"]+layer["pointer_keys"]) for call in calls for layer in call["layers"]),
                   "executed_branch_reads": len(calls), **final}
        (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        receipt = {"predictions": [r["prediction_sha256"] for r in rows], **final}
        archive = output/("_".join(folder.relative_to(output).parts)+".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for f in folder.iterdir():
                z.write(f, arcname=f.name)
        log_results_artifact(run, output, [archive.name])
        del model
        gc.collect()
        print(f"[author-sparse] completed {args.method} {sequence} {mode}", flush=True)
        return summary, receipt, calls

    try:
        log_results_artifact(run, output, ["setup.json", "sources.zip"])
        extended_reads = 0
        for seq in ("seq2", "seq3"):
            _, native, _ = rollout(seq, "native", output/"gate"/seq/"native")
            _, identity, _ = rollout(seq, "identity", output/"gate"/seq/"identity")
            if identity != native:
                raise RuntimeError(f"All-token native parity failed: {seq}")
            _, first, calls = rollout(seq, "regional", output/"gate"/seq/"regional_first")
            _, repeat, repeated_calls = rollout(seq, "regional", output/"gate"/seq/"regional_repeat")
            if repeat != first or repeated_calls != calls:
                raise RuntimeError(f"Regional fresh replay failed: {seq}")
            extended_reads += sum(len(c['layers'][0]['frames']) > 7 for c in calls)
        if extended_reads == 0:
            raise RuntimeError("Gate did not exercise expanded author history")
        validation = {"passed": True, "sequences": ["seq2", "seq3"], "all_token_bit_exact": True,
                      "regional_replay_bit_exact": True, "four_layers_all_branches": True,
                      "expanded_history_reads": extended_reads}
        (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        log_results_artifact(run, output, ["validation.json"])
        if args.gate_only:
            result = {"completed": True, "gate_only": True, **validation}
        else:
            if any(author.digest(ROOT/n) != sha for n, sha in setup["source_sha256"].items()):
                raise RuntimeError("Source changed during gate")
            results = {"native": {}, "regional": {}, "unpromptable": []}
            for seq in sorted(videos, key=lambda s: int(s[3:])):
                try:
                    prompt_index(videos[seq], protocol)
                except ValueError:
                    results["unpromptable"].append(seq)
                    continue
                for mode in ("native", "regional"):
                    results[mode][seq], _, _ = rollout(seq, mode, output/"full"/seq/mode)
                for key in ("targets", "present_targets", "gt_empty_targets"):
                    if results["native"][seq][key] != results["regional"][seq][key]:
                        raise RuntimeError("Paired cohort mismatch")
            if len(results["native"]) != 21 or results["unpromptable"] != ["seq1", "seq7"]:
                raise RuntimeError("Incomplete author cohort")
            native = {s: r["present_dice"] for s, r in results["native"].items()}
            regional = {s: r["present_dice"] for s, r in results["regional"].items()}
            differences = {s: regional[s]-native[s] for s in native}
            result = {"completed": True, "exploratory": True, "results": results,
                      "native_present_dice": float(np.mean(list(native.values()))),
                      "regional_present_dice": float(np.mean(list(regional.values()))),
                      "mean_difference": float(np.mean(list(differences.values()))),
                      "ci95": sequence_bootstrap_interval(differences, seed=20261025),
                      "wilcoxon_p": paired_wilcoxon(regional, native)["p"],
                      "holm2": "Computed by parent queue only after both authors finish"}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        finalize_tracking(run, output, {k: v for k, v in result.items() if k != "results"}, ["results.json", "validation.json", "setup.json"])
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"error": repr(error), "no_bypass": True})+"\n")
        raise
    finally:
        author.iter_predictions = original
        run.finish()


if __name__ == "__main__":
    main()

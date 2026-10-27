"""Exploratory C2/C3 amendment: full replay gate, then matched full cohort."""

import argparse
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
from datasets.video import load_manifest  # noqa: E402
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval  # noqa: E402
from modeling.matched_region_memory import MatchedRegionMemory  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.bank_integrity import fingerprint  # noqa: E402
from scripts.false_alarm_statistics import propagation_summary  # noqa: E402
from scripts.memory_token_io import prompt_index, validate_output  # noqa: E402
from scripts.run_long_history_screen import require_e2  # noqa: E402
from training.tracking import start_wandb, log_results_artifact, finalize_tracking  # noqa: E402

PROTOCOL = "protocols/region_sparse_C2_C3_amendment_v1.json"
SOURCES = [PROTOCOL, "modeling/matched_region_memory.py", "scripts/run_matched_region_experiment.py",
           "tests/test_matched_region_memory.py"]
CORE_KEYS = ("frame", "prediction_sha256", "low_res_sha256", "bank_sha256", "cache_sha256")


def load_reference(path, policy=None):
    records = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        trace = row["trace"]
        if policy is not None and trace.get("policy", "legacy_v1") != policy:
            raise ValueError("Reference selection policy differs from this experiment")
        index = row["frame"]
        if trace["condition"] != "C2" or trace["frame"] != index or index in records:
            raise ValueError("Invalid/duplicate C2 reference frame")
        bits = bytes.fromhex(trace["keep_bits_hex"])
        if len(bits) != 512 or int(np.unpackbits(np.frombuffer(bits, dtype=np.uint8)).sum()) != trace["retained_tokens"]:
            raise ValueError("C2 reference selector bits/count disagree")
        records[index] = trace
    if not records or sorted(records) != list(range(min(records), max(records)+1)):
        raise ValueError("C2 reference trajectory is incomplete")
    return records


def primary_statistics(results, baseline, seeds):
    c0 = {s: r["present_dice"] for s, r in baseline["results"]["C0"].items()}
    c2 = {s: r["present_dice"] for s, r in results["C2"].items()}
    if set(c2) != set(c0) or any(set(results["C3"][str(seed)]) != set(c2) for seed in seeds):
        raise ValueError("Incomplete matched cohort or missing C3 seed")
    c3 = {s: float(np.mean([results["C3"][str(seed)][s]["present_dice"] for seed in seeds])) for s in c2}
    contrasts = []
    for name, reference in (("C2-C0", c0), ("C2-C3_seed_mean", c3)):
        differences = {s: c2[s]-reference[s] for s in c2}
        contrasts.append({"contrast": name, "n_sequences": len(c2),
                          "mean_difference": float(np.mean(list(differences.values()))),
                          "ci95": sequence_bootstrap_interval(differences, seed=20261025),
                          "wilcoxon_p": paired_wilcoxon(c2, reference)["p"],
                          "sequence_differences": differences})
    adjusted = holm_adjust({r["contrast"]: r["wilcoxon_p"] for r in contrasts})
    for row in contrasts:
        row["holm2_p"] = adjusted[row["contrast"]]
    return {"C0_present_dice": float(np.mean(list(c0.values()))),
            "C1_present_dice": baseline["C1_present_dice"],
            "C2_present_dice": float(np.mean(list(c2.values()))),
            "C3_seed_mean_present_dice": float(np.mean(list(c3.values()))),
            "C3_present_dice_by_seed": {str(seed): float(np.mean([r["present_dice"] for r in results["C3"][str(seed)].values()])) for seed in seeds},
            "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--e2-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--policy", choices=("legacy_v1", "contribution_v2"), default="contribution_v2",
                        help="Versioned selection rule; historical results use legacy_v1")
    args = parser.parse_args()
    controller_class = MatchedRegionMemory
    protocol_path = PROTOCOL
    sources = list(SOURCES)
    if args.policy == "contribution_v2":
        from modeling.contribution_region_memory import ContributionRegionMemory
        controller_class = ContributionRegionMemory
        protocol_path = "protocols/region_contribution_C2_v2.json"
        sources += [protocol_path, "modeling/contribution_region_memory.py",
                    "tests/test_contribution_region_memory.py"]
    gate = require_e2(args.e2_dir)
    baseline = json.loads((args.baseline_dir/"results.json").read_text())
    delivered = json.loads((args.baseline_dir/"delivery.json").read_text())
    prior_setup = json.loads((args.baseline_dir/"setup.json").read_text())
    if not (baseline["completed"] and delivered["completed"] and delivered["wandb_artifact_upload_complete"]
            and len(baseline["results"]["C0"]) == len(baseline["results"]["C1"]) == 21):
        raise ValueError("Completed C0/C1 evidence is required")
    for name, sha in prior_setup["sources"].items():
        if author.digest(ROOT/name) != sha:
            raise ValueError(f"Original C0/C1 source changed: {name}")
    if (author.digest(args.checkpoint) != gate["setup"]["checkpoint_sha256"]
            or author.digest(args.manifest) != gate["setup"]["manifest_sha256"]
            or prior_setup["E2_setup"]["checkpoint_sha256"] != gate["setup"]["checkpoint_sha256"]
            or prior_setup["E2_setup"]["manifest_sha256"] != gate["setup"]["manifest_sha256"]):
        raise ValueError("Checkpoint/manifest mismatch")
    protocol = json.loads((ROOT/protocol_path).read_text())
    known = protocol["known_prior_results"]
    if not all(np.isclose(baseline[key], known[key], atol=1e-12, rtol=0)
               for key in ("C0_present_dice", "C1_present_dice")):
        raise ValueError("Amendment does not match the known C1 result")
    subprocess.run(["git", "ls-files", "--error-unmatch", *sources], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *sources], cwd=ROOT, check=True, capture_output=True)
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output/key.lower())
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    source_hashes = {s: author.digest(ROOT/s) for s in sources}
    setup = {"policy": args.policy, "protocol": protocol, "sources": source_hashes, "E2_setup": gate["setup"],
             "E2_validation_sha256": author.digest(args.e2_dir/"validation.json"),
             "baseline_results_sha256": author.digest(args.baseline_dir/"results.json"),
             "baseline_setup_sha256": author.digest(args.baseline_dir/"setup.json"),
             "author_revision": revision, "torch": torch.__version__,
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    (output/"baseline_results.json").write_text(json.dumps(baseline, indent=2)+"\n")
    with zipfile.ZipFile(output/"sources.zip", "w", zipfile.ZIP_DEFLATED) as packed:
        for name in {*sources, *gate["setup"]["sources"], *prior_setup["sources"],
                     "external/Efficient-SAM2/build_model/window_bypass.py",
                     "external/sam2/sam2/modeling/sam/transformer.py",
                     "external/sam2/sam2/modeling/position_encoding.py"}:
            packed.write(ROOT/name, arcname=name)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", f"C2/C3 {args.policy} - PolypGen", "online",
                     "https://forge.coreweave.com/api/wandb", setup, output, ["E3", "C2-C3", "exploratory-amendment"])
    original = author.iter_predictions
    seeds = protocol["fixed_settings"]["random_seeds"]
    try:
        log_results_artifact(run, output, ["setup.json", "baseline_results.json", "sources.zip"])
        sys.path.insert(0, str(repo))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
        prompt_protocol = json.loads((ROOT/"protocols/memory_token_regions.json").read_text())
        completed_runs = []

        def rollout(sequence, condition, seed, folder, reference_path=None, replay=None, old_e2=None):
            frames = videos[sequence]
            prompt = prompt_index(frames, prompt_protocol)
            reference = load_reference(reference_path, args.policy) if reference_path else None
            if reference is not None and sorted(reference) != list(range(prompt, len(frames))):
                raise ValueError("Reference frame range differs from manifest")
            reference_sha = author.digest(reference_path) if reference_path else None
            phase = folder.relative_to(output).parts[0]
            print(f"[matched] start phase={phase} sequence={sequence} condition={condition} seed={seed}", flush=True)
            model = author.build_author_predictor(args, repo)
            parameters = author.parameter_fingerprints(model)
            folder.mkdir(parents=True)
            integrity, metadata = {}, {}

            def predictions(predictor, state, method, initial):
                with controller_class(predictor, initial, sequence, reference=reference, seed=seed) as controller:
                    for index, ids, logits in original(predictor, state, method, initial):
                        trace = controller.capture(state, index)
                        outputs = state["output_dict_per_obj"][0]
                        own = outputs["cond_frame_outputs"].get(index, outputs["non_cond_frame_outputs"].get(index))
                        core = {"frame": index, "prediction_sha256": author.tensor_hash(logits),
                                "low_res_sha256": author.tensor_hash(own["pred_masks"]),
                                "bank_sha256": fingerprint(state["output_dict_per_obj"])["sha256"],
                                "cache_sha256": fingerprint(state["cached_features"])["sha256"],
                                **{k: trace[k] for k in ("allocation", "retained_tokens", "keep_bits_hex")}}
                        if replay is not None and replay.get(index) != core:
                            raise RuntimeError(f"Fresh-model replay differs: {sequence} {condition} seed={seed} frame={index}")
                        if old_e2 is not None and {k: core[k] for k in CORE_KEYS} != old_e2[index]:
                            raise RuntimeError("C2 differs from the validated E2 region_long trajectory")
                        integrity[index] = core
                        score = float(own["object_score_logits"].item())
                        if not np.isfinite(score):
                            raise RuntimeError("Nonfinite native object score")
                        metadata[index] = {"current_object_score": score, "prediction_nonempty": bool((logits > 0).any()),
                                           "memory_attention_seconds": trace["memory_attention_seconds"],
                                           "query_key_pairs": sum(r["query_tokens"]*(r["selected_spatial_keys"]+r["pointer_keys"]) for r in trace["layers"]),
                                           "selected_spatial_keys": trace["allocation"]["selected_spatial_keys"] if trace["allocation"] else 0}
                        with (folder/"integrity.jsonl").open("a") as stream:
                            stream.write(json.dumps({**core, **metadata[index], "trace": trace})+"\n")
                        if index == initial or index % 10 == 0 or index == len(frames)-1:
                            (output/"progress.json").write_text(json.dumps({"state": phase, "sequence": sequence,
                                "condition": condition, "seed": seed, "frame": index, "last_frame": len(frames)-1,
                                "completed_runs": completed_runs}, indent=2)+"\n")
                        yield index, ids, logits

            author.iter_predictions = predictions
            rows, seconds = author.rollout(model, "sam2", frames, prompt, folder, "cpu", run)
            author.assert_parameter_integrity(model, parameters, folder, "C2_C3")
            if len(integrity) != len(frames)-prompt:
                raise RuntimeError("Incomplete engineering receipts")
            for row in rows:
                row.update(metadata[row["frame"]])
            result = {"sequence": sequence, "condition": condition, "seed": seed,
                      **propagation_summary(rows), "wall_seconds": seconds,
                      "iou": float(np.mean([r["scores"]["iou"] for r in rows if r["present"]])),
                      "boundary_f": float(np.mean([r["scores"]["boundary_f"] for r in rows if r["present"]])),
                      "memory_attention_seconds": sum(r["memory_attention_seconds"] for r in rows),
                      "query_key_pairs": sum(r["query_key_pairs"] for r in rows),
                      "mean_selected_spatial_keys": float(np.mean([r["selected_spatial_keys"] for r in rows])),
                      "max_selected_spatial_keys": max(r["selected_spatial_keys"] for r in rows),
                      "C2_reference_sha256": reference_sha}
            expected = baseline["results"]["C0"][sequence]
            if any(result[k] != expected[k] for k in ("targets", "present_targets", "gt_empty_targets")):
                raise RuntimeError("Scored cohort differs from completed C0")
            if reference_path and author.digest(reference_path) != reference_sha:
                raise RuntimeError("C2 allocation source changed during C3")
            (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
            packed_path = output/("_".join(folder.relative_to(output).parts)+".zip")
            with zipfile.ZipFile(packed_path, "w", zipfile.ZIP_DEFLATED) as packed:
                for path in folder.iterdir():
                    packed.write(path, arcname=path.name)
            log_results_artifact(run, output, [packed_path.name])
            completed_runs.append({"phase": phase, "sequence": sequence, "condition": condition,
                                   "seed": seed, "targets": len(rows)})
            print(f"[matched] completed phase={phase} sequence={sequence} condition={condition} seed={seed} targets={len(rows)}", flush=True)
            del model
            gc.collect()
            return result, integrity

        completed_gate = []
        for sequence in ("seq2", "seq3"):
            # Different selection must not reproduce v1's regional predictions.
            # Native/all-token E2 and fresh own-policy replay remain mandatory.
            old_e2 = None
            if args.policy == "legacy_v1":
                old_e2 = {r["frame"]: {k: r[k] for k in CORE_KEYS}
                          for r in map(json.loads, (args.e2_dir/sequence/"region_long_0/integrity.jsonl").read_text().splitlines())}
            folder = output/"gate"/sequence/"C2_first"
            _, first = rollout(sequence, "C2", None, folder, old_e2=old_e2)
            rollout(sequence, "C2", None, output/"gate"/sequence/"C2_repeat", replay=first)
            for seed in seeds:
                _, first_random = rollout(sequence, "C3", seed, output/"gate"/sequence/f"C3_{seed}_first", reference_path=folder/"integrity.jsonl")
                rollout(sequence, "C3", seed, output/"gate"/sequence/f"C3_{seed}_repeat", reference_path=folder/"integrity.jsonl", replay=first_random)
            completed_gate.append(sequence)
            (output/"progress.json").write_text(json.dumps({"state": "gate", "completed_sequences": completed_gate})+"\n")
        validation = {"passed": True, "sequences": completed_gate, "seeds": seeds,
                      "sources": source_hashes, "policy": args.policy,
                      "C2_matches_E2": True if args.policy == "legacy_v1" else None,
                      "C2_C3_counts_and_frames_match": True, "all_replays_bit_exact": True}
        (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        log_results_artifact(run, output, ["validation.json"])
        if args.gate_only:
            finalize_tracking(run, output, {"completed": True, "scope": "gate only", "gate_passed": True}, ["validation.json", "setup.json"])
            return
        if any(author.digest(ROOT/s) != sha for s, sha in source_hashes.items()):
            raise RuntimeError("Source changed during gate: no full inference")
        results = {"C2": {}, "C3": {str(seed): {} for seed in seeds}, "unpromptable": []}
        for sequence in sorted(videos, key=lambda s: int(s[3:])):
            try:
                prompt_index(videos[sequence], prompt_protocol)
            except ValueError:
                results["unpromptable"].append(sequence)
                continue
            folder = output/"full"/sequence/"C2"
            result, _ = rollout(sequence, "C2", None, folder)
            results["C2"][sequence] = result
            for seed in seeds:
                random_result, _ = rollout(sequence, "C3", seed, output/"full"/sequence/f"C3_{seed}", reference_path=folder/"integrity.jsonl")
                if any(random_result[k] != result[k] for k in ("query_key_pairs", "mean_selected_spatial_keys", "max_selected_spatial_keys")):
                    raise RuntimeError("Full C3 actual compute-count mismatch")
                results["C3"][str(seed)][sequence] = random_result
            (output/"progress.json").write_text(json.dumps({"state": "full", "results": results}, indent=2)+"\n")
        if len(results["C2"]) != 21 or sorted(results["unpromptable"]) != ["seq1", "seq7"]:
            raise RuntimeError("Full PolypGen cohort incomplete")
        summary = {"completed": True, "policy": args.policy, "scope": f"exploratory C2/C3 {args.policy}", "results": results,
                   **primary_statistics(results, baseline, seeds), "C4_launched": False, "E4_launched": False}
        (output/"results.json").write_text(json.dumps(summary, indent=2)+"\n")
        lines = [f"# C2 regional memory versus matched random C3 ({args.policy})", "", "Exploratory amendment after negative C1; sequence is the unit. C3 is the average of three fixed random seeds, conditional on C2 allocation. No training or tuning.", "", "| Sequence | C0 | C1 | C2 | C3 seed mean | C2−C0 | C2−C3 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for seq, r in results["C2"].items():
            c3 = float(np.mean([results["C3"][str(seed)][seq]["present_dice"] for seed in seeds]))
            c0, c1, c2 = baseline["results"]["C0"][seq]["present_dice"], baseline["results"]["C1"][seq]["present_dice"], r["present_dice"]
            lines.append(f"| {seq} | {c0:.5f} | {c1:.5f} | {c2:.5f} | {c3:.5f} | {c2-c0:+.5f} | {c2-c3:+.5f} |")
        lines += ["", "| Contrast | Mean difference | 95% sequence-bootstrap CI | Holm2 p |", "|---|---:|---|---:|"]
        for contrast in summary["contrasts"]:
            lines.append(f"| {contrast['contrast']} | {contrast['mean_difference']:+.5f} | {contrast['ci95']} | {contrast['holm2_p']:.6g} |")
        lines += ["", "All seed-level and sequence secondary metrics, false alarms, censored re-detection events and actual query/key work counts are in results.json and per-frame archives. Timing includes verification and is not isolated kernel speedup. No held-out confirmation; C4 and E4 have not run."]
        (output/"results.md").write_text("\n".join(lines)+"\n")
        finalize_tracking(run, output, {k: v for k, v in summary.items() if k != "results"}, ["results.json", "results.md", "validation.json", "setup.json"])
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"completed": False, "error": repr(error), "no_integrity_bypass": True}, indent=2)+"\n")
        run.summary.update({"completed": False, "failure": repr(error)})
        raise
    finally:
        author.iter_predictions = original
        run.finish()


if __name__ == "__main__":
    main()

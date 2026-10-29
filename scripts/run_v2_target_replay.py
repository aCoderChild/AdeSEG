"""V2-matched single-target diagnostic: engineering gate, then full cohort.

Uses the pinned Efficient-SAM2 gather/RoPE adapter and the existing native
SAM2 replay path. Selection and control orchestration are project logic.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.contiguous_exclusion import rigid_control
from modeling.contribution_region_memory import contribution_cells
from modeling.matched_region_memory import random_cells
from modeling.region_definitions import mixed_cell_regions
from modeling.sparse_memory import SparseMemoryRead
from scripts import run_author_baseline as author
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import prompt_index, validate_output
from scripts.run_contiguous_exclusions import infer, logits_exact, snapshot
from scripts.run_long_history_screen import require_e2
from scripts.run_region_exclusions import evaluate
from training.tracking import start_wandb, log_results_artifact, finalize_tracking

PROTOCOL = "protocols/v2_target_replay_v1.json"
SEEDS = (20261025, 20261026, 20261027)
SOURCES = [PROTOCOL, "scripts/run_v2_target_replay.py", "tests/test_v2_target_replay.py",
           "modeling/sparse_memory.py", "modeling/contribution_region_memory.py",
           "modeling/matched_region_memory.py", "modeling/contiguous_exclusion.py",
           "modeling/region_definitions.py", "modeling/region_tokens.py",
           "scripts/run_contiguous_exclusions.py", "scripts/run_region_exclusions.py",
           "scripts/bank_integrity.py", "scripts/run_author_baseline.py",
           "scripts/memory_token_io.py", "evaluation/statistics.py", "training/tracking.py"]


def selection(bank, sequence, target, condition, seed=None):
    """Keep the v2 dense endpoints; randomize only older spatial positions."""
    result, metadata = {}, []
    for position, (frame, item) in enumerate(bank.items()):
        regional = item["keep"].bool()
        info = {}
        if position in (0, len(bank)-1):
            regional = torch.ones(4096, dtype=torch.bool)
            keep, info = regional, {"protected_dense": True}
        elif condition == "regional":
            keep = regional
        elif condition == "random":
            keep = random_cells(int(regional.sum()), f"{sequence}|{target}", frame, seed)
        elif condition == "rigid":
            removed, info = rigid_control((~regional).reshape(64, 64), seed,
                                          f"v2|{sequence}|{target}|{frame}")
            keep = ~removed.flatten()
        else:
            raise ValueError("Unknown intervention")
        if keep.numel() != 4096 or int(keep.sum()) != int(regional.sum()):
            raise RuntimeError("Control changed per-frame token count")
        result[frame] = {**item, "keep": keep}
        metadata.append({"frame": frame, "kept": int(keep.sum()),
                         "keep_bits_hex": np.packbits(keep.numpy()).tobytes().hex(), **info})
    return result, metadata


def verify_layers(layers, frames, bank, reference=None):
    if [r["layer"] for r in layers] != list(range(4)):
        raise RuntimeError("Missing memory-attention layers")
    counts = [4096 if i in (0, len(frames)-1) else int(bank[f]["keep"].sum())
              for i, f in enumerate(frames)]
    for row in layers:
        if row["frames"] != frames or row["selected_spatial_keys"] != sum(counts):
            raise RuntimeError("Actual selection differs from native window / protected endpoints")
        if row["selected_spatial_keys"] > 28672:
            raise RuntimeError("Spatial budget exceeded")
    if reference is not None:
        for row, original in zip(layers, reference):
            if any(row[k] != original[k] for k in ("layer", "frames", "raw_spatial_keys",
                                                  "pointer_keys", "temporal_slots")):
                raise RuntimeError("Frame, pointer or temporal encoding changed")
    return counts


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")


@torch.inference_mode()
def sequence(args, repo, frames, name, folder, gate=False):
    folder.mkdir(parents=True)
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    reference = {r["frame"]: r for r in map(json.loads,
        (args.baseline_dir/"full"/name/"C0"/"integrity.jsonl").read_text().splitlines())}
    inputs = json.loads((args.assignment_dir/name/"inputs.json").read_text())
    if len(inputs) != len(frames):
        raise RuntimeError("Archived input coverage changed")
    for frame, receipt in zip(frames, inputs):
        if (frame.frame_index != receipt["frame"]
                or author.digest(frame.image_path) != receipt["image_sha256"]
                or author.digest(frame.mask_path) != receipt["mask_sha256"]):
            raise RuntimeError("Archived image/GT/chronology changed")
    (folder/"inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
    model = author.build_author_predictor(args, repo)
    parameters = author.parameter_fingerprints(model)
    bank, cases, checked, targets = {}, [], 0, 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frames_", dir=folder) as staged:
        stage_frames(frames, Path(staged))
        state = model.init_state(video_path=staged, offload_video_to_cpu=True,
                                 offload_state_to_cpu=False)
        initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
        model.add_new_mask(state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial))
        for index, ids, baseline in model.propagate_in_video(state, start_frame_idx=prompt):
            if ids != [1] or index != prompt+len(bank):
                raise RuntimeError("Object identity or chronology changed")
            own = state["output_dict_per_obj"][0]
            stored = own["cond_frame_outputs"].get(index, own["non_cond_frame_outputs"].get(index))
            if author.tensor_hash(baseline) != reference[index]["prediction_sha256"]:
                raise RuntimeError(f"Native archived prediction mismatch: {name}/{index}")
            if author.tensor_hash(stored["pred_masks"]) != reference[index]["low_res_sha256"]:
                raise RuntimeError("Native stored mask changed")
            if index > prompt:
                targets += 1
                memory_frames = [prompt]+list(range(max(prompt+1, index-6), index))
                active = {f: bank[f] for f in memory_frames}
                units = [{"frame": f} for f in memory_frames]
                def signature():
                    return fingerprint(bank_contents(state, index, units, model.max_obj_ptrs_in_encoder))["sha256"]
                before = signature()
                cached = snapshot(state, index)

                def replay(mode, selected):
                    controller = SparseMemoryRead(model, prompt, mode)
                    controller.frame, controller.bank = index, selected
                    with controller:
                        result = infer(model, cached, index)
                    layers = controller.layers.copy()
                    if signature() != before:
                        raise RuntimeError("Replay mutated native bank/image cache")
                    return result, layers

                # Every target, including nonintervention gate frames, checks the
                # exact all-token gather path against unmodified native inference.
                restored, native_layers = replay("all_native", active)
                dense = {f: {**v, "keep": torch.ones(4096, dtype=torch.bool)} for f, v in active.items()}
                verify_layers(native_layers, memory_frames, dense)
                if not logits_exact(restored, baseline):
                    raise RuntimeError(f"All-token native identity failed: {name}/{index}")
                selected_target = not gate or index in args.protocol["gate_targets"][name]
                if selected_target:
                    truth = load_label_mask(frames[index].mask_path, [1]) == 1
                    base_scores = evaluate(baseline, truth)
                    Image.fromarray((baseline[0, 0].numpy()>0).astype(np.uint8)*255).save(folder/f"{index:06d}_native.png")
                    reference_layers = None
                    for condition, seed in [("regional", None)]+[(c, s) for c in ("random", "rigid") for s in SEEDS]:
                        selected, metadata = selection(active, name, index, condition, seed)
                        result, layers = replay("region_native", selected)
                        counts = verify_layers(layers, memory_frames, selected, native_layers)
                        if reference_layers is None:
                            reference_layers = counts
                        elif counts != reference_layers:
                            raise RuntimeError("Controls not count/frame matched")
                        if gate:
                            repeated, repeated_layers = replay("region_native", selected)
                            if not logits_exact(result, repeated) or layers != repeated_layers:
                                raise RuntimeError("Counterfactual repeat mismatch")
                        suffix = condition if seed is None else f"{condition}_{seed}"
                        Image.fromarray((result[0, 0].numpy()>0).astype(np.uint8)*255).save(folder/f"{index:06d}_{suffix}.png")
                        row = {"sequence": name, "frame": index, "condition": condition, "seed": seed,
                               "present": bool(truth.any()), "gt_area_fraction": float(truth.mean()),
                               "baseline": base_scores, "scores": evaluate(result, truth),
                               "prediction_sha256": author.tensor_hash(result), "layers": layers,
                               "per_frame_counts": counts, "selection": metadata,
                               "control_informative": any(m.get("informative", False) for m in metadata[1:-1]),
                               "absent_false_alarm": bool((result>0).any()) if not truth.any() else None}
                        append(folder/"cases.jsonl", row)
                        cases.append(row)
                    restored, layers = replay("all_native", active)
                    if not logits_exact(restored, baseline) or layers != native_layers:
                        raise RuntimeError("Post-intervention restoration mismatch")
                    checked += 1
                append(folder/"integrity.jsonl", {"frame": index, "native_exact": True,
                       "all_token_exact": True, "restoration_exact": True,
                       "no_writeback": True, "bank_cache_sha256": before,
                       "prediction_sha256": author.tensor_hash(baseline), "layers": native_layers})
                print(f"[{name}] frame={index} interventions={checked} gate={gate}", flush=True)
            labels = mixed_cell_regions(stored["pred_masks"][0, 0])["labels"]
            bank[index] = {"features": stored["maskmem_features"].detach(),
                           "pos": stored["maskmem_pos_enc"][-1].detach(),
                           "keep": contribution_cells(labels)}
        model.reset_state(state)
    if set(reference) != set(range(prompt, len(frames))) or targets != len(frames)-prompt-1:
        raise RuntimeError("Incomplete native target coverage")
    if checked != (len(args.protocol["gate_targets"][name]) if gate else targets):
        raise RuntimeError("Incomplete intervention coverage")
    author.assert_parameter_integrity(model, parameters, folder, "v2_replay")
    summary = {"sequence": name, "targets": targets, "intervention_targets": checked,
               "wall_seconds": time.monotonic()-started, "passed": True}
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    del model
    gc.collect()
    return summary, cases


def summarize(cases):
    """Sequence units; control seeds averaged inside each target first."""
    by_target = {}
    for row in cases:
        by_target.setdefault((row["sequence"], row["frame"]), []).append(row)
    contrasts = {"regional_minus_native": {}, "regional_minus_random": {}, "regional_minus_rigid": {}}
    for (name, frame), rows in by_target.items():
        regional = next(r for r in rows if r["condition"] == "regional")
        if not regional["present"]:
            continue
        value = regional["scores"]["dice"]
        for condition in ("native", "random", "rigid"):
            controls = [r for r in rows if r["condition"] == condition]
            if condition == "rigid" and not any(r["control_informative"] for r in controls):
                continue
            reference = (regional["baseline"]["dice"] if condition == "native" else
                         float(np.mean([r["scores"]["dice"] for r in controls])))
            contrasts[f"regional_minus_{condition}"].setdefault(name, []).append(value-reference)
    results = {}
    for key, values in contrasts.items():
        means = {s: float(np.mean(v)) for s, v in values.items()}
        if not means:
            results[key] = {"n_sequences": 0, "undefined": True}
            continue
        results[key] = {"n_sequences": len(means), "sequence_differences": means,
                        "mean_difference": float(np.mean(list(means.values()))),
                        "ci95": sequence_bootstrap_interval(means, seed=20261030),
                        "wilcoxon_p": paired_wilcoxon(means, {s: 0. for s in means})["p"]}
    adjusted = holm_adjust({k: v["wilcoxon_p"] for k, v in results.items() if "wilcoxon_p" in v})
    for key, value in adjusted.items():
        results[key]["holm3_p"] = value
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "e2-dir", "baseline-dir", "assignment-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    gate = require_e2(args.e2_dir)
    if author.digest(args.checkpoint) != gate["setup"]["checkpoint_sha256"] or author.digest(args.manifest) != gate["setup"]["manifest_sha256"]:
        raise ValueError("Checkpoint/manifest differs from verified E2")
    old = json.loads((args.baseline_dir/"setup.json").read_text())
    for name, sha in old["sources"].items():
        if author.digest(ROOT/name) != sha:
            raise ValueError(f"Archived native source changed: {name}")
    if old["E2_setup"]["checkpoint_sha256"] != gate["setup"]["checkpoint_sha256"] or old["E2_setup"]["manifest_sha256"] != gate["setup"]["manifest_sha256"]:
        raise ValueError("Archived baseline inputs differ")
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    args.protocol = json.loads((ROOT/PROTOCOL).read_text())
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR", "TMPDIR"):
        os.environ[key] = str(output/key.lower())
        Path(os.environ[key]).mkdir()
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    sys.path.insert(0, str(repo))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = {n: author.digest(ROOT/n) for n in set(SOURCES) | set(gate["setup"]["sources"]) | set(old["sources"])}
    setup = {"protocol": args.protocol, "sources": sources, "author_revision": revision,
             "checkpoint_sha256": author.digest(args.checkpoint), "manifest_sha256": author.digest(args.manifest),
             "baseline_setup_sha256": author.digest(args.baseline_dir/"setup.json"),
             "torch": torch.__version__, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    with zipfile.ZipFile(output/"sources.zip", "w", zipfile.ZIP_DEFLATED) as packed:
        for name in sources:
            packed.write(ROOT/name, arcname=name)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "28-v2 matched target replay", "online",
                     "https://forge.coreweave.com/api/wandb", setup, output, ["v2", "one-target", "exploratory"])
    try:
        videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
        gates = []
        for name in ("seq2", "seq3"):
            (output/"progress.json").write_text(json.dumps({"state": "gate_running", "sequence": name})+"\n")
            summary, _ = sequence(args, repo, videos[name], name, output/"gate"/name, gate=True)
            gates.append(summary)
        validation = {"passed": True, "sequences": gates, "sources": sources,
                      "all_token_native_bit_exact": True, "counterfactual_repeat_bit_exact": True,
                      "bank_cache_restored": True, "no_writeback": True, "four_layers": True}
        (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        with zipfile.ZipFile(output/"gate.zip", "w", zipfile.ZIP_DEFLATED) as packed:
            for path in (output/"gate").rglob("*"):
                if path.is_file():
                    packed.write(path, arcname=path.relative_to(output))
        log_results_artifact(run, output, ["setup.json", "sources.zip", "validation.json", "gate.zip"])
        if not args.gate_only:
            summaries, cases = [], []
            for name in sorted(videos, key=lambda s: int(s[3:])):
                if name in ("seq1", "seq7"):
                    continue
                if any(author.digest(ROOT/n) != sha for n, sha in sources.items()):
                    raise RuntimeError("Frozen implementation changed during execution")
                summary, rows = sequence(args, repo, videos[name], name, output/"full"/name)
                summaries.append(summary)
                cases.extend(rows)
                with zipfile.ZipFile(output/f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as packed:
                    for path in (output/"full"/name).iterdir():
                        packed.write(path, arcname=path.name)
                log_results_artifact(run, output, [f"{name}.zip"])
                (output/"progress.json").write_text(json.dumps({"state": "running", "completed": summaries}, indent=2)+"\n")
            if len(summaries) != 21 or sum(r["targets"] for r in summaries) != 2012:
                raise RuntimeError("Full cohort coverage differs from frozen protocol")
            result = {"completed": True, "sequences": summaries, "statistics": summarize(cases)}
            (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
            lines = ["# V2 single-target diagnostic", "", "Exploratory; native-bank branches without writeback.", "",
                     "| Contrast (regional minus reference) | Sequences | Dice difference | 95% CI | Holm3 p |",
                     "|---|---:|---:|---|---:|"]
            for name, row in result["statistics"].items():
                if not row.get("undefined"):
                    lines.append(f"| {name} | {row['n_sequences']} | {row['mean_difference']:.6f} | {row['ci95']} | {row['holm3_p']:.6f} |")
            (output/"results.md").write_text("\n".join(lines)+"\n")
            log_results_artifact(run, output, ["results.json", "results.md"])
        finalize_tracking(run, output, {"gate_passed": True, "gate_only": args.gate_only},
                          ["validation.json"] if args.gate_only else ["results.json", "results.md"])
        run.finish()
    except BaseException as error:
        (output/"failure.json").write_text(json.dumps({"error": repr(error), "time": time.time()}, indent=2)+"\n")
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

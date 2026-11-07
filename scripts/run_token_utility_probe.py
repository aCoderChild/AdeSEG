"""Run173: provenance-checked offline utility prediction, without SAM2 inference."""
import argparse
import gc
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import sklearn
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.token_utility_probe import (
    ARMS, MODES, block_indices, layer_features, nested_predictions, summarize,
    Ridge, StandardScaler, Pipeline, GroupKFold,
)
from modeling.complementary_memory.integration import _tensor_sha256
from scripts.diagnose_complementary_memory import digest, atomic_json, timestamp, drive_root, _bank_outputs

PROTOCOL = ROOT / "protocols/token_utility_probe_v1.json"
SEQUENCES = [f"seq{i}" for i in range(2, 24) if i != 7]
SOURCES = ["evaluation/token_utility_probe.py", "scripts/run_token_utility_probe.py",
           "protocols/token_utility_probe_v1.json", "tests/test_token_utility_probe.py"]
DEPENDENCIES = ["scripts/diagnose_complementary_memory.py", "evaluation/statistics.py",
                "evaluation/segmentation.py", "datasets/io.py", "datasets/video.py",
                "modeling/complementary_memory/integration.py"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_json(path, expected):
    require(digest(path) == expected, f"Hash mismatch: {path}")
    return read_json(path)


def intervention_features(native, arm, outputs):
    """Use past memory and a shared native first pass; never treatment outputs."""
    frame = native["frame"]
    nread, read = native["read"], arm["read"]
    require(frame == arm["frame"], "Target mismatch")
    for key in ("memory_frames", "pointer_keys", "pointer_input_sha256", "pointer_position_sha256"):
        require(read[key] == nread[key], f"Altered native layout/pointers: {key}")
    for record in (read, nread):
        require(record["layers"] == [0, 1, 2, 3], "Missing native attention layer")
        require(all(record[k] is True for k in ("stored_mask_alignment", "native_positions_verified", "bank_read_unchanged")),
                "Native integrity receipt failed")
    ids = [s["frame"] for s in nread["memory_frames"]]
    require(len(ids) >= 3 and len(set(ids)) == len(ids) and max(ids) < frame and ids[-1] == frame-1,
            "Noncausal/invalid memory layout")
    latest = outputs[ids[-1]]["maskmem_features"][0].float().flatten(1).T.mean(0).numpy()
    # Current native score/probability are shared first-pass descriptors; no current memory vectors.
    require(float(outputs[frame]["object_score_logits"].item()) == native["object_score_logit"],
            "Archived native score mismatch")
    shared = [native["object_score_logit"], float(outputs[frame]["pred_masks"].sigmoid().mean()), len(ids)-2]
    generic, geometry, regional = [], [], []
    require(len(read["selection_layers"]) == 4, "Wrong intervention layer count")
    for layer, record in enumerate(read["selection_layers"]):
        require(record["layer"] == layer and record["frame_ids"] == ids, "Misordered intervention")
        require(record["counts"] == [4096] + [3840]*(len(ids)-2) + [4096], "Unmatched removal budget")
        require(record["spatial_keys_before"] == 4096*len(ids)
                and record["spatial_keys_after"] == sum(record["counts"])
                and record["pointer_keys"] == read["pointer_keys"], "Invalid actual key counts")
        details = record["intervention_groups"]
        require([d["memory_frame"] for d in details] == ids[1:-1], "Protected/future frame intervention")
        fields = {k: [] for k in ("vectors", "occupancy", "mixture", "indices", "ages", "usage")}
        for detail in details:
            indices = block_indices(detail["block_ids"])
            removed = torch.zeros(4096, dtype=torch.bool)
            removed[indices] = True
            require(detail["removed"] == 256 and _tensor_sha256(removed) == detail["mask_sha256"],
                    "Block reconstruction differs from actual intervention")
            position = ids.index(detail["memory_frame"])
            require(_tensor_sha256(torch.where(~removed)[0]) == record["selected_indices_sha256"][position],
                    "Retained indices mismatch")
            output = outputs[detail["memory_frame"]]
            vectors = output["maskmem_features"][0].float().flatten(1).T
            prob = output["pred_masks"].float().sigmoid()
            require(tuple(prob.shape) == (1, 1, 256, 256) and tuple(vectors.shape) == (4096,64),
                    "Unsupported native mask/vector shape")
            occupancy = F.avg_pool2d(prob, 4).flatten()
            mixture = (4*(F.avg_pool2d(prob.square(), 4).flatten()-occupancy.square())).clamp_min(0)
            # Match the original float32 boolean-index reductions, before casting for sklearn.
            for observed, expected in ((float(occupancy[removed].mean()), detail["mean_soft_foreground"]),
                                       (float(vectors[removed].norm(dim=-1).mean()), detail["mean_vector_norm"])):
                require(np.isclose(observed, expected, atol=2e-6, rtol=1e-6),
                        f"Archived bank differs from intervention descriptors: {observed} vs {expected}")
            fields["vectors"].append(vectors[indices].numpy())
            fields["occupancy"].append(occupancy[indices].numpy())
            fields["mixture"].append(mixture[indices].numpy())
            fields["indices"].append(indices)
            fields["ages"].append(np.full(256, frame-detail["memory_frame"]))
            fields["usage"].append(np.full(256, detail["first_read_usage_mass"]))
        g, c, r = layer_features(**{k:np.concatenate(v) for k,v in fields.items()}, latest_mean=latest)
        generic.append(g); geometry.append(c); regional.append(r)
    base = np.r_[np.concatenate(generic), shared]
    return {"features":base, "geometry":np.r_[base,np.concatenate(geometry)],
            "regional":np.r_[base,np.concatenate(regional)]}


def collect(drive, output, protocol):
    source = drive / "outputs/experiment/171_token_contribution_v1/attempt_02"
    archive = drive / "outputs/experiment/143_official_native_feature_audit_v1/full"
    pinned = protocol["input_receipts"]
    for relative, sha in pinned.items():
        require(digest(drive / "outputs/experiment" / relative) == sha, f"Input receipt changed: {relative}")
    complete = read_json(source / "completion.json")
    require(complete["complete"], "Run171 incomplete")
    setup = checked_json(source / "setup.json", complete["setup_sha256"])
    checked_json(source / "results.json", complete["results_sha256"])
    bank_complete = read_json(archive / "completion.json")
    audit = read_json(archive / "independent_audit.json")
    require(bank_complete["passed"] and audit["passed"], "Native archive not verified")
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    require(digest(manifest) == setup["protocol"]["manifest_sha256"], "Manifest changed")
    videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
    rows, matrices, outcomes, receipts = [], {m:[] for m in MODES}, [], {}
    absent = 0
    for name in SEQUENCES:
        folder = source / "sequences" / name
        seq_done = read_json(folder / "completion.json")
        require(seq_done["complete"], f"Incomplete {name}")
        records = {}
        for arm, sha in seq_done["frames_sha256"].items():
            path = folder / arm / "frames.jsonl"
            require(digest(path) == sha, f"Changed arm records: {name}/{arm}")
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            require(len({r['frame'] for r in entries}) == len(entries), "Duplicate target")
            records[arm] = {r["frame"]:r for r in entries}
            require(sorted(records[arm]) == setup["target_plan"][name]["targets"], "Target plan mismatch")
        bank_folder = Path(bank_complete["entries"][name])
        require(bank_folder.resolve().is_relative_to(drive.resolve()), "Archive outside Drive")
        receipt = checked_json(bank_folder / "receipt.json", audit["sequences"][name]["receipt_sha256"])
        for key in ("checkpoint_sha256", "manifest_sha256"):
            require(receipt[key] == setup["protocol"][key], f"Native lineage mismatch: {key}")
        require(receipt["prompt"] == setup["target_plan"][name]["prompt"], "Prompt differs")
        bank_path = bank_folder / "bank.pt"
        require(digest(bank_path) == receipt["bank_sha256"], "Native bank changed")
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        outputs = _bank_outputs(bank)
        prediction_hashes = {p["frame"]:p["sha256"] for p in receipt["predictions"]}
        input_frames = {r["local_frame"]:r for r in setup["inputs"][name]["frames"]}
        for frame, native in records["native"].items():
            require(native["prediction_sha256"] == prediction_hashes[frame], "Native trajectory changed")
            truth_path = videos[name][frame].mask_path
            require(digest(truth_path) == input_frames[frame]["mask_sha256"], "GT mask changed")
            truth = load_label_mask(truth_path, [1]) == 1
            present = bool(truth.any())
            for arm in ("native", *ARMS):
                row = records[arm][frame]
                require(row["present"] == present and row["source_frame_index"] == videos[name][frame].frame_index,
                        "GT/frame alignment changed")
                mask_path = folder / arm / "masks" / f"{frame:06d}.png"
                require(digest(mask_path) == row["mask_sha256"], "Prediction PNG changed")
                prediction = np.array(Image.open(mask_path)) > 0
                require(prediction.shape == truth.shape, "Mask geometry mismatch")
                dice = calculate_scores(prediction, truth)["dice"]
                require(abs(dice-row["scores"]["dice"]) < 1e-12, "Dice reconstruction mismatch")
            for arm in ARMS:
                features = intervention_features(native, records[arm][frame], outputs)
                if not present:
                    absent += 1
                    continue
                rows.append({"sequence":name, "frame":frame, "arm":arm})
                outcomes.append(native["scores"]["dice"]-records[arm][frame]["scores"]["dice"])
                for mode in MODES:
                    matrices[mode].append(features[mode])
        receipts[name] = {"targets":len(records["native"]), "bank_sha256":receipt["bank_sha256"],
                          "receipt_sha256":digest(bank_folder / "receipt.json"),
                          "run171_completion_sha256":digest(folder / "completion.json")}
        atomic_json(output / "progress.json", {"stage":"archive_validation", "sequences":len(receipts),
                                               "total":21, "updated":timestamp()})
        print(f"Verified {name}: {len(records['native'])} targets", flush=True)
        del outputs, bank
        gc.collect()
    require(len(rows) == 252 and absent == 90, "Cohort coverage differs from frozen protocol")
    matrices = {m:np.stack(v) for m,v in matrices.items()}
    require({m:x.shape[1] for m,x in matrices.items()} == protocol["feature_dimensions"], "Feature layout changed")
    atomic_json(output / "engineering_pass.json", {"passed":True, "rows":len(rows),
        "absent_rows_validated_not_fitted":absent, "receipts":receipts,
        "checks":["Native checkpoint/manifest/prompt and target prediction hashes match",
                  "Archived bank hashes match independent audit receipts",
                  "Every reconstructed block and retained-index hash matches",
                  "Four layers, causal frame IDs, equal budgets and unchanged pointers",
                  "Archived occupancy/vector norms match recorded intervention descriptors",
                  "Saved masks reproduce Dice exactly; target and GT hashes match"],
        "limit":"Run171 did not save full vector tensor hashes; identity is supported by native trajectory lineage and matching group descriptors, not an independent elementwise comparison."})
    return rows, np.asarray(outcomes), matrices


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    drive = drive_root(output)
    require(output.parent.name == "173_token_utility_probe_v1" and output.name.startswith("attempt_"), "Wrong run directory")
    require(not output.exists(), "Never overwrite an existing attempt")
    protocol = read_json(PROTOCOL)
    head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    for relative in SOURCES:
        committed = subprocess.check_output(["git","show",f"{head}:{relative}"], cwd=ROOT)
        require(committed == (ROOT / relative).read_bytes(), f"Uncommitted experiment source: {relative}")
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        snapshot = output / "sources"
        hashes = {}
        for relative in SOURCES + DEPENDENCIES:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            hashes[relative] = digest(target)
        for cls in (Ridge, StandardScaler, Pipeline, GroupKFold):
            path = Path(inspect.getfile(cls))
            target = snapshot / f"sklearn_{cls.__name__}.py"
            shutil.copy2(path, target)
            hashes[target.name] = digest(target)
        atomic_json(output / "setup.json", {"commit":head, "created":timestamp(), "protocol":protocol,
            "source_sha256":hashes, "sklearn":sklearn.__version__, "torch":torch.__version__,
            "tracking":"Drive JSON progress, fold models, OOF predictions and receipts; no W&B dependency"})
        rows, y, matrices = collect(drive, output, protocol)
        atomic_json(output / "rows.json", rows)
        np.savez_compressed(output / "features.npz", y=y, **matrices)
        groups = np.array([r["sequence"] for r in rows])
        counter = 0
        def on_fold(record, test, predictions):
            nonlocal counter
            counter += 1
            record["rows"] = test.tolist()
            record["predictions"] = {m:p.tolist() for m,p in predictions.items()}
            atomic_json(output / "folds" / f"{record['test_sequence']}.json", record)
            atomic_json(output / "progress.json", {"stage":"nested_validation", "folds":counter,
                "total":21, "updated":timestamp()})
            print(f"Fold {counter}/21 complete: {record['test_sequence']}", flush=True)
        with threadpool_limits(limits=2):
            predictions = nested_predictions(matrices, y, groups, alphas=protocol["alphas"],
                                            inner_folds=protocol["inner_folds"], on_fold=on_fold)
        atomic_json(output / "predictions.json", {m:p.tolist() for m,p in predictions.items()})
        results = summarize(rows, y, predictions, protocol["seed"])
        atomic_json(output / "results.json", results)
        lines = ["# Run173: offline utility probe", "", results["scope"], "",
                 "| Probe | MSE | Pair concordance | Offline chosen − native Dice |",
                 "|---|---:|---:|---:|"]
        for mode, metrics in results["means"].items():
            lines.append(f"| {mode} | {metrics['mse']:.8f} | {metrics['pair_concordance']} | {metrics['chosen_minus_native_dice']:+.6f} |")
        lines += ["", f"Continuation criteria passed: **{results['continuation_passed']}**", "",
                  "These are held-sequence predictions of archived joint removals, not a new segmentation run.",
                  results["uncertainty_limit"]]
        (output / "results.md").write_text("\n".join(lines)+"\n")
        for relative in SOURCES + DEPENDENCIES:
            require(digest(ROOT / relative) == hashes[relative], f"Source changed during run: {relative}")
        atomic_json(output / "completion.json", {"complete":True, "finished":timestamp(),
            "elapsed_seconds":time.monotonic()-started, "continuation_passed":results["continuation_passed"],
            "results_sha256":digest(output / "results.json"), "features_sha256":digest(output / "features.npz"),
            "predictions_sha256":digest(output / "predictions.json"), "setup_sha256":digest(output / "setup.json")})
        atomic_json(output / "progress.json", {"stage":"complete", "updated":timestamp(),
            "continuation_passed":results["continuation_passed"]})
        print(f"Complete. Continuation criteria passed: {results['continuation_passed']}", flush=True)
    except Exception as error:
        atomic_json(output / "failure.json", {"error":repr(error), "time":timestamp()})
        raise


if __name__ == "__main__":
    main()

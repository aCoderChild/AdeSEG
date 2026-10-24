"""No-inference feasibility audit of archived native predictions via W&B.

Downloads small, preserved author-baseline archives to Drive. No model is built.
The legacy masks do not identify exact score<=0 triggers; report this explicitly.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.false_alarm_statistics import native_opportunities, prediction_nonempty  # noqa: E402
from scripts.memory_token_io import validate_output  # noqa: E402

RUNS = {"sam2": "az54ik1q", "samurai": "chvr9c4l"}


def verify_archive_masks(archive, sequence, rows):
    for row in rows:
        name = f"{sequence}/frame_{row['frame']:06d}.png"
        with Image.open(io.BytesIO(archive.read(name))) as mask:
            if mask.mode != "L":
                raise ValueError("Expected archived binary prediction PNG")
            nonempty = mask.getbbox() is not None
        if nonempty != prediction_nonempty(row):
            raise RuntimeError("Saved metric-derived occupancy disagrees with prediction PNG")
    return len(rows)


def verify_cached_masks(output):
    output = Path(output)
    source = output / "opportunities.json"
    report = json.loads(source.read_text())
    counts = {}
    for method, result in report["methods"].items():
        checked = 0
        for entry in result["archive_provenance"]:
            path = output / "remote_sources" / method / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Cached native archive changed")
            sequence = entry["sequence"]
            with zipfile.ZipFile(path) as archive:
                rows = [json.loads(line) for line in archive.read(f"{sequence}/frames.jsonl").splitlines()]
                checked += verify_archive_masks(archive, sequence, rows)
                if native_opportunities(rows) != result["sequences"][sequence]:
                    raise RuntimeError("Opportunity aggregation differs on independent read")
        counts[method] = checked
    receipt = {"inference": False, "png_occupancy_matches_saved_metrics": True,
               "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "checked_targets": counts}
    (output / "mask_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def audit(output):
    output = validate_output(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_CACHE_DIR"] = str(output / "wandb_cache")
    os.environ["WANDB_DATA_DIR"] = str(output / "wandb_data")
    import wandb

    api = wandb.Api(overrides={"base_url": "https://forge.coreweave.com/api/wandb"}, timeout=60)
    result = {"inference": False, "exploratory": True, "methods": {},
              "limits": "Empty prediction is necessary but not sufficient for score<=0. Counts describe the native baseline, not a strict bound on all future counterfactual effects."}
    for method, run_id in RUNS.items():
        run = api.run(f"adenoid-hypertrophy/ade-seg/{run_id}")
        if run.state != "finished" or not run.summary.get("completed"):
            raise ValueError("Native baseline is not a completed run")
        sequences, provenance = {}, []
        for artifact in run.logged_artifacts():
            for entry in artifact.files():
                if not entry.name.startswith("seq") or not entry.name.endswith(".zip"):
                    continue
                manifest_entry = artifact.get_entry(entry.name)
                path = Path(manifest_entry.download(
                    root=str(output / "remote_sources" / method), skip_cache=True
                ))
                payload = path.read_bytes()
                if base64.b64encode(hashlib.md5(payload).digest()).decode() != manifest_entry.digest:
                    raise RuntimeError("Native archive digest mismatch")
                sequence = Path(entry.name).stem
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError("Native archive CRC failure")
                    rows = [json.loads(line) for line in archive.read(f"{sequence}/frames.jsonl").splitlines()]
                    saved = json.loads(archive.read(f"{sequence}/summary.json"))
                if len(rows) != saved["frames"] or sequence in sequences:
                    raise RuntimeError("Native sequence coverage/identity mismatch")
                sequences[sequence] = native_opportunities(rows)
                provenance.append({"sequence": sequence, "artifact": artifact.qualified_name,
                                   "file": entry.name, "md5": manifest_entry.digest,
                                   "sha256": hashlib.sha256(payload).hexdigest()})
                print(method, sequence, sequences[sequence], flush=True)
        if set(sequences) != {f"seq{i}" for i in range(2, 24) if i != 7}:
            raise RuntimeError("Native cohort is incomplete")
        totals = {key: sum(row[key] for row in sequences.values())
                  for key in next(iter(sequences.values()))}
        if totals["targets"] != 2012 or totals["gt_empty"] != 323:
            raise RuntimeError("Unexpected native target/absence coverage")
        result["methods"][method] = {"run_id": run_id, "sequences": sequences,
                                     "totals": totals, "archive_provenance": provenance,
                                     "exact_object_score_trigger_count": None}
    (output / "opportunities.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Relapse-prevention feasibility — no inference", "",
             "Archived native author predictions; 21 promptable sequences, 2,012 targets and 323 GT-empty targets per method. Initial prompt is nonempty by construction. Empty prediction is a proxy, NOT proof of score<=0; no exact trigger count is claimed.", "",
             "| Method | GT-empty | After empty prediction | Of these after correct empty | After missed present | Native FP after empty prediction | Native FP after nonempty prediction | Present targets after empty prediction |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, record in result["methods"].items():
        r = record["totals"]
        lines.append(f"| {method} | {r['gt_empty']} | {r['gt_empty_after_empty_prediction']} | {r['after_correct_empty']} | {r['after_missed_present']} | {r['false_alarms_after_empty_prediction']} | {r['false_alarms_after_nonempty_prediction']} | {r['present_after_empty_prediction']} |")
    lines += ["", "These are baseline opportunities for relapse prevention, not a test of clearing persistent false alarms. Counterfactual propagation can change later eligibility, so these counts are not a global upper bound on modified outputs. Previous empty-mask proxies may overcount exact nonpositive-score triggers. The exact score-defined risk set must be recorded by fresh native runs if execution is later authorized.", "",
              "All archive MD5 and ZIP CRC checks passed. Per-sequence counts and artifact SHA256 provenance are in opportunities.json. No masks/scores were re-inferred."]
    (output / "opportunities.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({m: r["totals"] for m, r in result["methods"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-cached-masks", action="store_true",
                        help="Check existing downloaded PNGs; no network or inference")
    args = parser.parse_args()
    if args.verify_cached_masks:
        verify_cached_masks(args.output_dir)
    else:
        audit(args.output_dir)

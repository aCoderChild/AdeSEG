"""CPU FP32 E2 native/sparse/long-read engineering gate; no E3 efficacy run."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest  # noqa: E402
from modeling.sparse_memory import SparseMemoryRead  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.bank_integrity import fingerprint  # noqa: E402
from scripts.memory_token_io import prompt_index, validate_output  # noqa: E402
from training.tracking import start_wandb, finalize_tracking  # noqa: E402

SOURCES = ["protocols/sparse_memory_E2_v1.json", "modeling/sparse_memory.py",
           "scripts/run_sparse_memory_gate.py", "modeling/region_definitions.py",
           "scripts/bank_integrity.py", "scripts/region_sparse_statistics.py",
           "scripts/run_author_baseline.py", "modeling/region_tokens.py",
           "scripts/memory_token_io.py", "training/tracking.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    args = parser.parse_args()
    protocol = json.loads((ROOT/SOURCES[0]).read_text())
    if author.digest(args.checkpoint) != protocol["checkpoint_sha256"] or author.digest(args.manifest) != protocol["manifest_sha256"]:
        raise ValueError("E2 checkpoint/manifest identity differs")
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        os.environ[key] = str(output/key.lower())
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    setup = {"protocol": protocol, "sources": {s: author.digest(ROOT/s) for s in SOURCES},
             "author_revision": revision, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "checkpoint_sha256": author.digest(args.checkpoint), "manifest_sha256": author.digest(args.manifest)}
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "22-E2 sparse memory correctness gate - seq2 seq3", "online",
                      "https://forge.coreweave.com/api/wandb", setup, output, ["E2", "engineering-gate", "cpu-fp32"])
    original = author.iter_predictions
    try:
        sys.path.insert(0, str(repo))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
        prompt_protocol = json.loads((ROOT/"protocols/memory_token_regions.json").read_text())
        completed, older_reads = [], 0
        for sequence in protocol["sequences"]:
            frames = videos[sequence]
            prompt = prompt_index(frames, prompt_protocol)
            reference, first_sparse = {}, {}
            for mode in protocol["conditions"]:
                repetitions = 2 if mode.startswith("region") else 1
                for repeat in range(repetitions):
                    model = author.build_author_predictor(args, repo)
                    before = author.parameter_fingerprints(model)
                    folder = output/sequence/f"{mode}_{repeat}"
                    folder.mkdir(parents=True)
                    receipts = []

                    def predictions(predictor, state, method, initial):
                        nonlocal older_reads
                        controller = SparseMemoryRead(predictor, initial, mode) if mode != "native" else None
                        with controller if controller else nullcontext():
                            for index, ids, logits in original(predictor, state, method, initial):
                                trace = controller.capture(state, index) if controller else None
                                stored = state["output_dict_per_obj"][0]
                                own = stored["cond_frame_outputs"].get(index, stored["non_cond_frame_outputs"].get(index))
                                record = {"frame": index, "prediction_sha256": author.tensor_hash(logits),
                                          "low_res_sha256": author.tensor_hash(own["pred_masks"]),
                                          "bank_sha256": fingerprint(state["output_dict_per_obj"])["sha256"],
                                          "cache_sha256": fingerprint(state["cached_features"])["sha256"]}
                                if mode == "native":
                                    reference[index] = record
                                elif mode.startswith("all") and reference.get(index) != record:
                                    changed = [k for k in record if record[k] != reference[index][k]]
                                    raise RuntimeError(f"Native all-token parity failed: {sequence} {index} {mode} {changed}")
                                elif mode.startswith("region"):
                                    key = (mode, index)
                                    if repeat == 0:
                                        first_sparse[key] = record
                                    elif first_sparse.get(key) != record:
                                        raise RuntimeError(f"Fresh sparse replay differs: {sequence} {index} {mode}")
                                receipts.append(record)
                                with (folder/"integrity.jsonl").open("a") as stream:
                                    stream.write(json.dumps({**record, "trace": trace})+"\n")
                                if trace and mode == "region_long" and index > initial:
                                    older_reads += any(index-f > 6 and f != initial for f in trace["layers"][0]["frames"])
                                yield index, ids, logits

                    author.iter_predictions = predictions
                    rows, seconds = author.rollout(model, "sam2", frames, prompt, folder, "cpu", run)
                    author.assert_parameter_integrity(model, before, folder, "gate")
                    if len(receipts) != len(frames)-prompt:
                        raise RuntimeError("Missing gate integrity receipts")
                    completed.append({"sequence": sequence, "mode": mode, "repeat": repeat,
                                      "targets": len(rows), "seconds": seconds})
                    (output/"progress.json").write_text(json.dumps({"state": "running", "completed": completed}, indent=2)+"\n")
                    print("E2 completed", completed[-1], flush=True)
                    del model
                    gc.collect()
        if not older_reads:
            raise RuntimeError("No genuinely older nonconditioning memory was exercised")
        result = {"passed": True, "scope": "E2 engineering only, not efficacy", "completed": completed,
                  "long_history_reads": older_reads, "all_token_native_bit_exact": True,
                  "sparse_fresh_replay_bit_exact": True, "setup": setup,
                  "E3_launched": False, "E4_launched": False}
        (output/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
        finalize_tracking(run, output, {"completed": True, "passed": True, "long_history_reads": older_reads}, ["validation.json", "setup.json", "progress.json"])
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"passed": False, "error": repr(error), "E3_launched": False}, indent=2)+"\n")
        run.summary.update({"completed": False, "passed": False, "failure": repr(error)})
        raise
    finally:
        author.iter_predictions = original
        run.finish()


if __name__ == "__main__":
    main()

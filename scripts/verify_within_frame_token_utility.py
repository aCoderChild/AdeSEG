"""Engineering-only real-frame parity and source-isolation gate; no Dice scoring.

Reuses native prediction, private replay, source provenance and integrity checks
from Run171. The full scientific runner and safe-resume gate remain separate.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cfbi_training import capture_training_read
from modeling.within_frame_token_utility import ARMS, WithinFrameTokenRead, choose_source, removal_plan
from scripts import run_complementary_memory_pilot as base
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_replay import check_exact, check_layout
from scripts.run_regional_token_selection import EXTRA_SOURCES, replay
from scripts.run_token_contribution import native_signatures

SOURCES = ("protocols/within_frame_token_utility_v1.json",
           "modeling/within_frame_token_utility.py", "scripts/verify_within_frame_token_utility.py",
           "tests/test_within_frame_token_utility.py")


@torch.inference_mode()
def verify_sequence(model, frames, name, out):
    out.mkdir()
    weights = parameter_digest(model)
    reference = {}
    with tempfile.TemporaryDirectory(dir=out, prefix="native_") as temp:
        state, prompt = base.prepare(model, frames, Path(temp))
        for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
            if ids != [1]:
                raise RuntimeError("Unexpected PolypGen object identity")
            reference[frame] = native_signatures(state, frame, prompt, logits)
    del state
    gc.collect()
    observer = WithinFrameTokenRead(model)
    intervention_source = None
    receipts = []
    with tempfile.TemporaryDirectory(dir=out, prefix="observed_") as temp:
        state, prompt = base.prepare(model, frames, Path(temp))
        iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
        seen = []
        while True:
            observer.arm, observer.source_frame = "native", None
            with observer:
                try:
                    frame, ids, logits = next(iterator)
                except StopIteration:
                    break
            seen.append(frame)
            if ids != [1] or native_signatures(state, frame, prompt, logits) != reference[frame]:
                raise RuntimeError(f"Observer changed native trajectory at {name}/{frame}")
            if frame == prompt:
                continue
            collected = observer.calls[-1]
            sample = capture_training_read(model, state, frame)
            bank_hash = lambda: fingerprint(bank_contents(state, frame, [], model.max_obj_ptrs_in_encoder))["sha256"]
            before = bank_hash()
            native, trace = replay(model, sample, observer)
            check_exact(native, state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame])
            check_layout(trace, collected)
            full = model._get_orig_video_res_output(state, native["pred_masks"])[1]
            if tensor_hash(full) != tensor_hash(logits):
                raise RuntimeError("Full-resolution replay mismatch")
            if frame == prompt+3:
                plan = choose_source(trace["memory_frames"], name, frame)
                intervention_source = plan["source_frame"]
                if intervention_source is None:
                    raise RuntimeError("Engineering fixture has no eligible source")
                base.atomic_json(out/"source_plan.json", plan)
            if frame in (prompt+3, prompt+4):
                observer.source_frame = intervention_source
                arms = ARMS[1:] if frame == prompt+3 else ARMS[1:17]
                for arm in arms:
                    observer.arm = arm
                    result, changed = replay(model, sample, observer)
                    check_layout(changed, trace)
                    for layer in changed["selection_layers"]:
                        expected_counts = [4096-int(removal_plan(arm).sum()) if f == intervention_source else 4096
                                           for f in layer["frame_ids"]]
                        if layer["counts"] != expected_counts:
                            raise RuntimeError("Intervention changed an unintended source/count")
                    if not all(torch.isfinite(result[k]).all() for k in
                               ("pred_masks", "obj_ptr", "object_score_logits")):
                        raise RuntimeError("Nonfinite replay output")
                    repeated, repeated_trace = replay(model, sample, observer)
                    check_exact(result, repeated)
                    if repeated_trace != changed:
                        raise RuntimeError("Repeated intervention receipts differ")
                    receipts.append({"frame": frame, "arm": arm,
                        "output_hashes": {k: tensor_hash(result[k]) for k in
                                          ("pred_masks", "obj_ptr", "object_score_logits")},
                        "trace": changed})
            if bank_hash() != before:
                raise RuntimeError("Diagnostic changed native bank/cache")
            observer.calls.clear()
            base.atomic_json(out/"progress.json", {"pid": os.getpid(), "frame": frame,
                                                   "interventions": len(receipts)})
        if seen != list(range(prompt, len(frames))) or len(receipts) != 35:
            raise RuntimeError("Incomplete engineering coverage")
    if parameter_digest(model) != weights:
        raise RuntimeError("Pretrained weights changed")
    base.atomic_json(out/"receipt.json", {"passed": True, "native_frames": len(seen),
        "parameter_sha256": weights, "interventions": receipts,
        "scope": "single-object real-frame engineering; no scientific efficacy or multi-object claim"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Freeze this engineering implementation before even collecting gate outputs.
    for path in SOURCES:
        if subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT) != (ROOT/path).read_bytes():
            raise RuntimeError(f"Commit engineering sources first: {path}")
    protocol = json.loads((ROOT/"protocols/token_contribution_v1.json").read_text())
    manifest = ROOT/"data/polypgen_sequence.jsonl"
    if digest(args.checkpoint) != protocol["checkpoint_sha256"] or digest(manifest) != protocol["manifest_sha256"]:
        raise RuntimeError("Checkpoint or manifest differs from Run171")
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(out)
    tempfile.tempdir = str(out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = sorted(set(base.SOURCE_REL) | set(EXTRA_SOURCES) | set(SOURCES) |
                     {"scripts/run_token_contribution.py", "protocols/token_contribution_v1.json"})
    hashes = {p: digest(ROOT/p) for p in sources}
    videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
    inputs = base.input_hashes(videos, ["seq2", "seq3"])
    base.atomic_json(out/"setup.json", {"sources": hashes, "inputs": inputs,
        "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(manifest),
        "torch": torch.__version__, "pid": os.getpid(), "efficacy": False})
    with zipfile.ZipFile(out/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(ROOT/path, arcname=path)
    try:
        repo, _ = base.activate_author_repo()
        for name in ("seq2", "seq3"):
            model = base.init_predictor(args.checkpoint, repo)
            verify_sequence(model, videos[name], name, out/name)
            del model
            gc.collect()
        if hashes != {p: digest(ROOT/p) for p in sources} or inputs != base.input_hashes(videos, ["seq2", "seq3"]):
            raise RuntimeError("Sources or inputs changed during verification")
        base.atomic_json(out/"completion.json", {"passed": True, "sequences": ["seq2", "seq3"],
            "setup_sha256": digest(out/"setup.json"), "efficacy": False,
            "remaining": ["scientific runner", "safe resume verification", "scientific protocol freeze"]})
    except BaseException as exc:
        base.atomic_json(out/"failure.json", {"error": repr(exc), "no_bypass": True})
        raise


if __name__ == "__main__":
    main()

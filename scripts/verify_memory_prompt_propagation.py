"""Verify full-video native replay and auxiliary memory reads before efficacy training."""
import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.native_stage_capture import NativeStageCapture
from modeling.regional_memory_prompts import NativeRegionalMemoryPrompts, RegionalMemoryPrompts
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame, summarize


def evaluate(model, frames, name, reference, output, heads=None, mode="regional", *, disabled=False):
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    for module in model.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache.clear()
    before = parameter_digest(model)
    prompt_sha = parameter_digest(heads) if heads is not None else None
    native = {r["frame"]: r for r in reference}
    rows = []
    with tempfile.TemporaryDirectory(prefix="memory_prompts_", dir=output) as directory:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(directory))
            context = (NativeRegionalMemoryPrompts(model, heads, mode, enabled=not disabled)
                       if heads is not None else nullcontext())
            started = time.monotonic()
            try:
                with context as adapter, NativeStageCapture(model) as observer:
                    trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                    for expected in range(prompt, len(frames)):
                        observer.begin(False)
                        frame, ids, logits = next(trajectory)
                        if frame != expected or ids != [1] or not torch.isfinite(logits).all():
                            raise RuntimeError("Invalid native trajectory or object identity")
                        if frame == prompt:
                            continue
                        layers = observer.layers
                        if [r["layer"] for r in layers] != [0, 1, 2, 3]:
                            raise RuntimeError("Missing memory-attention layer")
                        reads = adapter.calls[-4:] if heads is not None and not disabled else []
                        if reads:
                            for index, (layer, record) in enumerate(zip(layers, reads)):
                                auxiliary = record["auxiliary_keys"]
                                if (record["frame"] != frame or record["object_batch"] != 1
                                        or layer["keys"] != record["spatial_keys"] + record["pointer_keys"] + auxiliary
                                        or layer["pointer_keys"] != record["pointer_keys"] + auxiliary):
                                    raise RuntimeError("Native/auxiliary read bookkeeping differs")
                                if record["auxiliary_keys"] != len(record["memory_frames"]) * 3 * 16:
                                    raise RuntimeError("Auxiliary token count differs from protocol")
                                if any(span["frame"] >= frame for span in record["memory_frames"]):
                                    raise RuntimeError("Future memory was read")
                                # The legacy observer names the entire non-RoPE tail pointers.
                                layers[index] = {**layer, "keys": layer["keys"] - auxiliary,
                                                 "pointer_keys": record["pointer_keys"]}
                        stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame]
                        row = score_frame(logits, frames, frame)
                        row.update(stored_sha256=fingerprint(stored)["sha256"],
                            attention_layers=layers, auxiliary_reads=reads,
                            object_score=float(stored["object_score_logits"].item()))
                        if heads is None or disabled:
                            if any(row[key] != native[frame][key] for key in
                                   ("prediction_sha256", "stored_sha256", "scores", "present", "absent_fp")):
                                raise RuntimeError(f"Disabled/native replay differs: {name}/{frame}")
                        rows.append(row)
                        write(output / "progress.json", {"sequence": name, "mode": "disabled" if disabled else mode,
                                                         "frame": frame, "targets": len(rows)})
            finally:
                model.reset_state(state)
            elapsed = time.monotonic() - started
    if [r["frame"] for r in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError("Incomplete trajectory")
    if parameter_digest(model) != before or (heads is not None and parameter_digest(heads) != prompt_sha):
        raise RuntimeError("Inference changed parameters")
    return rows, {**summarize(rows), "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-run", "engineering-gate", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        gate = json.loads((args.engineering_gate / "completion.json").read_text())
        protocol = json.loads((args.engineering_gate / "protocol.json").read_text())
        native = json.loads((args.native_run / "completion.json").read_text())
        audit = json.loads((args.native_run / "independent_audit.json").read_text())
        native_setup = json.loads((args.native_run / "setup.json").read_text())
        if not gate.get("passed") or not native.get("passed") or not audit.get("passed"):
            raise RuntimeError("Audited prerequisites missing")
        if native.get("scope") != "cohort" or set(native["entries"]) != set(audit["sequences"]):
            raise RuntimeError("Native cohort evidence differs")
        if (digest(args.checkpoint) != protocol["checkpoint_sha256"]
                or native_setup["checkpoint_sha256"] != protocol["checkpoint_sha256"]
                or native_setup["model_config"] != protocol["model_config"]
                or native_setup["manifest_sha256"] != digest(ROOT / protocol["manifest"])):
            raise RuntimeError("Native/checkpoint/manifest provenance differs")
        sources = {**gate["sources"], **{p: digest(ROOT / p) for p in (
            "scripts/verify_memory_prompt_propagation.py", "modeling/native_stage_capture.py",
            "protocols/regional_memory_prompt_propagation_gate_v1.json")}}
        if any(digest(ROOT / n) != sha for n, sha in sources.items()):
            raise RuntimeError("Engineering source inventory changed")
        write(out / "setup.json", {"sources": sources, "protocol": protocol,
            "engineering_completion_sha256": digest(args.engineering_gate / "completion.json"),
            "native_completion_sha256": digest(args.native_run / "completion.json"),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, revision = activate_author_repo()
        if revision != protocol["sam2_revision"]:
            raise RuntimeError("Author revision differs")
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        results, artifacts = {}, {}
        for name in protocol["sequences"]:
            folder = Path(native["entries"][name])
            receipt = json.loads((folder / "completion.json").read_text())
            if (digest(folder / "frames.json") != receipt["frames_sha256"]
                    or digest(folder / "completion.json") != audit["sequences"][name]["completion_sha256"]):
                raise RuntimeError("Archived native frames changed")
            reference = json.loads((folder / "frames.json").read_text())
            results[name] = {}
            schedules = None
            for mode in ("disabled", "regional", "global"):
                model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
                    checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
                model.requires_grad_(False)
                model.eval()
                heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"]).eval()
                weights = args.engineering_gate / name / f"{'regional' if mode == 'disabled' else mode}.pt"
                heads.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
                rows, summary = evaluate(model, videos[name], name, reference, out, heads,
                    "global" if mode == "global" else "regional", disabled=mode == "disabled")
                current = [r["attention_layers"] for r in rows]
                if schedules is not None and current != schedules:
                    raise RuntimeError("Full-video native schedules differ across arms")
                schedules = current
                filename = f"{name}_{mode}.json"
                write(out / filename, rows)
                artifacts[filename] = digest(out / filename)
                results[name][mode] = summary
                del model, heads
                gc.collect()
        if sources != {n: digest(ROOT / n) for n in sources}:
            raise RuntimeError("Sources changed during propagation gate")
        write(out / "completion.json", {"passed": True, "engineering_only": True,
            "sources": sources, "results": results, "artifacts": artifacts,
            "engineering_completion_sha256": digest(args.engineering_gate / "completion.json"),
            "note": "Diagnostic prompt weights only; no efficacy, independent generalization or multi-object claim"})
        write(out / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

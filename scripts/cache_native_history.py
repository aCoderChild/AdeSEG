"""Build verified native prediction banks for history-conditioned training."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from modeling.native_history_cache import bank_payload, capture_cached_block, load_bank
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.run_soft_role_affinity_cv import SEQUENCES, load_native_baseline

PROTOCOL = ROOT / "protocols/native_history_cache_v1.json"


def build_cache(model, frames, name, reference, output, provenance):
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    folder = output / name
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["sequence"] != name or receipt["num_frames"] != len(frames)
                or receipt["predictions"] != [{"frame": r["frame"], "sha256": r["prediction_sha256"]}
                                              for r in reference]):
            raise RuntimeError("Cache sequence, geometry or native references differ")
        return load_bank(folder / "bank.pt", receipt, **provenance), receipt
    folder.mkdir(parents=True, exist_ok=False)
    for module in model.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache.clear()
    expected = {row["frame"]: row for row in reference}
    predictions = []
    with tempfile.TemporaryDirectory(prefix="frames_", dir=folder) as directory:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(directory))
            try:
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if ids != [1] or not torch.isfinite(logits).all():
                        raise RuntimeError("Invalid native object or prediction")
                    if frame > prompt:
                        sha = tensor_hash(logits)
                        if sha != expected[frame]["prediction_sha256"]:
                            raise RuntimeError(f"Native cache prediction differs at {name}/{frame}")
                        predictions.append({"frame": frame, "sha256": sha})
                    write(output / "progress.json", {"phase": "cache", "sequence": name, "frame": frame})
                if [r["frame"] for r in predictions] != list(expected):
                    raise RuntimeError("Native cache target coverage differs")
                payload = bank_payload(state)
                receipt = {**provenance, "sequence": name, "object_id": 1,
                    "num_frames": len(frames), "prompt": prompt, "predictions": predictions,
                    "native_frames_checked": len(predictions), "payload_sha256": fingerprint(payload)["sha256"]}
                save_tensor(payload, folder / "bank.pt")
                receipt.update(bank_sha256=digest(folder / "bank.pt"), bank_bytes=(folder / "bank.pt").stat().st_size)
                restored = load_bank(folder / "bank.pt", receipt, **provenance)
                if fingerprint(payload)["sha256"] != receipt["payload_sha256"]:
                    raise RuntimeError("Serialization mutated the native bank")
                write(receipt_path, receipt)
            finally:
                model.reset_state(state)
    return restored, receipt


def verify_block(model, frames, spec, payload, capacity, output):
    samples, capture = capture_cached_block(model, frames, spec["frames"], output, payload)
    index = capacity["sequences"].index(spec["sequence"])
    with torch.no_grad():
        native_loss = native_qk_clip_loss(model, samples, check_native=True)
    if float(native_loss) != capacity["before"]["regional"][index]:
        raise RuntimeError("Cached and direct-history native losses differ")
    results = {}
    for mode in ("regional", "global"):
        for stage in ("zero", "trained"):
            torch.manual_seed(20261121)
            heads = RegionalLowRankHeads(model.mem_dim, 4)
            if stage == "trained":
                heads.load_state_dict(load(output / "capacity_weights" / f"{mode}.pt"))
            with torch.no_grad(), NativeRegionalLowRank(model, heads, mode) as observer:
                loss = native_qk_clip_loss(model, samples, check_native=stage == "zero")
            expected = capacity["before" if stage == "zero" else "after"][mode][index]
            if float(loss) != expected or len(observer.calls) != 4 * len(samples):
                raise RuntimeError("Cached adapter loss or read count differs")
            if [(r["frame"], r["layer"]) for r in observer.calls] != [
                (sample["frame"], layer) for sample in samples for layer in range(4)
            ] or any(not r["pointer_tokens_unchanged"] for r in observer.calls):
                raise RuntimeError("Cached layer/pointer schedule differs")
            if any(r["memory_frames"] != capture["records"][i // 4]["spatial_frames"]
                   for i, r in enumerate(observer.calls)):
                raise RuntimeError("Cached native memory-frame schedule differs")
            results[f"{mode}_{stage}"] = {"loss": float(loss), "reads": observer.calls}
            if mode == "regional" and stage == "zero":
                with NativeRegionalLowRank(model, heads, mode):
                    loss = native_qk_clip_loss(model, samples, check_native=True)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all()
                       for p in heads.parameters() if p.requires_grad):
                    raise RuntimeError("Cached inputs do not support adapter gradients")
                gradients = {name: float(p.grad.abs().sum()) for name, p in heads.named_parameters()
                             if p.requires_grad}
                if any(value == 0 for name, value in gradients.items() if name.endswith("lora_B")):
                    raise RuntimeError("Cached regional expert has no gradient")
                results[f"{mode}_{stage}"]["gradient_l1"] = gradients
    return {"sequence": spec["sequence"], "capture": capture, "results": results, "gradients_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capacity-run", "native-run", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-receipt", type=Path)
    args = parser.parse_args()
    if not args.gate_only and args.gate_receipt is None:
        parser.error("Full cache requires --gate-receipt from this cache gate")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=not args.gate_only)
    try:
        protocol = json.loads(PROTOCOL.read_text())
        capacity_done = json.loads((args.capacity_run / "completion.json").read_text())
        capacity_audit = json.loads((args.capacity_run / "independent_final_audit.json").read_text())
        capacity_setup = json.loads((args.capacity_run / "setup.json").read_text())
        capacity_protocol = json.loads((args.capacity_run / "protocol.json").read_text())
        capacity = json.loads((args.capacity_run / "results.json").read_text())
        if not capacity_done["passed_integrity"] or not capacity_audit["passed"] or not all(capacity["mean_loss_decreased"].values()):
            raise RuntimeError("Native-history capacity prerequisite failed")
        if any(digest(ROOT / name) != sha for name, sha in capacity_done["sources"].items()):
            raise RuntimeError("Sources changed since native-history verification")
        sources = {**capacity_done["sources"], **{name: digest(ROOT / name) for name in (
            "modeling/native_history_cache.py", "scripts/cache_native_history.py",
            "protocols/native_history_cache_v1.json", "tests/test_native_history_cache.py",
            "workflows/catalog.py", "tests/conftest.py")}}
        provenance = {"checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(ROOT / protocol["manifest"]), "model_config": protocol["model_config"]}
        if capacity_protocol["model_config"] != protocol["model_config"]:
            raise RuntimeError("Native-history model configuration differs")
        if any(capacity_setup[key] != provenance[key] for key in ("checkpoint_sha256", "manifest_sha256")):
            raise RuntimeError("Native-history checkpoint or manifest differs")
        if args.gate_only:
            for name in sources:
                target = output / "source_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            write(output / "protocol.json", protocol)
            write(output / "setup.json", {**provenance, "sources": sources,
                "capacity_completion_sha256": digest(args.capacity_run / "completion.json")})
            (output / "capacity_weights").mkdir()
            for mode in ("regional", "global"):
                source = args.capacity_run / f"{mode}.pt"
                if digest(source) != capacity_audit["weights_sha256"][mode]:
                    raise RuntimeError("Capacity diagnostic weights changed")
                shutil.copyfile(source, output / "capacity_weights" / source.name)
                if digest(output / "capacity_weights" / source.name) != capacity_audit["weights_sha256"][mode]:
                    raise RuntimeError("Copied capacity weights differ")
        else:
            gate = json.loads(args.gate_receipt.read_text())
            setup = json.loads((output / "setup.json").read_text())
            if (not gate["passed"] or gate["sources"] != sources or setup["sources"] != sources
                    or any(gate[key] != value for key, value in provenance.items())):
                raise RuntimeError("Full cache differs from its verified gate")
            for entry in gate["caches"].values():
                if (digest(output / entry["bank"]) != entry["bank_sha256"]
                        or digest(output / entry["receipt"]) != entry["receipt_sha256"]):
                    raise RuntimeError("Verified gate cache files changed")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, _ = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
            checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
        backbone = parameter_digest(model)
        if backbone != capacity_done["backbone_sha256"]:
            raise RuntimeError("Native backbone differs from the verified capacity run")
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        _, native = load_native_baseline(args.native_run, videos,
            provenance["checkpoint_sha256"], provenance["manifest_sha256"])
        names = [r["sequence"] for r in protocol["gate_blocks"]] if args.gate_only else SEQUENCES
        checks, index = [], {}
        for name in names:
            payload, receipt = build_cache(model, videos[name], name, native[name], output, provenance)
            index[name] = {"bank": f"{name}/bank.pt", "receipt": f"{name}/receipt.json",
                "bank_sha256": receipt["bank_sha256"], "receipt_sha256": digest(output / name / "receipt.json"),
                "native_frames_checked": receipt["native_frames_checked"]}
            if args.gate_only:
                spec = next(r for r in protocol["gate_blocks"] if r["sequence"] == name)
                checks.append(verify_block(model, videos[name], spec, payload, capacity, output))
                write(output / "gate_replay.json", checks)
            del payload
        if parameter_digest(model) != backbone or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Native backbone changed during cache verification")
        if any(digest(ROOT / name) != sha for name, sha in sources.items()):
            raise RuntimeError("Sources changed during cache generation")
        if args.gate_only:
            write(output / "gate_completion.json", {**provenance, "passed": True,
                "sources": sources, "sequences": names, "backbone_sha256": backbone,
                "gate_replay_sha256": digest(output / "gate_replay.json"), "caches": index})
            write(output / "progress.json", {"phase": "gate_complete"})
        else:
            write(output / "cache_index.json", index)
            write(output / "completion.json", {**provenance, "complete": True, "sequences": len(index),
                "native_targets": sum(r["native_frames_checked"] for r in index.values()),
                "sources": sources, "cache_index_sha256": digest(output / "cache_index.json"),
                "gate_completion_sha256": digest(args.gate_receipt)})
            write(output / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

"""Drive-only engineering verification. Never launches an efficacy experiment."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from modeling.complementary_memory import ComplementaryMemoryReader, NativeComplementaryRead
from modeling.cfbi_training import capture_training_read, clone_tensors
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write, save_tensor

PROTOCOL = "protocols/complementary_memory_v1.json"


def rollout(model, frames, output, reader=None, mode="regional", capture=False):
    rows, samples = [], []
    with tempfile.TemporaryDirectory(prefix="gate_frames_", dir=output) as directory:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(directory))
            observer = NativeComplementaryRead(model, reader, mode, inference_state=state) if reader is not None else None
            try:
                with observer if observer is not None else nullcontext():
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1] or not torch.isfinite(logits).all():
                            raise RuntimeError("Unexpected IDs or nonfinite propagation")
                        bank = state["output_dict_per_obj"][0]
                        stored = bank["cond_frame_outputs"].get(frame, bank["non_cond_frame_outputs"].get(frame))
                        row = {"frame": frame, "prediction_sha256": tensor_hash(logits),
                               "stored_sha256": fingerprint({k: stored[k] for k in (
                                   "pred_masks", "obj_ptr", "object_score_logits", "maskmem_features")})["sha256"]}
                        if observer is not None and frame > prompt:
                            row["read"] = observer.calls[-1]
                        rows.append(row)
                        if capture and prompt < frame <= prompt + 4:
                            sample = capture_training_read(model, state, frame, object_id=1)
                            with torch.inference_mode(False):
                                sample.update(clone_tensors({
                                    "native_logits": stored["pred_masks"],
                                    "native_object_score": stored["object_score_logits"],
                                    "target": torch.from_numpy((load_label_mask(frames[frame].mask_path, [1]) == 1).copy())[None, None].float(),
                                }))
                            samples.append(sample)
            finally:
                model.reset_state(state)
    return rows, samples


def signature(rows):
    return [{k: v for k, v in row.items() if k != "read"} for row in rows]


def gradient_norm(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or nonfinite added-module gradients")
    return sum(float(g.abs().sum()) for g in gradients)


def multi_object_gate(model, reader, frames, output):
    # Two persistent IDs from a split real prompt: engineering fixture only.
    mask = torch.from_numpy((load_label_mask(frames[0].mask_path, [1]) == 1).copy())
    coordinates = mask.nonzero()
    middle = int(coordinates[:, 1].median())
    left = mask.clone()
    left[:, middle:] = False
    right = mask & ~left
    if not left.any() or not right.any():
        raise RuntimeError("Cannot construct two nonempty fixture instances")
    masks = {7: left, 42: right}

    def run(order, active):
        result = {}
        with tempfile.TemporaryDirectory(prefix="instances_", dir=output) as directory:
            stage_frames(frames[:3], Path(directory))
            with torch.inference_mode():
                state = model.init_state(video_path=directory, offload_video_to_cpu=True, offload_state_to_cpu=False)
                try:
                    for object_id in order:
                        model.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=masks[object_id])
                    observer = NativeComplementaryRead(model, active, inference_state=state) if active is not None else None
                    with observer if observer is not None else nullcontext():
                        for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=0):
                            for index, object_id in enumerate(ids):
                                bank = state["output_dict_per_obj"][state["obj_id_to_idx"][object_id]]
                                stored = bank["cond_frame_outputs"].get(frame, bank["non_cond_frame_outputs"].get(frame))
                                result[f"{frame}:{object_id}"] = {
                                    "prediction": tensor_hash(logits[index]),
                                    "stored": fingerprint({k: stored[k] for k in (
                                        "pred_masks", "obj_ptr", "object_score_logits", "maskmem_features")})["sha256"]}
                    if observer is not None and {row["object_id"] for row in observer.calls} != set(order):
                        raise RuntimeError("Multi-object read lost persistent identity")
                finally:
                    model.reset_state(state)
        return result

    zero = copy.deepcopy(reader)
    torch.nn.init.zeros_(zero.output_projection.weight)
    if run([7, 42], None) != run([7, 42], zero):
        raise RuntimeError("Two-object zero identity failed")
    combined = run([7, 42], reader)
    if combined != run([42, 7], reader):
        raise RuntimeError("Instance order affects trained predictions")
    for object_id in (7, 42):
        if run([object_id], reader) != {k: v for k, v in combined.items() if k.endswith(f":{object_id}")}:
            raise RuntimeError("Cross-object leakage")
    return {"passed": True, "object_ids": [7, 42], "frames": 3,
            "scope": "Split-prompt fixture on real images; not natural multi-object efficacy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Requires the pinned official SAM2.1 Tiny checkpoint")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        sources = [ROOT / PROTOCOL]
        for folder in ("modeling", "datasets", "evaluation", "scripts", "training"):
            sources.extend(sorted((ROOT / folder).rglob("*.py")))
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(ROOT)))
        repo, revision = activate_author_repo()
        args.method, args.device = "sam2", "cpu"
        args.model_config = protocol["model"]["configuration"]
        model = build_author_predictor(args, repo)
        initial_parameters = parameter_digest(model)
        torch.manual_seed(protocol["seed"])
        initial = ComplementaryMemoryReader()
        manifest = ROOT / protocol["data"]["manifest"]
        videos = load_manifest(manifest, "dev")
        setup = {"protocol": protocol, "sources": hashes, "sam2_revision": revision,
                 "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(manifest),
                 "trainable_parameters": sum(p.numel() for p in initial.parameters()),
                 "source_snapshot_sha256": digest(output / "source_snapshot.zip")}
        write(output / "setup.json", setup)
        results = {}
        trained = None
        for name in protocol["verification"]["sequences"]:
            write(output / "status.json", {"phase": "native_capture", "sequence": name})
            native, samples = rollout(model, videos[name], output, capture=True)
            prompt = native[0]["frame"]
            short_frames = videos[name][prompt:prompt + 5]
            short_native, _ = rollout(model, short_frames, output)
            if len(samples) != 4:
                raise RuntimeError("Requires four consecutive captured training targets")
            write(output / f"{name}_native.json", native)
            bank_hashes = [fingerprint(s["output_dict"])["sha256"] for s in samples]
            modes = {}
            for mode in ("generic", "regional"):
                reader = copy.deepcopy(initial)
                write(output / "status.json", {"phase": "zero_identity", "sequence": name, "mode": mode})
                identity, _ = rollout(model, videos[name], output, reader, mode)
                if signature(identity) != signature(native):
                    raise RuntimeError(f"Full-sequence zero identity failed: {name}/{mode}")
                write(output / f"{name}_{mode}_identity.json", identity)
                optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-4, weight_decay=.01)
                updates = []
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with NativeComplementaryRead(model, reader, mode) as observer:
                        # Reuse existing native autoregressive loss; NEVER call
                        # enable_memory_qk_training. All original params stay frozen.
                        loss = native_qk_clip_loss(model, samples, check_native=(step == 0))
                    loss.backward()
                    grads = {"output": gradient_norm(reader.output_projection)}
                    if step == 1:
                        grads.update({"feature_pool": gradient_norm(reader.summarizer.feature_pred),
                                      "pool_weights": gradient_norm(reader.summarizer.weights_pred),
                                      "reader": gradient_norm(reader.reader)})
                    if not torch.isfinite(loss) or min(grads.values()) <= 0 or len(observer.calls) != 4:
                        raise RuntimeError("Training path lacks finite active gradients or reads")
                    if any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError("Original SAM2 received gradients")
                    updates.append({"step": step, "loss": float(loss.detach()), "gradient_l1": grads})
                    optimizer.step()
                if bank_hashes != [fingerprint(s["output_dict"])["sha256"] for s in samples]:
                    raise RuntimeError("Training mutated reference memory banks")
                reader.eval()
                active, _ = rollout(model, short_frames, output, reader, mode)
                if signature(active) == signature(short_native):
                    raise RuntimeError("Trained branch has no observable effect")
                write(output / f"{name}_{mode}_active.json", active)
                save_tensor(reader.state_dict(), output / f"{name}_{mode}_engineering.pt")
                modes[mode] = {"identity_frames": len(identity), "post_prompt_reads": len(identity) - 1,
                               "updates": updates, "active_frames": len(active),
                               "reference_banks_unchanged": True}
                trained = reader
            results[name] = modes
            write(output / "partial_checks.json", results)
        write(output / "status.json", {"phase": "multi_object_fixture"})
        from scripts.memory_token_io import prompt_index
        prompt = prompt_index(videos["seq2"], {"initialization": "first_labeled_frame"})
        multi = multi_object_gate(model, trained, videos["seq2"][prompt:], output)
        if parameter_digest(model) != initial_parameters:
            raise RuntimeError("Original SAM2 weights changed")
        if hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
            raise RuntimeError("Implementation changed during gate")
        receipt = {"passed": True, "scope": "engineering_only_no_efficacy_claim",
                   "sequences": results, "multi_object": multi,
                   "original_parameters_unchanged": True, "backbone_sha256": initial_parameters,
                   "setup_sha256": digest(output / "setup.json"),
                   "seconds": time.monotonic() - started,
                   "pilot_authorized_by_this_receipt": False,
                   "remaining": ["feature-versus-mask diagnostic", "frozen training/evaluation pilot protocol"]}
        write(output / "completion.json", receipt)
        write(output / "status.json", {"phase": "gate_complete", "efficacy_started": False})
        (output / "results.md").write_text(
            "# Complementary memory reader: engineering verification\n\n"
            "Native parity, learned-path gradients, bank integrity and instance isolation passed. "
            "These are implementation checks, not evidence of improved Dice.\n\n"
            "| Sequence | Generic parity frames | Regional parity frames | Training steps per arm |\n"
            "|---|---:|---:|---:|\n" + "".join(
                f"| {n} | {r['generic']['identity_frames']} | {r['regional']['identity_frames']} | 2 |\n"
                for n, r in results.items()) +
            "\nThe earlier merging results motivated preservation of native dense reads. "
            "This new branch passed engineering verification; regional utility remains untested.\n")
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error)})
        write(output / "status.json", {"phase": "gate_failed", "efficacy_started": False})
        raise


if __name__ == "__main__":
    main()

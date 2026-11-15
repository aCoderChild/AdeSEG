"""Verify instance isolation using native SAM2 and controlled real-frame prompts."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import ManifestFrame, load_manifest, stage_frames
from modeling.multi_object_video import capture_object_clips, object_clip_loss, propagate_object_prompts
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_regional_low_rank import SOURCES

PROTOCOL = ROOT / "protocols/multi_object_regional_gate_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", help="Override the config; recorded before execution")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        protocol = json.loads(PROTOCOL.read_text())
        if args.model_config:
            protocol["model_config"] = args.model_config
        sources = list(dict.fromkeys(SOURCES + [
            "protocols/multi_object_regional_gate_v1.json", "scripts/verify_multi_object_regional.py",
            "modeling/multi_object_video.py", "modeling/regional_low_rank.py",
            "modeling/cfbi_region_qk_training.py", "tests/test_multi_object_memory.py",
            "external/LoRA/loralib/layers.py", "external/sam2/tools/vos_inference.py",
            "external/sam2/sam2/modeling/position_encoding.py",
            "datasets/instance_masks.py", "scripts/run_multi_object_regional.py",
            "tests/test_instance_masks.py",
        ]))
        hashes = {name: digest(ROOT / name) for name in sources}
        write(out / "protocol.json", protocol)
        write(out / "setup.json", {"sources": hashes, "checkpoint_sha256": digest(args.checkpoint),
                                  "manifest_sha256": digest(ROOT / protocol["manifest"])})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, revision = activate_author_repo()
        if revision != protocol["sam2_revision"]:
            raise RuntimeError("SAM2 revision differs")
        model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
            checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
        model.requires_grad_(False)
        backbone = parameter_digest(model)
        videos = load_manifest(ROOT / protocol["manifest"], "dev")
        originals = sample_clip(videos[protocol["sequence"]], protocol["clip_length"], protocol["seed"])
        initial = load_label_mask(originals[0].mask_path, [1]) == 1
        split = int(np.median(np.nonzero(initial)[1]))
        frames, masks = [], []
        for index, frame in enumerate(originals):
            mask = load_label_mask(frame.mask_path, [1]) == 1
            columns = np.arange(mask.shape[1])[None, :]
            labels = np.where(mask, np.where(columns < split, 7, 42), 0).astype(np.uint8)
            path = out / f"fixture_{index}.png"
            Image.fromarray(labels).save(path)
            masks.append(labels)
            frames.append(ManifestFrame(frame.video_id, frame.frame_index, frame.image_path, path))
        if any(not (masks[0] == object_id).any() for object_id in (7, 42)):
            raise RuntimeError("Fixture must contain both instances")

        with tempfile.TemporaryDirectory(prefix="frames_", dir=out) as directory:
            stage_frames(frames, Path(directory))

            def new_state():
                from sam2.modeling.position_encoding import PositionEmbeddingSine
                for module in model.modules():
                    if isinstance(module, PositionEmbeddingSine):
                        module.cache.clear()
                return model.init_state(video_path=directory, offload_video_to_cpu=True,
                                        offload_state_to_cpu=False)

            def prompts(order, late=False):
                schedule = {}
                for object_id in order:
                    frame = 1 if late and object_id == 42 else 0
                    schedule.setdefault(frame, {})[object_id] = masks[frame] == object_id
                return schedule

            def rollout(name, order, heads=None, mode="regional", late=False):
                state = new_state()
                context = (NativeRegionalLowRank(model, heads, mode, inference_state=state)
                           if heads is not None else nullcontext())
                records = {}
                with context as observer:
                    for frame, ids, logits in propagate_object_prompts(model, state, prompts(order, late)):
                        if late and frame == 0 and ids != [7]:
                            raise RuntimeError("Later instance leaked into an earlier output")
                        for index, object_id in enumerate(ids):
                            bank = state["output_dict_per_obj"][state["obj_id_to_idx"][object_id]]
                            stored = bank["cond_frame_outputs"].get(frame,
                                bank["non_cond_frame_outputs"].get(frame))
                            records[f"{frame}:{object_id}"] = {
                                "prediction": tensor_hash(logits[index:index + 1]),
                                "stored": fingerprint(stored)["manifest"],
                            }
                calls = observer.calls if heads is not None else []
                if heads is not None:
                    for frame in range(len(frames)):
                        for object_id in order:
                            prompt_frame = 1 if late and object_id == 42 else 0
                            selected = [c for c in calls if c["frame"] == frame and c["object_id"] == object_id]
                            expected = [0, 1, 2, 3] if frame > prompt_frame else []
                            if [c["layer"] for c in selected] != expected:
                                raise RuntimeError("Wrong object ownership or native layer order")
                write(out / f"{name}.json", {"records": records, "reads": calls})
                model.reset_state(state)
                return records

            native = rollout("native", [7, 42])
            if native != rollout("native_reverse_order", [42, 7]):
                raise RuntimeError("Native registration-order mismatch")
            native_late = rollout("native_late", [7, 42], late=True)
            state = new_state()
            clips = capture_object_clips(model, state, frames, prompts([7, 42]), [7, 42])
            model.reset_state(state)
            training = {}
            for mode in ("regional", "global"):
                torch.manual_seed(protocol["seed"])
                heads = RegionalLowRankHeads(model.mem_dim, protocol["training"]["rank"])
                if native != rollout(f"{mode}_zero", [7, 42], heads, mode):
                    raise RuntimeError("Zero adapter differs from native multi-object propagation")
                if native_late != rollout(f"{mode}_zero_late", [7, 42], heads, mode, late=True):
                    raise RuntimeError("Zero adapter differs for a later-prompted object")
                optimizer = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],
                    lr=protocol["training"]["learning_rate"], weight_decay=protocol["training"]["weight_decay"])
                steps = []
                for step in range(protocol["training"]["updates"]):
                    optimizer.zero_grad(set_to_none=True)
                    per_object_gradients = {}
                    parameters = [p for p in heads.parameters() if p.requires_grad]
                    if step == 0:
                        for object_id, samples in clips.items():
                            single_loss, _ = object_clip_loss(model, heads, {object_id: samples},
                                                             mode, check_native=True)
                            grads = torch.autograd.grad(single_loss, parameters)
                            if any(not torch.isfinite(g).all() for g in grads):
                                raise RuntimeError("Nonfinite per-instance gradient")
                            per_object_gradients[object_id] = [g.detach() for g in grads]
                    loss, reads = object_clip_loss(model, heads, clips, mode, check_native=step == 0)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite joint loss")
                    loss.backward()
                    if step == 0:
                        for index, parameter in enumerate(parameters):
                            expected = torch.stack([g[index] for g in per_object_gradients.values()]).mean(0)
                            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)
                    gradients = {}
                    for name, parameter in heads.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        grad = parameter.grad
                        if grad is None or not torch.isfinite(grad).all():
                            raise RuntimeError(f"Invalid gradient {name}")
                        gradients[name] = float(grad.abs().sum())
                        if (step > 0 or name.endswith("lora_B")) and gradients[name] == 0:
                            raise RuntimeError(f"Missing gradient {name}")
                    steps.append({"loss": float(loss.detach()), "gradients": gradients, "reads": reads})
                    if step == 0:
                        names = [name for name, p in heads.named_parameters() if p.requires_grad]
                        instance_norms = {object_id: {name: float(g.abs().sum())
                            for name, g in zip(names, grads)}
                            for object_id, grads in per_object_gradients.items()}
                        if any(value == 0 for norms in instance_norms.values()
                               for name, value in norms.items() if name.endswith("lora_B")):
                            raise RuntimeError("An instance did not train every residual expert")
                        steps[-1]["per_instance_gradient_l1"] = instance_norms
                    torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                    optimizer.step()
                adapted = rollout(f"{mode}_trained", [7, 42], heads, mode)
                if adapted == native:
                    raise RuntimeError("Updated adapter has no measurable effect")
                if adapted != rollout(f"{mode}_trained_reverse_order", [42, 7], heads, mode):
                    raise RuntimeError("Nonzero adapter depends on instance registration order")
                for object_id in (7, 42):
                    alone = rollout(f"{mode}_alone_{object_id}", [object_id], heads, mode)
                    if alone != {k: v for k, v in adapted.items() if k.endswith(f":{object_id}")}:
                        raise RuntimeError("Another object changed this object's memory or prediction")
                rollout(f"{mode}_trained_late", [7, 42], heads, mode, late=True)
                torch.save(heads.state_dict(), out / f"{mode}.pt")
                training[mode] = steps
                write(out / "training.json", training)
        if parameter_digest(model) != backbone or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen SAM2 changed")
        if hashes != {name: digest(ROOT / name) for name in sources}:
            raise RuntimeError("Source changed during gate")
        write(out / "completion.json", {"passed": True, "sources": hashes,
            "checkpoint_sha256": digest(args.checkpoint), "model_config": protocol["model_config"],
            "manifest_frames": [f.frame_index for f in originals], "object_ids": [7, 42],
            "seconds": time.monotonic() - started, "scope": protocol["fixture"],
            "backbone_sha256": backbone, "training": training})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

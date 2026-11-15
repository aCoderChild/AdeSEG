"""Real-frame engineering gate for prompt-anchor local ToMe; no Dice scoring."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.native_memory_layout import selected_memory_frames
from modeling.prompt_anchor_tome import NativePromptAnchorToMe
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from modeling.token_diagnostic_replay import NativeReadCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write


def decode(model, state, capture):
    conditioned = model.memory_attention(**capture.memory)
    image = conditioned.permute(1, 2, 0).reshape_as(capture.decoder["image_embeddings"])
    sam = model._forward_sam_heads(
        backbone_features=image,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=capture.decoder["high_res_features"],
        multimask_output=capture.decoder["multimask_output"],
    )
    _, logits = model._get_orig_video_res_output(state, sam[3])
    return logits, sam[6]


def read_once(model, state, capture, frame, num_frames, native_logits, native_score, condition):
    bank = state["output_dict_per_obj"][0]
    spans = selected_memory_frames(model, frame, bank, num_frames)
    spatial_count = spans[-1]["stop"]
    pointers = int(capture.memory["num_obj_ptr_tokens"])
    if capture.memory["memory"].shape[0] != spatial_count + pointers:
        raise RuntimeError("Captured memory does not align with native frame spans/pointer tail")
    bank_hash = fingerprint(bank)["sha256"]

    if condition == "prompt_anchor":
        reader = NativePromptAnchorToMe(model)
        reader.current_output_dict = bank
    else:
        reader = NativeRegionConstrainedToMe(
            model,
            mode="global",
            merge_divisor=4097 if condition == "noop" else 2,
        )
        reader.memberships = torch.full((1, spatial_count, 3), 1 / 3)
    reader.frame = frame
    reader.spans = spans

    with reader:
        logits, presence = decode(model, state, capture)
    if not torch.isfinite(logits).all() or not torch.isfinite(presence).all():
        raise RuntimeError("Nonfinite output from prompt-anchor gate")
    if [call["layer"] for call in reader.calls] != [0, 1, 2, 3]:
        raise RuntimeError("Gate missed one or more native memory-attention layers")
    pointer_reference = None
    if condition == "prompt_anchor":
        references = {
            (call.get("pointer_reference_frame"), call.get("pointer_reference_sha256"))
            for call in reader.calls
        }
        if len(references) != 1 or None in next(iter(references)):
            raise RuntimeError("Prompt anchor pointer changed or was not recorded across layers")
        pointer_reference = next(iter(references))
    if fingerprint(bank)["sha256"] != bank_hash:
        raise RuntimeError("Read-only attention intervention modified the source bank")
    if condition == "noop":
        if not torch.equal(logits, native_logits) or not torch.equal(presence, native_score):
            raise RuntimeError("Native/no-op replay was not bit-identical")
        if any(call["spatial_keys_before"] != call["spatial_keys_after"] for call in reader.calls):
            raise RuntimeError("No-op unexpectedly changed the spatial token count")
    budgets = [
        [(source["frame"], [(window["window"], window["merges"])
                             for window in source["windows"]])
         for source in call["frames"]]
        for call in reader.calls
    ]
    if any(call["spatial_key_mass_sum"] != call["spatial_keys_before"] for call in reader.calls):
        raise RuntimeError("Spatial key mass was not conserved")
    return {
        "condition": condition,
        "frame": frame,
        "logits_sha256": digest_tensor(logits),
        "presence_sha256": digest_tensor(presence),
        "presence_score": float(presence.reshape(-1)[0]),
        "native_logits_exact": bool(torch.equal(logits, native_logits)),
        "native_presence_exact": bool(torch.equal(presence, native_score)),
        "attention_layers": [call["layer"] for call in reader.calls],
        "spatial_keys_before_after": [
            [call["spatial_keys_before"], call["spatial_keys_after"]]
            for call in reader.calls
        ],
        "pointer_keys": [call["pointer_keys"] for call in reader.calls],
        "budgets": budgets,
        "pointer_reference_frame": pointer_reference[0] if pointer_reference else None,
        "pointer_reference_sha256": pointer_reference[1] if pointer_reference else None,
        "attention_audit": reader.calls,
    }


def digest_tensor(tensor):
    import hashlib

    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "protocols/prompt_anchor_tome_gate_v2.json")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    source_files = protocol["gate_source_files"]
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    if source_hashes != protocol["source_sha256"]:
        raise RuntimeError("Gate sources differ from the frozen protocol")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    try:
        checkpoint = Path(protocol["checkpoint"])
        manifest = ROOT / protocol["manifest"]
        if digest(checkpoint) != protocol["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint checksum differs from frozen protocol")
        if digest(manifest) != protocol["manifest_sha256"]:
            raise RuntimeError("Manifest checksum differs from frozen protocol")
        tome_repo = ROOT / "external/ToMe"
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tome_repo, text=True).strip() != protocol["tome_revision"]:
            raise RuntimeError("ToMe revision differs from frozen protocol")
        repo = ROOT / protocol["sam2_repository"]
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != protocol["sam2_revision"]:
            raise RuntimeError("Reference SAM2 revision differs from frozen protocol")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
            raise RuntimeError("Reference SAM2 checkout has tracked modifications")
        if "sam2" in sys.modules:
            raise RuntimeError("SAM2 was imported before activating its pinned checkout")
        sys.path.insert(0, str(repo))
        write(out / "setup.json", {
            "protocol": protocol,
            "protocol_sha256": digest(args.protocol),
            "source_sha256": source_hashes,
        })
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        results = {}
        for sequence in protocol["sequences"]:
            frames = videos[sequence]
            model = build_author_predictor(SimpleNamespace(
                method="sam2", device="cpu", checkpoint=checkpoint,
                model_config=protocol["model_config"],
            ), repo)
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("SAM2 must be frozen and in evaluation mode")
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
                raise RuntimeError("Native memory schedule differs from protocol")
            parameters_before = parameter_digest(model)
            prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
            targets = {prompt + offset for offset in protocol["target_offsets"]}
            if max(targets) >= len(frames):
                raise RuntimeError("Frozen targets exceed video length")
            results[sequence] = []
            sequence_dir = out / sequence
            sequence_dir.mkdir()
            with tempfile.TemporaryDirectory(dir=sequence_dir) as temporary:
                state, prepared_prompt = prepare(model, frames, Path(temporary))
                if prepared_prompt != prompt:
                    raise RuntimeError("Prompt frame differs from protocol")
                trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                for expected in range(prompt, max(targets) + 1):
                    with NativeReadCapture(model) as capture:
                        actual, ids, native_logits = next(trajectory)
                    if actual != expected or ids != [1]:
                        raise RuntimeError("Native trajectory frame/object ordering changed")
                    if expected not in targets:
                        continue
                    native_replay, native_score = decode(model, state, capture)
                    if not torch.equal(native_replay, native_logits):
                        raise RuntimeError("Native capture replay differs from native propagation")
                    noop = read_once(model, state, capture, expected, len(frames), native_logits,
                                     native_score, "noop")
                    anchor = read_once(model, state, capture, expected, len(frames), native_logits,
                                       native_score, "prompt_anchor")
                    plan = {
                        frame["frame"]: {
                            window["window"]: window["merges"]
                            for window in frame["windows"]
                        }
                        for frame in anchor["attention_audit"][0]["frames"]
                    }
                    generic = NativeRegionConstrainedToMe(
                        model, mode="global", merge_plan_by_target={expected: plan},
                    )
                    bank = state["output_dict_per_obj"][0]
                    spans = selected_memory_frames(model, expected, bank, len(frames))
                    spatial = spans[-1]["stop"]
                    generic.frame, generic.spans = expected, spans
                    generic.memberships = torch.full((1, spatial, 3), 1 / 3)
                    bank_before = fingerprint(bank)["sha256"]
                    with generic:
                        generic_logits, generic_presence = decode(model, state, capture)
                    if fingerprint(bank)["sha256"] != bank_before:
                        raise RuntimeError("Matched generic control modified the bank")
                    generic_budget = [
                        [(source["frame"], [(window["window"], window["merges"])
                                             for window in source["windows"]])
                         for source in call["frames"]]
                        for call in generic.calls
                    ]
                    if generic_budget != anchor["budgets"]:
                        raise RuntimeError("Prompt-anchor and generic control merge budgets differ")
                    if [call["layer"] for call in generic.calls] != [0, 1, 2, 3]:
                        raise RuntimeError("Generic matched control missed an attention layer")
                    if not all(torch.isfinite(x).all() for x in (generic_logits, generic_presence)):
                        raise RuntimeError("Generic matched control produced nonfinite output")
                    if not torch.equal(decode(model, state, capture)[0], native_logits):
                        raise RuntimeError("Attention hooks did not restore native output")
                    results[sequence].append({
                        "frame": expected,
                        "native_noop": noop,
                        "prompt_anchor_tome": anchor,
                        "generic_matched": {
                            "logits_sha256": digest_tensor(generic_logits),
                            "presence_sha256": digest_tensor(generic_presence),
                            "presence_score": float(generic_presence.reshape(-1)[0]),
                            "budgets": generic_budget,
                            "attention_layers": [call["layer"] for call in generic.calls],
                            "pointer_keys": [call["pointer_keys"] for call in generic.calls],
                            "attention_audit": generic.calls,
                        },
                    })
                    write(out / "results.json", results)
                    write(out / "progress.json", {
                        "sequence": sequence,
                        "frame": expected,
                        "completed_targets": sum(map(len, results.values())),
                        "total_targets": len(protocol["sequences"]) * len(protocol["target_offsets"]),
                    })
                trajectory.close()
                model.reset_state(state)
            if parameter_digest(model) != parameters_before:
                raise RuntimeError("SAM2 parameters changed during the gate")
            del model
            gc.collect()
        if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
            raise RuntimeError("Source changed during gate execution")
        write(out / "completion.json", {
            "engineering_passed": True,
            "targets": sum(map(len, results.values())),
            "conditions": ["native_noop", "prompt_anchor_tome", "generic_matched"],
            "scope": "real-frame engineering only; no GT, Dice or efficacy claim",
            "native_noop_exact": True,
            "fixed_prompt_pointer_mapping": True,
            "matched_generic_budget": True,
            "source_banks_unchanged": True,
            "sam2_weights_unchanged": True,
            "results_sha256": digest(out / "results.json"),
        })
    except Exception as exc:
        write(out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

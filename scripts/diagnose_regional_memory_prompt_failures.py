"""Capture masks and stored-role maps for the stopped Run 150 failures.

This is a post-hoc diagnostic replay of frozen fold-1 weights. It does not train,
select checkpoints, or define a new method. A successful run must reproduce every
saved frame hash and metric before its masks or role maps are interpreted.
"""

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.native_stage_capture import NativeStageCapture
from modeling.regional_memory_prompts import NativeRegionalMemoryPrompts, RegionalMemoryPrompts
from scripts.bank_integrity import fingerprint
from scripts.continue_regional_memory_prompts_cv import read_native, validate_prerequisites
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_soft_role_affinity_cv import score_frame
from scripts.run_regional_refinement import write


FAILURE_FRAMES = {
    "seq8": set(range(1, 11)) | set(range(40, 73)),
    "seq14": set(range(1, 11)) | set(range(120, 186)) | set(range(200, 211)),
}


def replay(args, setup, native, audit, native_setup, repo, videos, output, sequence, mode):
    folder = args.run_dir / "fold1" / mode
    expected_path = folder / f"{sequence}_frames.json"
    receipt_path = folder / f"{sequence}_completion.json"
    expected = json.loads(expected_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if digest(expected_path) != receipt["frames_sha256"]:
        raise RuntimeError(f"Frozen frame receipt changed: {sequence}/{mode}")
    weights_path = folder / "prompts.pt"
    training_receipt = json.loads((folder / "training_completion.json").read_text())
    if digest(weights_path) != training_receipt["weights_sha256"]:
        raise RuntimeError(f"Frozen prompt weights changed: {sequence}/{mode}")

    model = build_author_predictor(SimpleNamespace(
        method="sam2", device="cpu", checkpoint=args.checkpoint,
        model_config=setup["protocol"]["model_config"]), repo)
    model.requires_grad_(False)
    model.eval()
    backbone_digest = parameter_digest(model)
    heads = RegionalMemoryPrompts(
        model.mem_dim, setup["protocol"]["tokens_per_region_per_frame"])
    heads.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    heads.eval()
    head_digest = parameter_digest(heads)
    del audit, native_setup

    prompt_norms = []
    prompt_layer_by_frame = {}
    adapter = NativeRegionalMemoryPrompts(model, heads, mode)

    def capture_prompt_norms(module, inputs, outputs):
        key, value = outputs
        frame = adapter.frame
        layer = prompt_layer_by_frame.get(frame, 0)
        prompt_layer_by_frame[frame] = layer + 1
        tokens = setup["protocol"]["tokens_per_region_per_frame"]
        offset = 0
        for span in inputs[3]:
            for region in range(3):
                start = offset + region * tokens
                stop = start + tokens
                prompt_norms.append({
                    "frame": frame,
                    "layer": layer,
                    "memory_frame": span["frame"],
                    "region": region,
                    "key_norm_mean": float(key[:, start:stop].float().norm(dim=-1).mean()),
                    "value_norm_mean": float(value[:, start:stop].float().norm(dim=-1).mean()),
                })
            offset += 3 * tokens

    handle = heads.register_forward_hook(capture_prompt_norms)
    rows, memory_masks, role_maps = [], [], {}
    sequence_frames = videos[sequence]
    expected_by_frame = {row["frame"]: row for row in expected}
    temp_parent = output / "temporary"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{sequence}_{mode}_", dir=temp_parent) as temp:
        state = None
        try:
            with torch.inference_mode():
                state, prompt = prepare(model, sequence_frames, Path(temp))
                with adapter, NativeStageCapture(model) as observer:
                    trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                    for expected_frame in range(prompt, len(sequence_frames)):
                        observer.begin(False)
                        frame, object_ids, logits = next(trajectory)
                        if frame != expected_frame or object_ids != [1] or not torch.isfinite(logits).all():
                            raise RuntimeError(f"Invalid trajectory at {sequence}/{frame}")
                        if frame == prompt:
                            continue
                        row = score_frame(logits, sequence_frames, frame)
                        stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame]
                        layers = observer.layers
                        reads = adapter.calls[-4:]
                        if [item["layer"] for item in layers] != [0, 1, 2, 3]:
                            raise RuntimeError(f"Missing attention layer at {sequence}/{frame}")
                        if len(reads) != 4 or [item["layer"] for item in reads] != [0, 1, 2, 3]:
                            raise RuntimeError(f"Missing auxiliary read at {sequence}/{frame}")
                        for index, (layer, read) in enumerate(zip(layers, reads, strict=True)):
                            aux = read["auxiliary_keys"]
                            if (read["frame"] != frame
                                    or layer["keys"] != read["spatial_keys"] + read["pointer_keys"] + aux
                                    or layer["pointer_keys"] != read["pointer_keys"] + aux
                                    or any(span["frame"] >= frame for span in read["memory_frames"])):
                                raise RuntimeError(f"Attention receipt mismatch at {sequence}/{frame}")
                            layers[index] = {**layer, "keys": layer["keys"] - aux,
                                             "pointer_keys": read["pointer_keys"]}
                        row.update(
                            stored_sha256=fingerprint(stored)["sha256"],
                            attention_layers=layers,
                            auxiliary_reads=reads,
                            object_score=float(stored["object_score_logits"].item()),
                        )
                        saved = expected_by_frame[frame]
                        for field in ("prediction_sha256", "stored_sha256", "scores", "present",
                                      "absent_fp", "object_score", "attention_layers", "auxiliary_reads"):
                            if row.get(field) != saved.get(field):
                                raise RuntimeError(f"Replay differs from frozen {field}: {sequence}/{mode}/{frame}")

                        mask = stored["pred_masks"]
                        if tuple(mask.shape) != (1, 1, 256, 256):
                            raise RuntimeError(f"Unexpected stored mask shape at {sequence}/{frame}: {tuple(mask.shape)}")
                        memory_masks.append(mask[0, 0].detach().cpu().numpy().astype(np.float16))
                        role_summary = []
                        memberships = adapter.memberships[0].detach().float().cpu()
                        for span in adapter.spans:
                            role = memberships[span["start"]:span["stop"]]
                            side = int(np.sqrt(role.shape[0]))
                            if side * side != role.shape[0]:
                                raise RuntimeError("Memory role map is not square")
                            role_summary.append({
                                "frame": span["frame"],
                                "temporal_position": span["t_pos"],
                                "mean_membership": role.mean(0).tolist(),
                                "hard_token_counts": torch.bincount(role.argmax(-1), minlength=3).tolist(),
                            })
                            if frame in FAILURE_FRAMES[sequence]:
                                role_maps[f"target_{frame}_memory_{span['frame']}"] = (
                                    role.reshape(side, side, 3).numpy().astype(np.float16))
                        row["memory_role_summary"] = role_summary
                        row["auxiliary_prompt_norms"] = [
                            record for record in prompt_norms if record["frame"] == frame]
                        expected_norms = 4 * 3 * len(adapter.spans)
                        if len(row["auxiliary_prompt_norms"]) != expected_norms:
                            raise RuntimeError(f"Expected per-memory-frame prompt norms at {sequence}/{frame}")
                        rows.append(row)
        finally:
            handle.remove()
            if state is not None:
                model.reset_state(state)

    if [row["frame"] for row in rows] != list(expected_by_frame):
        raise RuntimeError(f"Incomplete replay: {sequence}/{mode}")
    if parameter_digest(model) != backbone_digest or parameter_digest(heads) != head_digest:
        raise RuntimeError(f"Model weights changed during read-only replay: {sequence}/{mode}")
    npz_path = output / f"{sequence}_{mode}_memory_masks_and_roles.npz"
    np.savez_compressed(
        npz_path,
        frames=np.asarray([row["frame"] for row in rows], dtype=np.int32),
        stored_mask_logits=np.stack(memory_masks),
        selected_targets=np.asarray(sorted(FAILURE_FRAMES[sequence]), dtype=np.int32),
        **role_maps,
    )
    rows_path = output / f"{sequence}_{mode}_diagnostic.json"
    write(rows_path, rows)
    return {
        "frames": len(rows),
        "saved_cv_receipt_sha256": digest(receipt_path),
        "saved_cv_frames_sha256": digest(expected_path),
        "prompt_weights_sha256": digest(weights_path),
        "diagnostic_rows_sha256": digest(rows_path),
        "diagnostic_npz_sha256": digest(npz_path),
        "replay_bitwise_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "native-run", "propagation-gate", "history-gate", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        setup = json.loads((args.run_dir / "setup.json").read_text())
        setup["run_dir"] = str(args.run_dir)
        if not (args.run_dir / "continuation_failure.json").exists():
            raise RuntimeError("Expected Run 150 to be terminal before post-hoc replay")
        native, audit, native_setup, repo = validate_prerequisites(args, setup)
        references, receipts = read_native(native, audit)
        if not references or set(references) != set(audit["sequences"]):
            raise RuntimeError("Native cohort references are incomplete")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = setup["protocol"]
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        write(output / "setup.json", {
            "status": "post_hoc_diagnostic_only",
            "run150_setup_sha256": digest(args.run_dir / "setup.json"),
            "run150_continuation_setup_sha256": digest(args.run_dir / "continuation_setup.json"),
            "protocol_sha256": digest(ROOT / "protocols" / "regional_memory_prompts_cv_v1.json"),
            "sequences": list(FAILURE_FRAMES),
            "arms": ["regional", "global"],
            "no_training_or_checkpoint_selection": True,
            "diagnostic_source_sha256": digest(__file__),
        })
        results = {}
        for sequence in FAILURE_FRAMES:
            for mode in ("regional", "global"):
                results[f"{sequence}_{mode}"] = replay(
                    args, setup, native, audit, native_setup, repo, videos, output, sequence, mode)
                write(output / "progress.json", {"sequence": sequence, "mode": mode,
                    "phase": "diagnostic_replay_complete", "results": results})
        write(output / "completion.json", {
            "complete": True,
            "post_hoc_diagnostic_only": True,
            "all_replays_match_frozen_run": all(row["replay_bitwise_match"] for row in results.values()),
            "results": results,
        })
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

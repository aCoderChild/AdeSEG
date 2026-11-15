"""Compare nonzero adapter clip replay with the author's video propagation."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.prediction_conditioned_affinity import PredictionConditionedTrackStep, PredictionRoleScales
from modeling.prediction_conditioned_training import clip_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip


def capture_outputs(model):
    original = model._forward_sam_heads
    rows = []

    def forward(*args, **kwargs):
        outputs = original(*args, **kwargs)
        row = {name: tensor_hash(outputs[index]) for name, index in (
            ("low_res", 3), ("high_res", 4), ("pointer", 5), ("object_score", 6)
        )}
        row["tensors"] = {name: outputs[index].detach().float().cpu().clone()
                          for name, index in (("low_res", 3), ("high_res", 4),
                                              ("pointer", 5), ("object_score", 6))}
        rows.append(row)
        return outputs

    model._forward_sam_heads = forward
    return original, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    setup = json.loads((args.run / "setup.json").read_text())
    protocol = setup["protocol"]
    manifest = ROOT / protocol["data"]["manifest"]
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    repo, revision = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(
        method="sam2", device="cpu", checkpoint=ROOT / protocol["model"]["checkpoint"],
        model_config=protocol["model"]["config"],
    ), repo)
    frames = sample_clip(videos["seq5"], 5, protocol["training"]["seed"])
    samples = capture_clip(model, frames, out)
    results = {}
    for mode in ("regional", "global"):
        path = args.run / "fold1" / f"{mode}_scales.pt"
        scales = PredictionRoleScales().eval()
        scales.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        assert scales.raw_strength.count_nonzero() > 0
        replays = {}
        for gradients in (True, False):
            original, replay = capture_outputs(model)
            try:
                with torch.set_grad_enabled(gradients):
                    loss, _ = clip_loss(model, scales, samples, mode)
                assert torch.isfinite(loss)
            finally:
                model._forward_sam_heads = original
            assert len(replay) == 2 * (len(frames) - 1)
            replays[str(gradients)] = replay
        with tempfile.TemporaryDirectory(prefix="propagation_", dir=out) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            original, propagated = capture_outputs(model)
            try:
                with PredictionConditionedTrackStep(model, scales, mode), torch.inference_mode():
                    for _, _, _ in model.propagate_in_video(state, start_frame_idx=prompt):
                        pass
            finally:
                model._forward_sam_heads = original
                model.reset_state(state)
        assert len(propagated) == len(replay)
        comparisons = {}
        for gradients, replay in replays.items():
            comparisons[gradients] = []
            for i in range(len(frames) - 1):
                row = {"frame": int(frames[i + 1].frame_index)}
                for label, index in (("first_pass", 2 * i), ("second_pass", 2 * i + 1)):
                    left, right = replay[index]["tensors"], propagated[index]["tensors"]
                    row[label] = {
                        name: {"bit_equal": torch.equal(left[name], right[name]),
                               "max_absolute_difference": float((left[name] - right[name]).abs().max()),
                               "sign_disagreement_elements": int(((left[name] > 0) != (right[name] > 0)).sum())}
                        for name in left
                    }
                comparisons[gradients].append(row)
        results[mode] = {"checkpoint_sha256": digest(path), "frames": comparisons,
                         "strengths": scales.strengths().detach().tolist()}
        write(out / f"{mode}.json", results[mode])

    missing_small_targets = []
    for name in protocol["data"]["sequences"]:
        for frame in videos[name]:
            target = torch.from_numpy((load_label_mask(frame.mask_path, [1]) == 1).copy()).float()
            small = F.interpolate(target[None, None], (256, 256), mode="nearest")
            if target.any() and not small.any():
                missing_small_targets.append({"sequence": name, "frame": frame.frame_index})
    write(out / "completion.json", {
        "complete": True, "scope": "Diagnostic: seq5 five-frame nonzero replay plus full-cohort GT downsampling audit",
        "sampled_frames": [f.frame_index for f in frames], "arms": results,
        "sam2_revision": revision, "source_run": str(args.run),
        "gt_present_frames_erased_by_training_resize": missing_small_targets,
        "audit_script_sha256": digest(Path(__file__)),
    })


if __name__ == "__main__":
    main()

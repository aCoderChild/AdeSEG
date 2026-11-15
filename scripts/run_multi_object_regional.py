"""Export native or regional multi-instance predictions using SAM2 propagation."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.instance_masks import instance_prompts
from datasets.io import save_masks_to_dir
from datasets.video import load_manifest, stage_frames
from modeling.multi_object_video import propagate_object_prompts
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engineering-gate", type=Path, required=True,
                        help="Passing multi-object completion.json for this checkpoint/config/source")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--mode", choices=("native", "regional", "global"), default="native")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--object-ids", type=int, nargs="+")
    parser.add_argument("--prompt-frames", type=Path,
                        help='JSON object mapping persistent IDs to manifest frame indices')
    parser.add_argument("--void-id", type=int, help="Dataset's ignored label, e.g. 255 for DAVIS")
    args = parser.parse_args()
    if (args.mode != "native") != (args.adapter is not None):
        parser.error("Regional/global mode requires adapter weights; native mode uses none")
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        gate = json.loads(args.engineering_gate.read_text())
        if (gate.get("passed") is not True or gate.get("checkpoint_sha256") != digest(args.checkpoint)
                or gate.get("model_config") != args.model_config):
            raise RuntimeError("Engineering gate does not match this checkpoint/config")
        for name, expected in gate["sources"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Source changed since the engineering gate: {name}")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        frames = load_manifest(args.manifest, args.split)[args.sequence]
        schedule = None
        if args.prompt_frames:
            schedule = {int(k): int(v) for k, v in json.loads(args.prompt_frames.read_text()).items()}
        prompts = instance_prompts(frames, object_ids=args.object_ids, prompt_frames=schedule,
                                   void_id=args.void_id)
        repo, revision = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
            checkpoint=args.checkpoint, model_config=args.model_config), repo)
        setup = {"mode": args.mode, "sam2_revision": revision, "config": args.model_config,
                 "engineering_gate_sha256": digest(args.engineering_gate),
                 "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(args.manifest),
                 "adapter_sha256": digest(args.adapter) if args.adapter else None,
                 "void_id": args.void_id, "prompts": {str(frames[f].frame_index): list(ids)
                                                         for f, ids in prompts.items()}}
        write(out / "setup.json", setup)
        heads = None
        if args.adapter:
            heads = RegionalLowRankHeads(model.mem_dim, args.rank).eval()
            heads.load_state_dict(torch.load(args.adapter, map_location="cpu", weights_only=True), strict=True)
        with tempfile.TemporaryDirectory(prefix="frames_", dir=out) as directory:
            stage_frames(frames, Path(directory))
            state = model.init_state(video_path=directory, offload_video_to_cpu=True,
                                     offload_state_to_cpu=False)
            context = (NativeRegionalLowRank(model, heads, args.mode, inference_state=state)
                       if heads is not None else nullcontext())
            records = []
            with context as observer:
                for frame, ids, logits in propagate_object_prompts(model, state, prompts):
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Nonfinite instance prediction")
                    masks = {object_id: (logits[index] > 0).cpu().numpy()
                             for index, object_id in enumerate(ids)}
                    save_masks_to_dir(out / "predictions", args.sequence,
                        f"{frames[frame].frame_index:05d}", masks,
                        state["video_height"], state["video_width"], per_obj_png_file=True)
                    records.append({"frame": frames[frame].frame_index, "object_ids": list(ids),
                                    "logits_sha256": tensor_hash(logits)})
                    write(out / "frames.json", records)
            write(out / "reads.json", observer.calls if heads is not None else [])
            model.reset_state(state)
        expected = [f.frame_index for f in frames[min(prompts):]]
        if [record["frame"] for record in records] != expected:
            raise RuntimeError("Missing or repeated output frames")
        write(out / "completion.json", {**setup, "frames": len(records),
            "scope": "CPU FP32 prediction export; not an official benchmark score. Per-instance PNGs preserve IDs and native overlap behavior."})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

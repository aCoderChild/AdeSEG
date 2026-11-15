"""Check the regional LoRA integration on a native PolypGen training clip."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip
from scripts.verify_prediction_conditioned_affinity import SOURCES

PROTOCOL = ROOT / "protocols/regional_low_rank_gate_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        protocol = json.loads(PROTOCOL.read_text())
        sources = list(dict.fromkeys(SOURCES + [
            "protocols/regional_low_rank_gate_v1.json", "modeling/regional_low_rank.py",
            "scripts/verify_regional_low_rank.py", "tests/test_regional_low_rank.py",
            "modeling/cfbi_region_qk_training.py", "external/LoRA/loralib/layers.py",
        ]))
        hashes = {name: digest(ROOT / name) for name in sources}
        write(out / "protocol.json", protocol)
        write(out / "setup.json", {"sources": hashes, "checkpoint_sha256": digest(args.checkpoint),
                                  "manifest_sha256": digest(ROOT / protocol["manifest"])})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, revision = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(
            method="sam2", device="cpu", checkpoint=args.checkpoint,
            model_config=protocol["model_config"]), repo)
        model.requires_grad_(False)
        backbone = parameter_digest(model)
        videos = load_manifest(ROOT / protocol["manifest"], "dev")
        frames = sample_clip(videos[protocol["sequence"]], protocol["clip_length"], protocol["seed"])
        samples = capture_clip(model, frames, out)
        with torch.inference_mode():
            for sample in samples:
                reference = model.track_step(
                    frame_idx=sample["frame"], is_init_cond_frame=False,
                    current_vision_feats=sample["vision"], current_vision_pos_embeds=sample["positions"],
                    feat_sizes=sample["sizes"], point_inputs=None, mask_inputs=None,
                    output_dict=sample["output_dict"], num_frames=sample["num_frames"],
                    track_in_reverse=False, run_mem_encoder=False)
                sample["native_object_score"] = reference["object_score_logits"].clone()
        results = {}
        for mode in ("regional", "global"):
            torch.manual_seed(protocol["seed"])
            heads = RegionalLowRankHeads(channels=model.mem_dim, rank=protocol["rank"]).train()
            parameters = [p for p in heads.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=protocol["learning_rate"],
                                          weight_decay=protocol["weight_decay"])
            records = []
            for step in range(protocol["updates"] + 1):
                optimizer.zero_grad(set_to_none=True)
                with NativeRegionalLowRank(model, heads, mode) as observer:
                    loss = native_qk_clip_loss(model, samples, check_native=step == 0)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite propagation loss")
                expected = [(s["frame"], layer) for s in samples for layer in range(4)]
                if [(c["frame"], c["layer"]) for c in observer.calls] != expected:
                    raise RuntimeError("Native layer/read schedule changed")
                loss.backward()
                gradients = {}
                for name, parameter in heads.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    grad = parameter.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise RuntimeError(f"Invalid gradient: {name}")
                    norm = float(grad.abs().sum())
                    if (step > 0 or name.endswith("lora_B")) and norm == 0:
                        raise RuntimeError(f"No gradient: {name}")
                    gradients[name] = norm
                if any(p.grad is not None for p in model.parameters()) or parameter_digest(model) != backbone:
                    raise RuntimeError("Frozen backbone changed")
                records.append({"step": step, "loss": float(loss.detach()),
                                "gradient_l1": gradients, "reads": observer.calls})
                write(out / f"{mode}.json", records)
                write(out / "progress.json", {"mode": mode, "step": step})
                if step < protocol["updates"]:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
            torch.save(heads.state_dict(), out / f"{mode}.pt")
            results[mode] = {"records": records, "trainable_parameters": sum(p.numel() for p in parameters)}
        for left, right in zip(results["regional"]["records"], results["global"]["records"]):
            if left["reads"] != right["reads"]:
                raise RuntimeError("Regional/control memory schedules differ")
        if hashes != {name: digest(ROOT / name) for name in sources}:
            raise RuntimeError("Sources changed during gate")
        write(out / "completion.json", {"passed": True, "sam2_revision": revision,
            "manifest_frames": [f.frame_index for f in frames], "modes": results,
            "backbone_sha256": backbone, "sources": hashes,
            "seconds": time.monotonic() - started, "scope": protocol["scope"]})
        write(out / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

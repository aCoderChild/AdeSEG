"""Verify auxiliary regional memory prompts with every original SAM2 weight frozen."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.decoder_core_training import decoder_clip_loss
from modeling.regional_memory_prompts import NativeRegionalMemoryPrompts, RegionalMemoryPrompts
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip
from scripts.verify_prediction_conditioned_affinity import SOURCES

PROTOCOL = ROOT / "protocols/regional_memory_prompts_gate_v1.json"


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
        if digest(args.checkpoint) != protocol["checkpoint_sha256"]:
            raise RuntimeError("Official checkpoint hash differs")
        names = list(dict.fromkeys(SOURCES + [
            "protocols/regional_memory_prompts_gate_v1.json",
            "modeling/regional_memory_prompts.py", "tests/test_regional_memory_prompts.py",
            "scripts/verify_regional_memory_prompts.py", "modeling/simvos_region_prototypes.py",
            "external/SimVOS/model/models_vit.py", "external/MemSeg-Agent/src/memseg_agent/model.py",
            "external/sam2/sam2/modeling/sam/transformer.py", "scripts/bank_integrity.py",
            "modeling/decoder_core_training.py", "modeling/decoder_adaptation.py",
        ]))
        hashes = {name: digest(ROOT / name) for name in names}
        write(out / "protocol.json", protocol)
        write(out / "setup.json", {"sources": hashes, "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(ROOT / protocol["manifest"]),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, revision = activate_author_repo()
        if revision != protocol["sam2_revision"]:
            raise RuntimeError("Author revision differs")
        model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
            checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
        model.requires_grad_(False)
        model.eval()
        backbone = parameter_digest(model)
        videos = load_manifest(ROOT / protocol["manifest"], "dev")
        results = {}
        for sequence in protocol["sequences"]:
            folder = out / sequence
            folder.mkdir()
            write(out / "progress.json", {"sequence": sequence, "phase": "native_capture"})
            frames = sample_clip(videos[sequence], protocol["clip_length"], protocol["seed"])
            samples = capture_clip(model, frames, folder)
            with torch.inference_mode():
                for sample in samples:
                    native = model.track_step(frame_idx=sample["frame"], is_init_cond_frame=False,
                        current_vision_feats=sample["vision"], current_vision_pos_embeds=sample["positions"],
                        feat_sizes=sample["sizes"], point_inputs=None, mask_inputs=None,
                        output_dict=sample["output_dict"], num_frames=sample["num_frames"],
                        track_in_reverse=False, run_mem_encoder=False)
                    sample["native_object_score"] = native["object_score_logits"].clone()
            banks = [fingerprint(s["output_dict"])["sha256"] for s in samples]
            schedule = []
            handles = [layer.cross_attn_image.register_forward_pre_hook(
                lambda module, args, kw, index=index: schedule.append({
                    "layer": index, "spatial_keys": kw["k"].shape[1] - int(kw.get("num_k_exclude_rope", 0)),
                    "pointer_keys": int(kw.get("num_k_exclude_rope", 0))}), with_kwargs=True)
                for index, layer in enumerate(model.memory_attention.layers)]
            try:
                heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"])
                with torch.no_grad(), NativeRegionalMemoryPrompts(model, heads, enabled=False):
                    decoder_clip_loss(model, samples, check_native=True)
            finally:
                for handle in handles:
                    handle.remove()
            arms = {}
            for mode in ("regional", "global"):
                torch.manual_seed(protocol["seed"])
                heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"]).train()
                initial = parameter_digest(heads)
                optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["learning_rate"],
                                              weight_decay=protocol["weight_decay"])
                records = []
                for step in range(protocol["updates"] + 1):
                    write(out / "progress.json", {"sequence": sequence, "mode": mode, "step": step})
                    optimizer.zero_grad(set_to_none=True)
                    with NativeRegionalMemoryPrompts(model, heads, mode) as observer:
                        loss, info = decoder_clip_loss(model, samples, retain_first_write=True)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    temporal, = torch.autograd.grad(info["last_loss"], info["first_write"], retain_graph=True)
                    if not torch.isfinite(temporal).all() or temporal.abs().sum() == 0:
                        raise RuntimeError("Missing gradient through previous predicted memory")
                    loss.backward()
                    gradients = {n: float(p.grad.abs().sum()) if p.grad is not None else None
                                 for n, p in heads.named_parameters()}
                    if any(g is None or not torch.isfinite(torch.tensor(g)) for g in gradients.values()):
                        raise RuntimeError("Missing/nonfinite external-module gradient")
                    if sum(gradients.values()) == 0 or any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError("Invalid trainable parameter scope")
                    native_schedule = [{key: r[key] for key in ("layer", "spatial_keys", "pointer_keys")}
                                       for r in observer.calls]
                    if native_schedule != schedule:
                        raise RuntimeError("Native read schedule changed")
                    if [(r["frame"], r["layer"]) for r in observer.calls] != [
                            (s["frame"], layer) for s in samples for layer in range(4)]:
                        raise RuntimeError("Missing native layer")
                    records.append({"step": step, "loss": float(loss.detach()), "terms": info["terms"],
                        "gradient_l1": gradients, "temporal_gradient_l1": float(temporal.abs().sum()),
                        "reads": observer.calls})
                    write(folder / f"{mode}.json", records)
                    if step < protocol["updates"]:
                        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                if any(not torch.isfinite(p).all() for p in heads.parameters()) or parameter_digest(heads) == initial:
                    raise RuntimeError("External module did not update finitely")
                torch.save(heads.state_dict(), folder / f"{mode}.pt")
                arms[mode] = {"initial_sha256": initial, "records": records,
                              "parameters": sum(p.numel() for p in heads.parameters())}
            if arms["regional"]["initial_sha256"] != arms["global"]["initial_sha256"]:
                raise RuntimeError("Control initialization differs")
            if [r["reads"] for r in arms["regional"]["records"]] != [r["reads"] for r in arms["global"]["records"]]:
                raise RuntimeError("Regional/control schedules differ")
            if banks != [fingerprint(s["output_dict"])["sha256"] for s in samples] or parameter_digest(model) != backbone:
                raise RuntimeError("Original parameters or input banks changed")
            results[sequence] = {"manifest_frames": [f.frame_index for f in frames], "arms": arms,
                                 "disabled_native_exact": True, "native_schedule": schedule}
        if hashes != {name: digest(ROOT / name) for name in names}:
            raise RuntimeError("Sources changed during verification")
        write(out / "completion.json", {"passed": True, "sequences": results,
            "backbone_sha256": backbone, "sources": hashes, "seconds": time.monotonic() - started,
            "scope": "Two short real-data clips; engineering and gradients only, not efficacy or full-video verification"})
        write(out / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

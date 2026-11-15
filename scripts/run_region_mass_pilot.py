"""Train and evaluate bounded regional attention-mass biases on PolypGen."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.native_region_readout import NativeRegionalReadout, RegionalMassHeads
from modeling.paired_readout_training import clip_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import save_tensor, write
from scripts.verify_on_demand_clip import capture_clip
import scripts.run_cfbi_region_pilot as propagation

PROTOCOL = ROOT / "protocols/region_mass_polypgen_split_v1.json"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class SeededReadout(NativeRegionalReadout):
    def __init__(self, model, heads, mode, sequence, seed):
        super().__init__(model, heads, mode, sequence, shuffle_seed=seed)

    @property
    def trace(self):
        return self.observer.trace


def positive_frames(frames):
    return [frame for frame in frames
            if (load_label_mask(frame.mask_path, [1]) == 1).any()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--engineering", action="store_true",
                         help="Two training clips and seq17 only; pipeline gate, not an efficacy result")
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        protocol = json.loads(args.protocol.read_text())
        gate = json.loads(args.gate.read_text())
        manifest = ROOT / protocol["data"]["manifest"]
        if not gate.get("passed") or gate["checkpoint_sha256"] != digest(args.checkpoint):
            raise RuntimeError("Real-data engineering gate/checkpoint mismatch")
        if gate.get("model_config") != MODEL_CONFIG:
            raise RuntimeError("Gate was not run with the registered Hiera-T configuration")
        if gate["manifest_sha256"] != digest(manifest):
            raise RuntimeError("Engineering gate manifest mismatch")

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(protocol["training"]["seed"])
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        train_names = protocol["data"]["training_sequences"]
        eval_names = protocol["data"]["evaluation_sequences"]
        epochs = protocol["training"]["epochs"]
        if args.engineering:
            train_names, eval_names, epochs = train_names[:2], ["seq17"], 1
        if args.engineering and "seq17" not in protocol["data"]["evaluation_sequences"]:
            raise RuntimeError("Engineering sequence is absent from the registered evaluation set")
        if set(train_names) & set(eval_names):
            raise RuntimeError("Training and evaluation sequences overlap")
        for name in train_names:
            if name not in videos or not positive_frames(videos[name]):
                raise RuntimeError(f"Training sequence has no positive masks: {name}")
        for name in eval_names:
            if name not in videos or not positive_frames(videos[name]):
                raise RuntimeError(f"Evaluation sequence has no positive masks: {name}")

        repo, revision = propagation.activate_author_repo()
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        source_paths = [args.protocol.resolve(), Path(__file__).resolve(),
                        ROOT / "modeling/region_mass_attention.py",
                        ROOT / "modeling/native_region_readout.py",
                        ROOT / "modeling/region_readout.py",
                        ROOT / "modeling/paired_readout_training.py",
                        ROOT / "modeling/cfbi_region_gate.py",
                        ROOT / "modeling/region_definitions.py",
                        ROOT / "modeling/region_tokens.py",
                        ROOT / "datasets/sam2_clip_sampler.py",
                        ROOT / "scripts/verify_region_mass_clip.py",
                        ROOT / "scripts/verify_on_demand_clip.py",
                        ROOT / "scripts/run_cfbi_region_pilot.py",
                        ROOT / "scripts/run_author_baseline.py",
                        ROOT / "workflows/catalog.py",
                        ROOT / "tests/test_region_mass_attention.py",
                        ROOT / "tests/test_sam2_clip_sampler.py"]
        source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
        setup = {"protocol": protocol, "protocol_sha256": digest(args.protocol),
                 "source_hashes": source_hashes, "checkpoint_sha256": digest(args.checkpoint),
                 "manifest_sha256": digest(manifest), "sam2_revision": revision,
                 "model_config": MODEL_CONFIG, "gate": str(args.gate),
                 "gate_sha256": digest(args.gate), "training_sequences": train_names,
                 "evaluation_sequences": eval_names, "epochs": epochs,
                 "engineering_only": args.engineering,
                 "patient_independence_verified": False}
        write(args.output_dir / "setup.json", setup)

        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        modes = ("regional", "shuffled")
        heads = {mode: RegionalMassHeads().train() for mode in modes}
        optimizers = {mode: torch.optim.Adam(
            heads[mode].parameters(), lr=protocol["training"]["learning_rate"])
            for mode in modes}
        records = []
        update = 0
        for epoch in range(epochs):
            for sequence_index, name in enumerate(train_names):
                seed = protocol["training"]["seed"] + epoch * len(train_names) + sequence_index
                frames = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                if not (load_label_mask(frames[0].mask_path, [1]) == 1).any():
                    raise RuntimeError(f"Sampler returned a nonpositive prompt: {name}")
                samples = capture_clip(model, frames, args.output_dir)
                row = {"epoch": epoch, "sequence": name, "seed": seed,
                       "frames": [frame.frame_index for frame in frames], "arms": {}}
                for mode in modes:
                    optimizer = optimizers[mode]
                    optimizer.zero_grad(set_to_none=True)
                    loss = clip_loss(model, heads[mode], samples, mode, name,
                                     check_native=(update == 0),
                                     shuffle_seed=protocol["training"]["seed"])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss for {mode}/{name}")
                    loss.backward()
                    grads = [float(layer.log_bias.grad.detach().abs().sum())
                             if layer.log_bias.grad is not None else 0.0
                             for layer in heads[mode].layers]
                    if not all(np.isfinite(value) for value in grads) or not any(value > 0 for value in grads):
                        raise RuntimeError(f"Missing/nonfinite regional gradients for {mode}/{name}")
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise RuntimeError("Frozen SAM2 acquired parameter gradients")
                    optimizer.step()
                    bound = float(np.log(2.0))
                    actual = [float((layer.max_log_scale * torch.tanh(layer.log_bias.detach())).abs().max())
                              for layer in heads[mode].layers]
                    if not all(np.isfinite(value) and value <= bound + 1e-6 for value in actual):
                        raise RuntimeError("Regional log-mass bound was exceeded")
                    row["arms"][mode] = {"loss": float(loss.detach()),
                                          "gradient_l1_by_layer": grads,
                                          "effective_abs_log_bias_by_layer": actual}
                    update += 1
                records.append(row)
                write(args.output_dir / "training.json", records)
                write(args.output_dir / "progress.json", {
                    "phase": "training", "clips_complete": len(records),
                    "clips_total": epochs * len(train_names),
                    "last_sequence": name, "epoch": epoch})
                del samples

        for mode in modes:
            folder = args.output_dir / mode
            folder.mkdir()
            save_tensor(heads[mode].state_dict(), folder / "head.pt")
            write(folder / "head_receipt.json", {"sha256": digest(folder / "head.pt"),
                                                  "mode": mode, "updates": update // 2})
        native_dir = args.output_dir / "native"
        native_dir.mkdir()
        native_heads = RegionalMassHeads()
        save_tensor(native_heads.state_dict(), native_dir / "head.pt")
        native_hash = digest(native_dir / "head.pt")
        del model, heads, optimizers, native_heads

        class PilotReadout(SeededReadout):
            def __init__(self, model, gate_head, mode, sequence):
                super().__init__(model, gate_head, mode, sequence, protocol["training"]["seed"])

        propagation.RegionGate = RegionalMassHeads
        propagation.NativeCFBIGate = PilotReadout
        results = {"native": {}, "conditions": {mode: {} for mode in modes},
                   "scope": protocol["status"]}
        arms = [("native", "generic", native_dir / "head.pt", native_hash),
                *((mode, mode, args.output_dir / mode / "head.pt",
                   digest(args.output_dir / mode / "head.pt")) for mode in modes)]
        for arm_name, mode, head_path, head_hash in arms:
            for name in eval_names:
                rows, summary = propagation.validate(args, videos[name], repo, mode, name,
                                                      head_path, head_hash)
                write(args.output_dir / arm_name / f"{name}_frames.json", rows)
                destination = results["native"] if arm_name == "native" else results["conditions"][arm_name]
                destination[name] = summary
                write(args.output_dir / "results.json", results)
                write(args.output_dir / "progress.json", {
                    "phase": "evaluation", "arm": arm_name,
                    "sequence": name, "completed_sequences": sum(
                        len(group) for group in [results["native"], *results["conditions"].values()])})

        if source_hashes != {str(path.relative_to(ROOT)): digest(path) for path in source_paths}:
            raise RuntimeError("Source code or protocol changed during the run")
        summary_table = []
        for arm_name, by_sequence in [("native", results["native"]),
                                      *results["conditions"].items()]:
            summary_table.append({"arm": arm_name,
                "mean_sequence_dice": float(np.mean([by_sequence[name]["present_dice"] for name in eval_names])),
                "mean_sequence_iou": float(np.mean([by_sequence[name]["iou"] for name in eval_names])),
                "mean_sequence_boundary_f": float(np.mean([by_sequence[name]["boundary_f"] for name in eval_names])),
                "false_positive_frames": int(sum(by_sequence[name]["gt_empty_false_alarm_count"] for name in eval_names))})
        results["summary"] = summary_table
        write(args.output_dir / "results.json", results)
        lines = ["# Regional attention-mass PolypGen split pilot", "",
                 protocol["status"], "", "| Arm | Mean sequence Dice | IoU | Boundary F | Empty-GT false-positive frames |",
                 "|---|---:|---:|---:|---:|"]
        for row in summary_table:
            lines.append(f"| {row['arm']} | {row['mean_sequence_dice']:.6f} | {row['mean_sequence_iou']:.6f} | {row['mean_sequence_boundary_f']:.6f} | {row['false_positive_frames']} |")
        lines += ["", "Per-sequence frame-level scores and memory traces are in the corresponding arm folders.",
                  "This is one exploratory seed on previously examined PolypGen sequences; it is not independent confirmation."]
        (args.output_dir / "results.md").write_text("\n".join(lines) + "\n")
        write(args.output_dir / "completion.json", {
            "complete": True, "training_updates_per_arm": update // 2,
            "evaluation_sequences": eval_names, "checkpoint_sha256": digest(args.checkpoint),
            "engineering_only": args.engineering,
            "protocol_sha256": digest(args.protocol), "elapsed_seconds": time.monotonic() - start,
            "result": "see results.json and results.md"})
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()

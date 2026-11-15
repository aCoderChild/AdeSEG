"""Real-sequence engineering gate for RegionContrast-supervised SAM2 Q/K."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_region_qk_training import enable_memory_qk_training
from modeling.region_contrast import (
    REGIONCONTRAST_REVISION,
    RegionContrastBank,
)
from modeling.region_contrast_training import region_contrast_clip_loss
from modeling.region_tokens import SAM2_REVISION
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_cfbi_region_affinity_cv import checked_native_run
from scripts.run_regioncontrast_memory_qk_cv import (
    _match_initial_states,
    run_sequence,
    train_arm,
)
from scripts.run_regional_refinement import write

PROTOCOL = ROOT / "protocols/regioncontrast_memory_qk_polypgen_v1.json"
SOURCES = [
    "protocols/regioncontrast_memory_qk_polypgen_v1.json",
    "modeling/region_contrast.py",
    "modeling/region_contrast_training.py",
    "modeling/cfbi_region_qk_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/verify_regioncontrast_memory_qk.py",
    "scripts/run_regioncontrast_memory_qk_cv.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/memory_token_io.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/RegionContrast/pyseg/models/decoder_contrast.py",
]


def _layer_gradient(model, names):
    parameters = dict(model.named_parameters())
    values = {index: 0.0 for index in range(4)}
    for name in names:
        grad = parameters[name].grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing or nonfinite q/k gradient: {name}")
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        values[layer] += float(grad.abs().sum())
    if any(value <= 0 for value in values.values()):
        raise RuntimeError(f"At least one attention layer has no Q/K gradient: {values}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(PROTOCOL.read_text())
    manifest = ROOT / protocol["data"]["manifest"]
    frames = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    sequence = "seq5"
    clip = sample_clip(
        frames[sequence], protocol["training"]["clip_length"],
        protocol["training"]["seed"],
    )
    repo, sam2_revision = activate_author_repo()
    if sam2_revision != SAM2_REVISION:
        raise RuntimeError("SAM2 source revision differs from the frozen assignment code")
    if REGIONCONTRAST_REVISION != protocol["prior_work"]["RegionContrast"]["revision"]:
        raise RuntimeError("RegionContrast revision differs from the frozen protocol")

    args.method = "sam2"
    args.model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    native, native_rows = checked_native_run(
        args.native_run,
        frames,
        digest(args.checkpoint),
        digest(manifest),
    )
    identity_model = build_author_predictor(args, repo).to(args.device).eval()
    identity_receipts = {}
    for identity_sequence in ("seq2", "seq3"):
        rows, summary = run_sequence(
            args,
            identity_model,
            frames[identity_sequence],
            identity_sequence,
            native_rows[identity_sequence],
            args.output_dir,
        )
        if any(
            row["prediction_sha256"] != row["native_prediction_sha256"]
            for row in rows
        ):
            raise RuntimeError(
                f"Untrained Q/K full propagation differs from native: {identity_sequence}"
            )
        identity_path = args.output_dir / f"identity_{identity_sequence}_frames.json"
        write(identity_path, rows)
        identity_receipts[identity_sequence] = {
            "frames": len(rows),
            "present_dice": summary["present_dice"],
            "gt_empty_false_alarm_count": summary["gt_empty_false_alarm_count"],
            "all_prediction_hashes_equal_native": True,
            "frames_sha256": digest(identity_path),
            "attention_layers_per_read": [
                row["attention_read"]["layers"] for row in rows
            ],
            "spatial_pointer_keys_per_read": [
                [row["attention_read"]["spatial_keys"],
                 row["attention_read"]["pointer_keys"]]
                for row in rows
            ],
        }
    del identity_model
    import gc
    gc.collect()

    smoke_protocol = {
        **protocol,
        "training": {**protocol["training"], "epochs": 1},
    }
    smoke_root = args.output_dir / "matched_training_smoke" / "fold1"
    smoke_root.mkdir(parents=True)
    smoke_results = {}
    smoke_states = {}
    smoke_schedules = {}
    for arm in ("qk_only", "regional"):
        torch.manual_seed(smoke_protocol["training"]["seed"] + 1)
        smoke_model = build_author_predictor(args, repo).to(args.device)
        arm_dir = smoke_root / arm
        arm_dir.mkdir()
        records, names, initial_state, banks = train_arm(
            smoke_model,
            frames,
            ["seq5"],
            smoke_protocol,
            arm,
            arm_dir,
            1,
        )
        if len(records) != 1 or not torch.isfinite(torch.tensor(records[0]["loss"])):
            raise RuntimeError(f"Invalid matched-control training smoke: {arm}")
        smoke_states[arm] = initial_state
        smoke_schedules[arm] = [
            (row["epoch"], row["sequence"], row["seed"], row["clip_frames"])
            for row in records
        ]
        smoke_results[arm] = {
            "loss": records[0]["loss"],
            "clip_frames": records[0]["clip_frames"],
            "qk_gradient_l1_by_layer": records[0]["qk_gradient_l1_by_layer"],
            "contrast_loss": records[0]["contrast_loss"],
            "trainable_qk_parameters": names,
            "queue_counts": [bank.count.tolist() for bank in banks]
            if banks is not None else None,
        }
        del smoke_model, banks
        gc.collect()
    if not _match_initial_states(smoke_states["qk_only"], smoke_states["regional"]):
        raise RuntimeError("Matched-training smoke arms had different initial Q/K weights")
    if smoke_schedules["qk_only"] != smoke_schedules["regional"]:
        raise RuntimeError("Matched-training smoke arms sampled different clips")
    if smoke_results["regional"]["contrast_loss"] is None or not all(
        count > 0
        for layer_counts in smoke_results["regional"]["queue_counts"]
        for count in layer_counts
    ):
        raise RuntimeError("Regional training smoke failed to exercise all role queues")
    write(args.output_dir / "matched_training_smoke.json", {
        "passed": True,
        "same_initial_qk": True,
        "same_clip_schedule": True,
        "fold": 1,
        "sequence": "seq5",
        "arms": smoke_results,
    })

    model = build_author_predictor(args, repo).to(args.device)
    qk_names, qk_parameters = enable_memory_qk_training(model)
    samples = capture_presence_clip(model, clip, args.output_dir)
    banks = [
        RegionContrastBank(
            capacity=protocol["training"]["contrast_queue_capacity_per_role_layer"],
            temperature=protocol["training"]["contrast_temperature"],
        ).to(args.device)
        for _ in range(4)
    ]
    optimizer = torch.optim.AdamW(
        qk_parameters,
        lr=protocol["training"]["qk_learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )

    optimizer.zero_grad(set_to_none=True)
    initial_loss, initial_records, trace = region_contrast_clip_loss(
        model, banks, samples, sequence,
        contrast_weight=protocol["training"]["contrast_weight"],
        check_native=True,
    )
    if not torch.isfinite(initial_loss):
        raise RuntimeError("Nonfinite initial RegionContrast loss")
    if not any(
        layer["contrast_roles"] > 0
        for frame in initial_records
        for layer in frame["layers"]
    ):
        raise RuntimeError("No query role had both positive and negative memory roles")
    initial_loss.backward()
    initial_gradient = _layer_gradient(model, qk_names)
    if not all(torch.isfinite(bank.queue).all() for bank in banks):
        raise RuntimeError("Nonfinite RegionContrast queue")
    if any(not bank.count.any() for bank in banks):
        raise RuntimeError("RegionContrast queue did not receive any region centers")
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    updated_loss, updated_records, updated_trace = region_contrast_clip_loss(
        model, banks, samples, sequence,
        contrast_weight=protocol["training"]["contrast_weight"],
    )
    updated_loss.backward()
    updated_gradient = _layer_gradient(model, qk_names)
    if not torch.isfinite(updated_loss):
        raise RuntimeError("Nonfinite post-update RegionContrast loss")
    if any(not torch.isfinite(parameter).all() for parameter in qk_parameters):
        raise RuntimeError("Nonfinite updated Q/K parameter")
    if len(trace) != len(samples) or len(updated_trace) != len(samples):
        raise RuntimeError("Engineering gate did not observe every clip read")
    if any(record["layers"] != [0, 1, 2, 3] for record in trace + updated_trace):
        raise RuntimeError("Engineering gate missed one or more attention layers")
    for record in trace + updated_trace:
        if (
            record["spatial_keys"] <= 0
            or record["spatial_keys"] % 4096
            or record["pointer_keys"] < 0
        ):
            raise RuntimeError("Invalid spatial or pointer key accounting")
    if [
        (record["spatial_keys"], record["pointer_keys"])
        for record in trace
    ] != [
        (record["spatial_keys"], record["pointer_keys"])
        for record in updated_trace
    ]:
        raise RuntimeError("Q/K update changed the native spatial/pointer key schedule")

    receipt = {
        "passed": True,
        "sequence": sequence,
        "clip_frames": [frame.frame_index for frame in clip],
        "sam2_revision": sam2_revision,
        "regioncontrast_revision": REGIONCONTRAST_REVISION,
        "full_propagation_native_identity": identity_receipts,
        "matched_training_smoke": "matched_training_smoke.json",
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(PROTOCOL),
        "sources": {path: digest(ROOT / path) for path in SOURCES},
        "initial_native_replay_max_abs_delta": 0.0,
        "initial_loss": float(initial_loss.detach()),
        "post_update_loss": float(updated_loss.detach()),
        "initial_qk_gradient_l1_by_layer": initial_gradient,
        "post_update_qk_gradient_l1_by_layer": updated_gradient,
        "queue_counts_by_layer": [bank.count.tolist() for bank in banks],
        "spatial_key_counts": sorted({record["spatial_keys"] for record in trace}),
        "pointer_key_counts": sorted({record["pointer_keys"] for record in trace}),
        "attention_layers_per_read": [record["layers"] for record in trace],
        "contrast_roles_per_frame_layer": [
            frame["layers"] for frame in initial_records
        ],
    }
    (args.output_dir / "completion.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

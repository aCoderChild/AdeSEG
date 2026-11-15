"""Engineering-only real-frame interruption/replay gate for durable checkpoints."""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets.video import load_manifest
from scripts.cohort_audit_checkpoints import (
    archive_incomplete_condition, load_condition_checkpoint, save_condition_checkpoint,
)
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_spatiotemporal_tome_value_exchange_cohort import load_protocol, preflight
from scripts.run_spatiotemporal_value_exchange_fullprop_gate import (
    CONDITIONS, MANIFEST, RecordingValueExchange, verify_fullprop_audits,
)
from scripts.run_wider_history_tome_gate import CHECKPOINT_DEFAULT, NATIVE_PROTOCOL, NATIVE_RESULTS, run_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kill-during-local", action="store_true")
    args = parser.parse_args()
    gate_bytes = args.gate.read_bytes()
    if hashlib.sha256(gate_bytes).hexdigest() != args.gate.with_suffix(args.gate.suffix + ".sha256").read_text().strip():
        raise RuntimeError("Engineering-gate protocol checksum differs")
    gate = json.loads(gate_bytes)
    for name, expected in gate["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Engineering-gate source changed: {name}")
    base_path = ROOT / gate["base_protocol"]
    if digest(base_path) != gate["base_protocol_sha256"]:
        raise RuntimeError("Base protocol changed")
    base = load_protocol(base_path)
    preflight(base)
    out = validate_output(args.output_dir)
    if not out.is_relative_to(Path(gate["output_root"]).resolve()):
        raise RuntimeError("Engineering output must be under its frozen Drive root")
    gate_hash = hashlib.sha256(gate_bytes).hexdigest()
    if args.resume:
        if json.loads((out / "setup.json").read_text()) != {"gate_sha256": gate_hash}:
            raise RuntimeError("Resume gate identity differs")
        if (out / "completion.json").exists():
            raise RuntimeError("Cannot resume a completed gate attempt")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "setup.json").write_text(json.dumps({"gate_sha256": gate_hash}) + "\n")

    torch.set_num_threads(base["runtime"]["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(base["runtime"]["seed"])
    videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}
    frames = videos[gate["sequence"]]
    native_data = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    prompt, native_rows, _ = native_reference(NATIVE_RESULTS.parent, native_data, native_protocol, gate["sequence"], frames)
    if prompt != gate["prompt"]:
        raise RuntimeError("Gate prompt differs")
    target_frames = list(range(prompt + 1, prompt + 1 + gate["target_count"]))

    if args.kill_during_local:
        def watchdog():
            progress = out / "progress.json"
            while True:
                try:
                    state = json.loads(progress.read_text())
                    if state.get("condition") == "local" and state.get("targets_done", 0) >= 1:
                        os.kill(os.getpid(), signal.SIGKILL)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(0.02)
        threading.Thread(target=watchdog, daemon=True).start()

    audits = {}
    for mode in CONDITIONS:
        folder = out / mode / gate["sequence"]
        if args.resume and (folder / "condition_checkpoint.json").exists():
            rows, calls, receipt = load_condition_checkpoint(folder, gate["sequence"], mode)
        else:
            if args.resume:
                archive_incomplete_condition(out, gate["sequence"], mode)
            calls = []
            rows, receipt = run_condition(
                "global", gate["sequence"], frames, prompt, native_rows, CHECKPOINT_DEFAULT, out,
                max_noncond_maps=6, allocation="uniform", condition_name=mode,
                end_frame=target_frames[-1],
                memory_reader_factory=lambda model, chosen=mode, sink=calls: RecordingValueExchange(model, mode=chosen, audit_sink=sink),
                native_exact_required=(mode == "noop"),
            )
            save_condition_checkpoint(folder, gate["sequence"], mode, rows, calls, receipt)
        if [x["frame"] for x in rows] != target_frames:
            raise RuntimeError("Gate target frames differ")
        if not receipt["parameter_hash_unchanged"] or not receipt["temporal_slots_restored"]:
            raise RuntimeError("Gate model invariants failed")
        audits[mode] = calls
    summary = verify_fullprop_audits(audits, target_frames)
    (out / "completion.json").write_text(json.dumps({"complete": True, "audit": summary}, indent=2) + "\n")
    print("Real-frame gate attempt complete", out, flush=True)


if __name__ == "__main__":
    main()

"""Verify frozen recovery evidence before launching the full cohort runner."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(path):
    path = Path(path)
    if sha(path) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise RuntimeError("Full-cohort protocol checksum differs")
    protocol = json.loads(path.read_text())
    expected_sequences = [f"seq{i}" for i in range(2, 24) if i != 7]
    if (protocol.get("status") != "frozen_for_cohort_efficacy"
        or protocol.get("sequences") != expected_sequences
        or protocol.get("unique_post_prompt_targets") != 2012
        or protocol.get("condition_frame_outputs") != 6036):
        raise RuntimeError("Protocol is not the frozen full 21-sequence cohort")
    for source, expected in protocol["source_hashes"].items():
        if sha(ROOT / source) != expected:
            raise RuntimeError(f"Frozen source differs: {source}")
    gates = protocol["recovery_gates"]
    if set(gates) != {"durable_main", "object_isolation"}:
        raise RuntimeError("Missing required recovery gate")
    verified = {}
    for name, gate in gates.items():
        receipt = Path(gate["receipt"])
        gate_protocol = ROOT / gate["protocol"]
        if sha(receipt) != gate["receipt_sha256"] or sha(gate_protocol) != gate["protocol_sha256"]:
            raise RuntimeError(f"Recovery gate evidence differs: {name}")
        evidence = json.loads(receipt.read_text())
        if evidence.get("status") != "passed" or not all(evidence["checks"].values()):
            raise RuntimeError(f"Recovery gate did not pass: {name}")
        evidence_protocol = evidence.get("protocol_sha256", evidence.get("gate_sha256"))
        if evidence_protocol != gate["protocol_sha256"]:
            raise RuntimeError(f"Recovery gate is bound to a different protocol: {name}")
        gp = json.loads(gate_protocol.read_text())
        for source, expected in gp["source_hashes"].items():
            if sha(ROOT / source) != expected:
                raise RuntimeError(f"Recovery gate source differs: {source}")
        verified[name] = gate["receipt_sha256"]
    main_gate = json.loads((ROOT / gates["durable_main"]["protocol"]).read_text())
    for source in ("scripts/run_spatiotemporal_tome_value_exchange_durable.py", "scripts/cohort_audit_checkpoints.py"):
        if main_gate["source_hashes"][source] != protocol["source_hashes"][source]:
            raise RuntimeError(f"Durable runner differs from real-frame gate: {source}")
    return protocol, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol, gates = check_protocol(args.protocol)
    if args.check_only:
        print(json.dumps({"status": "passed", "sequences": len(protocol["sequences"]), "gates": gates}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is set")
    argv = [sys.executable, str(ROOT / "scripts/run_spatiotemporal_tome_value_exchange_durable.py"),
            "--protocol", str(args.protocol), "--output-dir", str(args.output_dir)]
    if args.resume:
        argv.append("--resume")
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()

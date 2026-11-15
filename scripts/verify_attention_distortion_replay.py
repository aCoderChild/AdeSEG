"""Require exact target/measurement equality across two observer processes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output
from scripts.run_regional_refinement import write
from scripts.run_attention_distortion_diagnostic import observer_settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    first, second = args.first.resolve(), args.second.resolve()
    receipt_path = validate_output(args.receipt)
    if first == second or not first.is_dir() or not second.is_dir():
        raise ValueError("Provide two distinct, completed Drive attempt directories")
    a_completion = json.loads((first / "completion.json").read_text())
    b_completion = json.loads((second / "completion.json").read_text())
    a_setup = json.loads((first / "setup.json").read_text())
    b_setup = json.loads((second / "setup.json").read_text())
    if not a_completion.get("passed") or not b_completion.get("passed"):
        raise RuntimeError("Both attempts must have passed their frozen integrity checks")
    if a_setup.get("protocol_sha256") != b_setup.get("protocol_sha256"):
        raise RuntimeError("Replay attempts used different frozen protocols")
    if a_setup.get("source_hashes") != b_setup.get("source_hashes"):
        raise RuntimeError("Replay attempts used different source hashes")
    a_path, b_path = first / "results.json", second / "results.json"
    for folder, completion, path in ((first, a_completion, a_path),
                                     (second, b_completion, b_path)):
        if completion.get("results_sha256") != sha256(path):
            raise RuntimeError(f"Result hash differs from completion receipt: {folder}")
    settings = observer_settings(a_setup["protocol"])
    expected_conditions = settings["conditions"]
    a_results, b_results = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    if a_results != b_results:
        raise RuntimeError("Target, tensor hashes, or attention measurements differ across processes")
    targets = []
    for sequence in ("seq2", "seq3"):
        rows = a_results.get(sequence)
        if not isinstance(rows, list) or len(rows) != 3:
            raise RuntimeError(f"Expected three frozen targets for {sequence}")
        for row in rows:
            if not row.get("native_decoder_parity") or not row.get("native_bank_unchanged"):
                raise RuntimeError(f"Native invariants failed at {sequence}/{row.get('frame')}")
            layers = row.get("layers", [])
            if [layer.get("layer") for layer in layers] != [0, 1, 2, 3]:
                raise RuntimeError(f"Layer coverage differs at {sequence}/{row.get('frame')}")
            if any(len(layer.get("conditions", [])) != len(expected_conditions) for layer in layers):
                raise RuntimeError(f"Condition coverage differs at {sequence}/{row.get('frame')}")
            for layer in layers:
                for condition, expected in zip(layer["conditions"], expected_conditions):
                    if any(condition.get(k) != v for k, v in expected.items()):
                        raise RuntimeError("Condition identity/order differs from protocol")
                    if settings["calibration_shadow"]:
                        shadow = condition.get("calibrated_shadow", {})
                        hashes = shadow.get("shared_merge_input_sha256", [])
                        if (not shadow.get("returns_native_output") or len(hashes) != 3
                                or any(len(h) != 64 for h in hashes)):
                            raise RuntimeError("Shared calibration input evidence missing")
            targets.append({"sequence": sequence, "frame": row["frame"],
                            "object_id": row["object_id"]})
    if set(a_results) != {"seq2", "seq3"}:
        raise RuntimeError("Unexpected sequence in diagnostic results")

    receipt = {
        "passed": True,
        "scope": "Independent-process exact replay of bounded shadow-attention measurements",
        "protocol_sha256": a_setup["protocol_sha256"],
        "source_hashes_sha256": hashlib.sha256(json.dumps(
            a_setup["source_hashes"], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "attempts": [str(first), str(second)],
        "results_sha256": [sha256(a_path), sha256(b_path)],
        "parsed_results_exactly_equal": True,
        "native_outputs_and_banks_verified_in_both": True,
        "target_count": len(targets),
        "layer_count": 4,
        "conditions_per_layer": len(expected_conditions),
        "calibration_shadow": settings["calibration_shadow"],
        "verifier_sha256": sha256(Path(__file__)),
        "targets": targets,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

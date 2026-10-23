"""Identify byte/structure/alias changes behind the failed legacy bank digest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_contiguous_exclusions as runner  # noqa: E402
from scripts import diagnose_contiguous_exclusions as diagnosis  # noqa: E402


def tensor_manifest(value, path="state", result=None):
    result = {} if result is None else result
    if isinstance(value, runner.torch.Tensor):
        raw = (
            value.detach()
            .contiguous()
            .reshape(-1)
            .view(runner.torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        result[path] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data_ptr": value.data_ptr(),
        }
    elif isinstance(value, dict):
        result[path] = {
            "keys": [f"{type(k).__name__}:{k}" for k in sorted(value, key=str)]
        }
        for key in sorted(value, key=str):
            tensor_manifest(value[key], f"{path}/{type(key).__name__}:{key}", result)
    elif isinstance(value, (tuple, list)):
        result[path] = {"type": type(value).__name__, "length": len(value)}
        for index, item in enumerate(value):
            tensor_manifest(item, f"{path}/{index}", result)
    else:
        result[path] = {"type": type(value).__name__, "value": str(value)}
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    original_digest, original_infer = runner.bank_digest, runner.infer
    before = None
    calls = 0

    def detailed_digest(state, frame, units, horizon):
        nonlocal before
        wanted = {u["frame"] for u in units} | set(
            range(max(0, frame - horizon), frame + 1)
        )
        outputs = state["output_dict_per_obj"][0]
        selected = {
            "conditioning": outputs["cond_frame_outputs"],
            "recent": {
                k: v
                for k, v in outputs["non_cond_frame_outputs"].items()
                if k in wanted
            },
            "image_cache": state["cached_features"],
            "frame_keys": {k: sorted(v) for k, v in outputs.items()},
        }
        manifest = tensor_manifest(selected)
        legacy = original_digest(state, frame, units, horizon)
        if before is None:
            before = manifest
            label = "before"
        else:
            label = "after"
            changed = {
                k: {"before": before.get(k), "after": manifest.get(k)}
                for k in set(before) | set(manifest)
                if before.get(k) != manifest.get(k)
            }
            (args.output_dir / "buffer_changes.json").write_text(
                json.dumps(changed, indent=2) + "\n"
            )
        (args.output_dir / f"bank_{label}.json").write_text(
            json.dumps(
                {"frame": frame, "legacy_digest": legacy, "manifest": manifest},
                indent=2,
            )
            + "\n"
        )
        return legacy

    def traced_infer(model, state, frame):
        nonlocal calls
        calls += 1
        before_call = tensor_manifest(
            {"bank": state["output_dict_per_obj"], "cache": state["cached_features"]}
        )
        result = original_infer(model, state, frame)
        after_call = tensor_manifest(
            {"bank": state["output_dict_per_obj"], "cache": state["cached_features"]}
        )
        changes = {
            k: {"before": before_call.get(k), "after": after_call.get(k)}
            for k in set(before_call) | set(after_call)
            if before_call.get(k) != after_call.get(k)
        }
        with (args.output_dir / "buffer_calls.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"call": calls, "frame": frame, "changes": changes}) + "\n"
            )
        return result

    runner.bank_digest = detailed_digest
    runner.infer = traced_infer
    diagnosis.main()


if __name__ == "__main__":
    main()

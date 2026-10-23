"""Exact, alias-independent content/layout fingerprints of native read buffers."""

import hashlib
import json

import torch


def bank_contents(state, frame, units, horizon):
    outputs = state["output_dict_per_obj"][0]
    wanted = {u["frame"] for u in units} | set(
        range(max(0, frame - horizon), frame + 1)
    )
    return {
        "conditioning": outputs["cond_frame_outputs"],
        "recent": {
            k: v for k, v in outputs["non_cond_frame_outputs"].items() if k in wanted
        },
        "image_cache": state["cached_features"],
        "frame_keys": {k: sorted(v) for k, v in outputs.items()},
    }


def fingerprint(value):
    """Hash every tensor path, not storage addresses or alias multiplicity."""
    manifest, addresses = {}, {}

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            raw = (
                item.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            manifest[path] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "stride": list(item.stride()),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            addresses[path] = item.data_ptr()
        elif isinstance(item, dict):
            manifest[path] = {
                "keys": [f"{type(k).__name__}:{k}" for k in sorted(item, key=str)]
            }
            for key in sorted(item, key=str):
                visit(item[key], path + "/" + f"{type(key).__name__}:{key}")
        elif isinstance(item, (tuple, list)):
            manifest[path] = {"type": type(item).__name__, "length": len(item)}
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
        else:
            manifest[path] = {"type": type(item).__name__, "value": str(item)}

    visit(value, "state")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest": manifest,
        "addresses": addresses,
    }


def changed_paths(before, after):
    return {
        k: {"before": before["manifest"].get(k), "after": after["manifest"].get(k)}
        for k in sorted(set(before["manifest"]) | set(after["manifest"]))
        if before["manifest"].get(k) != after["manifest"].get(k)
    }

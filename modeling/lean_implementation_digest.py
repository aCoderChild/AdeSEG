"""Source fingerprint for the NEW structured Lean-SAM2 adapter path.

The legacy v1 refiner retains its historical single-file source check.
New experiments additionally authenticate all computation and routing blocks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

FILES = (
    "multiscale_channel_value_refiner_v1.py",
    "structured_feature_aggregation.py",
    "feature_aggregation.py",
    "aggregation_blocks/__init__.py",
    "aggregation_blocks/projection.py",
    "aggregation_blocks/attention.py",
    "aggregation_blocks/contrast.py",
    "aggregation_blocks/fusion.py",
    "lean_channel_value_routing_v1.py",
    "contrast_refinement_routing.py",
)


def implementation_digest():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for relative in FILES:
        h.update(relative.encode("utf-8"))
        h.update(b"\x00")
        h.update((root / relative).read_bytes())
    return h.hexdigest()

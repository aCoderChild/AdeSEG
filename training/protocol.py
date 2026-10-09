"""Validation for an explicit video-level experiment protocol."""

from __future__ import annotations

import json
from pathlib import Path


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {"training_videos", "validation_videos", "exploratory_videos", "locked_test"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Protocol is missing: {sorted(missing)}")
    roles = ("training_videos", "validation_videos", "exploratory_videos")
    groups = {role: set(protocol[role]) for role in roles}
    if any(not group for group in groups.values()):
        raise ValueError("Training, validation and exploratory video sets must not be empty.")
    if any(groups[left] & groups[right] for index, left in enumerate(roles) for right in roles[index + 1:]):
        raise ValueError("Protocol video roles must be disjoint.")
    locked = protocol["locked_test"]
    if locked is not None and set(locked) & set().union(*groups.values()):
        raise ValueError("The locked test set must be disjoint from every development role.")
    return protocol


def verify_training(protocol: dict, train_videos, validation_videos) -> None:
    actual = {"training_videos": set(train_videos), "validation_videos": set(validation_videos)}
    for role, videos in actual.items():
        expected = set(protocol[role])
        if videos != expected:
            raise ValueError(f"{role} must match the protocol exactly; expected {sorted(expected)}, got {sorted(videos)}.")


def evaluation_role(protocol: dict, videos) -> str:
    selected = set(videos)
    for role, name in (("validation", "validation_videos"), ("exploratory", "exploratory_videos")):
        if selected == set(protocol[name]):
            return role
    locked = protocol["locked_test"]
    if locked is not None and selected == set(locked):
        return "locked_test"
    raise ValueError("Selected videos do not exactly match a declared protocol role.")

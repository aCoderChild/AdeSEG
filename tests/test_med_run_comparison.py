"""Synthetic receipts validate matching, provenance and zero-adapter parity."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from datasets.video import ManifestFrame
from scripts.infer_med_video import infer_sequence, sha256
from scripts.compare_med_runs import compare_runs


class FixedPredictor:
    def __init__(self, negative=False):
        self.negative = negative

    def init_state(self, **kwargs):
        return {}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        assert frame_idx == 0 and obj_id == 1

    def propagate_in_video(self, state, start_frame_idx, reverse):
        assert start_frame_idx == 0 and not reverse
        yield 0, [1], torch.ones(1, 1, 4, 4)
        yield 1, [1], (-1 if self.negative else 1) * torch.ones(1, 1, 4, 4)


def fixture_runs(tmp_path, *, changed=False):
    frames, records = [], []
    for index in (0, 1):
        image, mask = tmp_path / f"{index}.jpg", tmp_path / f"{index}.png"
        Image.fromarray(np.full((4, 4, 3), 80, np.uint8)).save(image)
        Image.fromarray(np.ones((4, 4), np.uint8)).save(mask)
        frames.append(ManifestFrame("case1", index, image, mask))
        records.append({"split": "dev", "video_id": "case1", "frame_index": index,
                        "image": image.name, "mask": mask.name})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in records))
    paths = []
    for mode, predictor in (("native", FixedPredictor()),
                            ("zero_initialized_control", FixedPredictor(negative=changed))):
        root = tmp_path / mode
        root.mkdir()
        output = infer_sequence(predictor, frames, root, record_logits=True)
        record = {
            "protocol": "oracle_first_present_mask_post_prompt_v1",
            "adapter_mode": mode,
            "adapter_sha256": None,
            "model_config": "test",
            "backbone_sha256": "a" * 64,
            "device": "cpu",
            "postprocessing": "no_hole_filling",
            "split": "dev", "label_id": 1,
            "manifest_sha256": sha256(manifest),
            "results": {"case1": output},
        }
        (root / "run.json").write_text(json.dumps(record))
        paths.append(root)
    return manifest, *paths


def test_zero_control_requires_matching_masks_and_logits(tmp_path):
    manifest, native, zero = fixture_runs(tmp_path)
    result = compare_runs(manifest, native, zero, require_logit_identity=True)
    assert result["identical_binary_masks"]
    assert result["identical_float32_logits"]
    assert result["post_prompt_frames"] == 1
    assert result["paired_present_dice_mean_delta"] == 0


def test_zero_control_detects_logit_and_mask_change(tmp_path):
    manifest, native, zero = fixture_runs(tmp_path, changed=True)
    with pytest.raises(ValueError, match="parity failed"):
        compare_runs(manifest, native, zero, require_logit_identity=True)


def test_comparison_rejects_mismatched_protocol(tmp_path):
    manifest, native, zero = fixture_runs(tmp_path)
    record_path = zero / "run.json"
    record = json.loads(record_path.read_text())
    record["model_config"] = "wrong"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_runs(manifest, native, zero)

def test_comparison_requires_authenticated_raw_logits(tmp_path):
    manifest, native, zero = fixture_runs(tmp_path)
    record = json.loads((zero / "run.json").read_text())
    del record["results"]["case1"]["predictions"][0]["logit_file"]
    (zero / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="float32 tensors|missing them"):
        compare_runs(manifest, native, zero, require_logit_identity=True)

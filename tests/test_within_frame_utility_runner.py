"""Bookkeeping integration with deterministic fake inference, not a SAM2 gate."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
import numpy as np

from scripts import run_within_frame_token_utility as runner


def test_sequence_resume_rebuilds_identical_outputs_and_repeats_altered_arms(tmp_path, monkeypatch):
    frames = [SimpleNamespace(frame_index=i) for i in range(5)]
    outputs = {i: {"pred_masks": torch.full((1, 1, 2, 2), float(i)),
                   "obj_ptr": torch.tensor([[float(i)]]),
                   "object_score_logits": torch.tensor([[1.]])} for i in range(5)}
    bank = {"cond_frame_outputs": {0: outputs[0]},
            "non_cond_frame_outputs": {i: outputs[i] for i in range(1, 5)}}
    state = {"output_dict_per_obj": {0: bank}}
    replayed = []

    class Observer:
        def __init__(self, model):
            self.calls = []
            self.arm, self.source_frame = "native", None
            model.observer = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def trace(frame):
        ids = list(range(frame))
        spans = [{"frame": i, "t_pos": j, "start": j*4096, "stop": (j+1)*4096} for j, i in enumerate(ids)]
        return {"memory_frames": spans, "frame": frame}

    class Model:
        max_obj_ptrs_in_encoder = 16
        def propagate_in_video(self, state, start_frame_idx):
            for i in range(5):
                if i:
                    self.observer.calls.append(trace(i))
                yield i, [1], outputs[i]["pred_masks"]
        def _get_orig_video_res_output(self, state, value):
            return None, value

    def replay(model, sample, observer):
        frame = sample["frame"]
        replayed.append((frame, observer.arm))
        result = {k: v.clone() for k, v in outputs[frame].items()}
        if observer.arm != "native":
            result["pred_masks"] -= .25
        t = trace(frame)
        ids = [s["frame"] for s in t["memory_frames"]]
        t["selection_layers"] = [{"frame_ids": ids,
            "counts": [4096-int(runner.removal_plan(observer.arm).sum())
                       if observer.arm != "native" and f == observer.source_frame else 4096 for f in ids]} for _ in range(4)]
        observer.calls.append(t)
        return result, t

    def scored(full, frames, frame, folder, trace):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f"{frame:06d}.png"
        Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(path)
        return {"frame": frame, "source_frame_index": frame, "present": True, "absent_fp": None,
                "scores": {m: float(full.mean())/5 for m in ("dice", "iou", "boundary_f")},
                "mask_sha256": runner.digest(path), "prediction_sha256": runner.tensor_hash(full), "read": trace}

    monkeypatch.setattr(runner, "WithinFrameTokenRead", Observer)
    monkeypatch.setattr(runner.base, "prepare", lambda *args: (state, 0))
    monkeypatch.setattr(runner, "parameter_digest", lambda model: "frozen")
    monkeypatch.setattr(runner, "capture_training_read", lambda model, state, frame: {"frame": frame, "output_dict": bank})
    monkeypatch.setattr(runner, "bank_contents", lambda *args: bank)
    monkeypatch.setattr(runner, "check_layout", lambda a, b: None)
    monkeypatch.setattr(runner, "replay", replay)
    monkeypatch.setattr(runner, "scored", scored)
    core = {"setup_sha256": "fixed", "target_plan": {"seq2": {"prompt": 0, "targets": [3, 4]}}}
    first, followups, checks = runner.sequence(None, {"seq2": frames}, "seq2", Model(), tmp_path, core)
    assert len(first) == 2 and checks == 0
    replayed.clear()
    second, later, checks = runner.sequence(None, {"seq2": frames}, "seq2", Model(), tmp_path, core)
    assert first == second and followups == later
    assert checks == 3  # two main targets plus the available next-query replay
    assert sum(arm == "tile_00" for _, arm in replayed) == 3
    assert all(arm in ("native", "tile_00") for _, arm in replayed)
    # If regeneration changes, refuse before overwriting the original masks.
    saved_mask = tmp_path/"sequences/seq2/native_masks/000001.png"
    saved_bytes = saved_mask.read_bytes()
    outputs[1]["pred_masks"] += .1
    with pytest.raises(RuntimeError, match="Regenerated native trajectory changed"):
        runner.sequence(None, {"seq2": frames}, "seq2", Model(), tmp_path, core)
    assert saved_mask.read_bytes() == saved_bytes

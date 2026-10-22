"""All-frame coverage contracts; these are not full-dataset scientific results."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.audit_memory_token_regions import (
    audit_full_frame,
    assignment_scores,
    cohort_inventory,
)
from scripts import memory_token_io


def test_full_cohort_accounts_for_unpromptable_and_pre_prompt_frames(monkeypatch):
    from scripts import audit_memory_token_regions as audit

    videos = {
        f"seq{i}": [SimpleNamespace(frame_index=j) for j in range(4)]
        for i in range(1, 24)
    }

    def prompt(frames, protocol):
        if frames is videos["seq1"] or frames is videos["seq7"]:
            raise ValueError("No eligible labeled prompt under the selected protocol")
        return 2

    monkeypatch.setattr(audit, "prompt_index", prompt)
    cohort = cohort_inventory(videos)
    assert len(cohort) == 23
    assert cohort["seq1"]["native_memory_frames"] == 0
    assert cohort["seq7"]["gt_only_frames"] == 4
    assert cohort["seq2"]["native_memory_frames"] == 2
    assert (
        sum(x["native_memory_frames"] + x["gt_only_frames"] for x in cohort.values())
        == 92
    )
    with pytest.raises(ValueError, match="all 23"):
        cohort_inventory({"seq2": videos["seq2"]})


def test_all_frame_saving_deduplicates_vectors_but_retains_exact_indices(tmp_path):
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 1
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(np.repeat(mask[..., None] * 255, 3, axis=2)).save(image_path)
    Image.fromarray(mask).save(mask_path)
    frame = SimpleNamespace(
        video_id="seq2", frame_index=5, image_path=image_path, mask_path=mask_path
    )
    memory = torch.randn(1, 64, 64, 64).bfloat16().float()
    torch.save({"features": memory.bfloat16()}, tmp_path / "native_memory.pt")
    rows = audit_full_frame(
        frame, memory, 0, mask.astype(bool), tmp_path, [0.01, 0.02, 0.04]
    )
    assert len(rows) == 9
    for file in tmp_path.glob("frame_*.pt"):
        saved = torch.load(file, weights_only=True)
        assert saved["memory_row"] == 0 and saved["memory_available"]
        assert "groups" not in saved and "vectors" not in saved
        indices = torch.cat(list(saved["indices"].values()))
        assert torch.equal(indices.sort().values, torch.arange(4096))
        assert saved["vectors_file"] == "native_memory.pt"
    assert len(list(tmp_path.glob("*ratio*.png"))) == 6
    assert all(sum(row["counts"].values()) == 4096 for row in rows if "counts" in row)
    assert all(
        torch.tensor(row["confusion"]).diag().sum() == 4096
        for row in rows
        if "confusion" in row
    )


def test_gt_only_frame_never_claims_native_vectors(tmp_path):
    path = tmp_path / "empty.png"
    Image.fromarray(np.zeros((64, 64), np.uint8)).save(path)
    frame = SimpleNamespace(
        video_id="seq1", frame_index=0, image_path=path, mask_path=path
    )
    rows = audit_full_frame(frame, None, None, None, tmp_path, [0.02])
    assert len(rows) == 1 and rows[0]["memory_available"] is False
    saved = torch.load(next(tmp_path.glob("frame_*.pt")), weights_only=True)
    assert saved["vectors_file"] is None and saved["memory_row"] is None
    assert saved["labels"].eq(0).all()


def test_agreement_metrics_do_not_replace_missing_classes_with_perfect_scores():
    scores = assignment_scores(torch.zeros(3, 3))
    assert scores["mean_class_iou"] is None
    assert scores["regions"]["boundary"]["iou"] is None
    assert assignment_scores(torch.eye(3))["mean_class_iou"] == 1


def test_conditioning_memory_capture_uses_native_conditioning_bucket(
    tmp_path, monkeypatch
):
    frames = [
        SimpleNamespace(video_id="seq2", frame_index=i, mask_path=i) for i in range(2)
    ]
    memory = torch.ones(1, 64, 4, 4)
    monkeypatch.setattr(memory_token_io, "prompt_index", lambda *args: 0)
    monkeypatch.setattr(memory_token_io, "stage_frames", lambda *args: None)
    monkeypatch.setattr(
        memory_token_io, "load_label_mask", lambda *args: np.ones((8, 8), np.uint8)
    )
    monkeypatch.setattr(memory_token_io, "synchronize_storage", lambda *args: None)

    class Predictor:
        def init_state(self, **kwargs):
            return {
                "output_dict": {
                    "cond_frame_outputs": {0: {"maskmem_features": memory}},
                    "non_cond_frame_outputs": {1: {"maskmem_features": memory * 2}},
                }
            }

        def add_new_mask(self, *args, **kwargs):
            pass

        def propagate_in_video(self, *args, **kwargs):
            for i in range(2):
                yield i, [1], torch.ones(1, 1, 8, 8)

        def reset_state(self, state):
            pass

    captured, records = memory_token_io.extract_memory_features(
        Predictor(), frames, {"seq2": frames}, "cpu", tmp_path, include_prompt=True
    )
    assert torch.equal(captured[0], memory[0]) and torch.equal(
        captured[1], memory[0] * 2
    )
    assert [r["frame"] for r in records] == [0, 1]


def test_full_runner_visits_every_sequence_and_frame(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from scripts import audit_memory_token_regions as audit

    expected = json.loads(
        (audit.ROOT / "protocols/memory_token_regions.json").read_text()
    )["model"]["checkpoint_sha256"]
    videos = {
        f"seq{i}": [
            SimpleNamespace(
                video_id=f"seq{i}",
                frame_index=j,
                image_path=tmp_path / "image",
                mask_path=tmp_path / "mask",
            )
            for j in range(2)
        ]
        for i in range(1, 24)
    }

    def prompt(frames, protocol):
        if frames[0].video_id in {"seq1", "seq7"}:
            raise ValueError("No eligible labeled prompt under the selected protocol")
        return 0

    monkeypatch.setattr(audit, "prompt_index", prompt)
    monkeypatch.setattr(audit, "digest", lambda *args: expected)
    monkeypatch.setattr(audit, "validate_output", lambda path: path)
    monkeypatch.setattr(
        audit, "build_video_predictor", lambda *args: torch.nn.Linear(1, 1)
    )

    def capture(
        model,
        frames,
        videos,
        device,
        output,
        mask_output=None,
        include_prompt=False,
        progress=None,
    ):
        assert include_prompt
        if mask_output is not None:
            for frame in frames:
                mask_output[(frame.video_id, frame.frame_index)] = torch.zeros(
                    8, 8, dtype=torch.bool
                )
        return torch.zeros(len(frames), 64, 64, 64), [
            {"frame": f.frame_index} for f in frames
        ]

    monkeypatch.setattr(audit, "extract_memory_features", capture)
    visited = []

    def assign(frame, memory, row, predicted, folder, ratios):
        visited.append((frame.video_id, frame.frame_index))
        result = []
        for ratio in ratios:
            for source in (
                ["ground_truth", "predicted"]
                if memory is not None
                else ["ground_truth"]
            ):
                result.append(
                    {
                        "frame": frame.frame_index,
                        "source": source,
                        "ratio": ratio,
                        "counts": {
                            "background": 4096,
                            "foreground_interior": 0,
                            "boundary": 0,
                        },
                    }
                )
            if memory is not None:
                result.append(
                    {
                        "frame": frame.frame_index,
                        "ratio": ratio,
                        "confusion": [[4096, 0, 0], [0, 0, 0], [0, 0, 0]],
                    }
                )
        return result

    monkeypatch.setattr(audit, "audit_full_frame", assign)

    class Artifact:
        def __init__(self, *args, **kwargs):
            pass

        def add_dir(self, path):
            assert Path(path).is_dir()

        def add_file(self, *args, **kwargs):
            pass

    class Run:
        id = "test"
        summary = {}
        artifacts = 0

        def log(self, data):
            pass

        def log_artifact(self, artifact):
            self.artifacts += 1
            return SimpleNamespace(wait=lambda: None)

        def finish(self, **kwargs):
            pass

    run = Run()
    monkeypatch.setattr(audit, "start_wandb", lambda *args: run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    output = tmp_path / "full"
    args = SimpleNamespace(
        checkpoint=tmp_path / "checkpoint",
        manifest=tmp_path / "manifest",
        device="cpu",
        output_dir=output,
    )
    audit.run_full_validation(args, videos)
    summary = json.loads((output / "summary.json").read_text())
    assert len(visited) == len(set(visited)) == 46
    assert summary["completed"] and summary["all_dataset_frames_audited"]
    assert summary["native_memory_frames"] == 42 and summary["gt_only_frames"] == 4
    assert summary["unpromptable_sequences"] == ["seq1", "seq7"]
    assert run.artifacts == 24

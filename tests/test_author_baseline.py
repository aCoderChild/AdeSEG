"""Output adapters must follow each author's actual API, not assume parity."""

import ast
from pathlib import Path

import pytest

from scripts.run_author_baseline import (
    REPOSITORIES,
    iter_predictions,
    source_identity,
    replay_difference,
    write_results,
)


class Predictor:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def propagate_in_video(self, state, **kwargs):
        self.calls.append((state, kwargs))
        return self.output


def test_native_generator_keeps_frame_indices_and_logits():
    predictor = Predictor(iter([(4, [1], "first"), (5, [1], "second")]))
    assert list(iter_predictions(predictor, {}, "sam2", 4)) == [
        (4, [1], "first"),
        (5, [1], "second"),
    ]
    assert predictor.calls[0][1] == {"start_frame_idx": 4, "reverse": False}


def test_long_trajectory_uses_author_selected_masks_with_prompt_offset():
    predictor = Predictor(([1], ["prompt", "next", "last"]))
    assert list(iter_predictions(predictor, {"num_frames": 7}, "sam2long", 4)) == [
        (4, [1], "prompt"),
        (5, [1], "next"),
        (6, [1], "last"),
    ]


@pytest.mark.parametrize("output", [([1], ["short"]), ([2], [1, 2, 3])])
def test_long_rejects_missing_frames_or_wrong_object(output):
    with pytest.raises(RuntimeError, match="coverage"):
        list(iter_predictions(Predictor(output), {"num_frames": 3}, "sam2long", 0))


def test_author_pins_and_long_cli_defaults_match_recorded_values():
    root = Path(__file__).resolve().parents[1]
    for method in REPOSITORIES:
        source_identity(method)
    tree = ast.parse((root / "external/SAM2Long/tools/vos_inference.py").read_text())
    expected = {"--num_pathway": 3, "--iou_thre": 0.3, "--uncertainty": 1}
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value in expected:
            found[node.args[0].value] = next(
                ast.literal_eval(k.value) for k in node.keywords if k.arg == "default"
            )
    assert found == expected


def test_replay_reports_mask_changes_without_hiding_logit_differences(tmp_path):
    import torch

    torch.save(torch.tensor([-0.01, 2.0]), tmp_path / "logits_000004.pt")
    torch.save(torch.tensor([0.01, 2.0]), tmp_path / "repeat_logits_000004.pt")
    row = replay_difference(tmp_path, [{"frame": 4}])[0]
    assert not row["exact"] and row["changed_mask_pixels"] == 1
    assert row["max_absolute_logit_difference"] > 0


def test_report_does_not_assign_scores_to_unpromptable_sequences(tmp_path):
    write_results(
        tmp_path,
        [
            {"sequence": "seq1", "status": "unpromptable", "frames": 0},
            {
                "sequence": "seq2",
                "status": "completed",
                "frames": 7,
                "present_dice": 0.7,
                "all_frame_dice": 0.8,
            },
        ],
        "smoke",
    )
    report = (tmp_path / "results.md").read_text()
    assert "N/A: no prompt" in report and "not efficacy estimates" in report
    assert "0.70000" in report

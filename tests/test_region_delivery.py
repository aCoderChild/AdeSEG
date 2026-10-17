import json

from scripts.analyze_region_token_pilot import analyze
from scripts.collect_region_results import validate


def test_missing_or_failed_results_never_validate(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    receipt = validate(tmp_path)
    assert not receipt["delivery_valid"]
    assert not receipt["full_study_complete"]
    assert "No E1 records" in receipt["problems"]


def test_pilot_empty_or_incomplete_sequence_coverage_never_passes():
    result = analyze([], [f"seq{i}" for i in range(1, 11)])
    assert result["decision"] == "stop_or_pivot"
    assert not result["gate_pass"]
    assert not result["full_study_complete"]


def test_random_controls_match_feasible_category_frames_before_averaging():
    rows = [
        {
            "state": "complete",
            "video": "seq1",
            "frame": 1,
            "bank": "recent",
            "category": "near_background",
            "seed": 0,
            "harm": {"dice": 0.02, "boundary_f": 0.03},
            "attention": [],
        }
    ]
    for seed in range(5):
        for frame, harm in ((1, 0.0), (2, 0.9)):
            rows.append(
                {
                    "state": "complete",
                    "video": "seq1",
                    "frame": frame,
                    "bank": "recent",
                    "category": "random",
                    "seed": seed,
                    "harm": {"dice": harm, "boundary_f": harm},
                }
            )
    result = analyze(rows, ["seq1"])
    effect = result["effects"]["recent/near_background/dice"]
    assert effect["random_seed_range"] == [0, 0]
    assert effect["excess_over_max_random"] == 0.02
    assert not result["gate_pass"]  # one sequence cannot pass the minimum-n gate

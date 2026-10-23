import torch
import pytest

from scripts.bank_integrity import bank_contents, fingerprint, changed_paths
from scripts.run_contiguous_exclusions import (
    bank_digest,
    logits_exact,
    assert_restoration,
)


def state_fixture():
    value = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    return {
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: {"a": value, "alias": value}},
                "non_cond_frame_outputs": {1: {"tensor": torch.zeros(2, 2)}},
            }
        },
        "cached_features": {1: (torch.zeros(2, 2), {})},
    }


def test_regression_legacy_alias_false_alarm_and_exact_path_fix():
    state = state_fixture()
    before = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    legacy_before = bank_digest(state, 1, [{"frame": 0}], 16)
    output = state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
    output["alias"] = output["a"].clone()
    after = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    assert bank_digest(state, 1, [{"frame": 0}], 16) != legacy_before
    assert before["sha256"] == after["sha256"]
    assert not changed_paths(before, after)
    assert before["addresses"] != after["addresses"]


@pytest.mark.parametrize(
    "change",
    ["bf16_bit", "cache_bit", "current_target", "stride", "structure", "signed_zero"],
)
def test_every_real_byte_layout_and_structure_change_is_detected(change):
    state = state_fixture()
    before = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    output = state["output_dict_per_obj"][0]
    if change == "bf16_bit":
        output["cond_frame_outputs"][0]["a"].view(torch.uint8)[0, 0] ^= 1
    elif change == "cache_bit":
        state["cached_features"][1][0][0, 0] = 1e-10
    elif change == "current_target":
        output["non_cond_frame_outputs"][1]["tensor"][0, 0] = 1e-10
    elif change == "stride":
        value = output["non_cond_frame_outputs"][1]["tensor"]
        output["non_cond_frame_outputs"][1]["tensor"] = value.T
    elif change == "structure":
        output["non_cond_frame_outputs"][1]["unexpected"] = None
    else:
        state["cached_features"][1][0][0, 0] = -0.0
    after = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    assert before["sha256"] != after["sha256"]
    assert changed_paths(before, after)


def test_all_alias_paths_are_checked_and_dict_order_is_irrelevant():
    state = state_fixture()
    value = bank_contents(state, 1, [{"frame": 0}], 16)
    before = fingerprint(value)
    assert len(before["addresses"]) == 4
    assert (
        fingerprint(dict(reversed(list(value.items()))))["sha256"] == before["sha256"]
    )


def test_replay_exactness_is_byte_exact_not_just_torch_equal():
    value = torch.zeros(1)
    signed = torch.tensor([-0.0])
    assert torch.equal(value, signed)
    assert not logits_exact(value, signed)
    assert logits_exact(value, value.clone())


def test_restoration_accepts_alias_only_but_stops_real_content_change(tmp_path):
    state = state_fixture()
    before = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    output = state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
    output["alias"] = output["a"].clone()
    after = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    logits = torch.zeros(1, 1, 2, 2)
    path = tmp_path / "failure.json"
    assert_restoration(logits, logits, before["sha256"], after["sha256"], path, 62)
    assert not path.exists()
    output["alias"].view(torch.uint8)[0, 0] ^= 1
    changed = fingerprint(bank_contents(state, 1, [{"frame": 0}], 16))
    with pytest.raises(RuntimeError, match="bank_exact=False"):
        assert_restoration(
            logits, logits, before["sha256"], changed["sha256"], path, 62
        )

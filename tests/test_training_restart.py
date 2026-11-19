"""CPU checks for authenticated epoch restarts without real model weights."""
import random

import numpy as np
import pytest
import torch

from training.restart_state import (
    capture_rng, restore_rng, digest, save_epoch_snapshot, load_epoch_snapshot,
)


def test_rng_snapshot_restores_python_numpy_and_torch_exactly():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    snapshot = capture_rng()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng(snapshot)
    actual = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_epoch_snapshot_is_authentic_and_fails_closed(tmp_path):
    path = tmp_path/"epoch.pt"
    payload = {"format":"toy", "completed_epochs":1, "training_steps":2,
               "state_dict":{"weight":torch.ones(2)},
               "optimizer_state_dict":{"state":{}},
               "losses":[0.5,0.4], "rng_state":capture_rng()}
    save_epoch_snapshot(path, payload)
    record = load_epoch_snapshot(path,digest(path),expected={"format":"toy"})
    assert record["training_steps"] == 2
    with pytest.raises(FileExistsError):
        save_epoch_snapshot(path,payload)
    with pytest.raises(ValueError,match="SHA256"):
        load_epoch_snapshot(path,"a"*64,expected={"format":"toy"})
    with pytest.raises(ValueError,match="provenance"):
        load_epoch_snapshot(path,digest(path),expected={"format":"wrong"})


def test_incomplete_restart_not_accepted(tmp_path):
    path=tmp_path/"broken.pt"
    torch.save({"format":"toy","completed_epochs":1,"training_steps":1,
                "state_dict":{},"optimizer_state_dict":{},"losses":[]},path)
    with pytest.raises(ValueError,match="RNG state"):
        load_epoch_snapshot(path,digest(path),expected={"format":"toy"})

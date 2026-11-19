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

def test_epoch_resume_reproduces_contiguous_cpu_adamw_training(tmp_path):
    def epoch(model, optimizer):
        for _ in range(3):
            x = torch.randn(4, 4)
            y = torch.randn(4, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

    torch.manual_seed(34)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epoch(model, optimizer)
    snap = {
        "format": "toy", "completed_epochs": 1, "training_steps": 3,
        "state_dict": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": [0.1, 0.1, 0.1],
        "rng_state": capture_rng()
    }
    path=tmp_path/"epoch1.pt"
    save_epoch_snapshot(path, snap)
    epoch(model, optimizer)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}

    torch.manual_seed(987)
    resumed = torch.nn.Linear(4,2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    record = load_epoch_snapshot(path, digest(path), expected={"format":"toy"})
    resumed.load_state_dict(record["state_dict"])
    resumed_optimizer.load_state_dict(record["optimizer_state_dict"])
    restore_rng(record["rng_state"])
    epoch(resumed, resumed_optimizer)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

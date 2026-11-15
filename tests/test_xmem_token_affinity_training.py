import torch
from torch import nn

import modeling.xmem_token_affinity_training as training


class _Decoder(nn.Module):
    def forward(self, logits):
        return logits, logits.new_ones((logits.shape[0], 1))


class _Model:
    def __init__(self):
        self.sam_mask_decoder = _Decoder()
        self.seen_memory = []

    def track_step(self, **kwargs):
        assert kwargs["mask_inputs"] is None
        assert kwargs["point_inputs"] is None
        bank = kwargs["output_dict"]
        self.seen_memory.append(sorted(
            set(bank["cond_frame_outputs"]) | set(bank["non_cond_frame_outputs"])
        ))
        frame = kwargs["frame_idx"]
        logits = torch.full((1, 1, 4, 4), float(frame), requires_grad=True)
        self.sam_mask_decoder(logits)
        return {
            "pred_masks": logits,
            "obj_ptr": torch.tensor([[float(frame)]]),
            "object_score_logits": torch.tensor([[1.0]]),
            "maskmem_features": torch.full((1, 64, 2, 2), float(frame)),
            "maskmem_pos_enc": [torch.zeros((1, 64, 2, 2))],
        }


def _sample(frame, target):
    return {
        "frame": frame,
        "num_frames": 4,
        "vision": (),
        "positions": (),
        "sizes": (),
        "target": target,
        "mask_inputs": None,
        "point_inputs": None,
    }


def _initial_bank():
    return {
        "cond_frame_outputs": {0: {"pred_masks": torch.zeros((1, 1, 4, 4))}},
        "non_cond_frame_outputs": {},
    }


def test_autoregressive_xmem_clip_reads_its_own_prior_predictions(monkeypatch):
    monkeypatch.setattr(training, "present_mask_loss", lambda logits, target, fn: logits.mean())
    monkeypatch.setattr(training, "object_presence_loss", lambda score, target: score.mean() * 0)
    model = _Model()
    target = torch.zeros((1, 1, 4, 4))

    loss, bank, writes = training.xmem_clip_loss(
        model, [_sample(1, target), _sample(2, target)], _initial_bank()
    )

    assert torch.isfinite(loss)
    assert model.seen_memory == [[0], [0, 1]]
    assert torch.equal(bank["non_cond_frame_outputs"][1]["pred_masks"],
                       torch.ones((1, 1, 4, 4)))
    assert torch.equal(bank["non_cond_frame_outputs"][2]["pred_masks"],
                       torch.full((1, 1, 4, 4), 2.0))
    assert all(row["source"] == "this_arm_predicted_output" for row in writes)
    assert all(row["target_used_for_memory"] is False for row in writes)
    assert bank["non_cond_frame_outputs"][1]["maskmem_features"].dtype == torch.bfloat16


def test_xmem_clip_observer_receives_each_native_track_step_return_read_only(monkeypatch):
    monkeypatch.setattr(training, "present_mask_loss", lambda logits, target, fn: logits.mean())
    monkeypatch.setattr(training, "object_presence_loss", lambda score, target: score.mean() * 0)
    model = _Model()
    observed = []

    def observer(frame, output):
        observed.append((frame, output["pred_masks"].detach().clone()))

    _, bank, _ = training.xmem_clip_loss(
        model,
        [_sample(1, torch.zeros((1, 1, 4, 4))), _sample(2, torch.zeros((1, 1, 4, 4)))],
        _initial_bank(),
        step_observer=observer,
    )

    assert [frame for frame, _ in observed] == [1, 2]
    assert torch.equal(observed[0][1], bank["non_cond_frame_outputs"][1]["pred_masks"])
    assert torch.equal(observed[1][1], bank["non_cond_frame_outputs"][2]["pred_masks"])


def test_ground_truth_changes_xmem_loss_not_memory_writes(monkeypatch):
    monkeypatch.setattr(
        training,
        "present_mask_loss",
        lambda logits, target, fn: (logits - target.mean()).square().mean(),
    )
    monkeypatch.setattr(training, "object_presence_loss", lambda score, target: score.mean() * 0)
    target0 = torch.zeros((1, 1, 4, 4))
    target1 = torch.ones((1, 1, 4, 4))
    model0, model1 = _Model(), _Model()

    loss0, bank0, _ = training.xmem_clip_loss(model0, [_sample(1, target0)], _initial_bank())
    loss1, bank1, _ = training.xmem_clip_loss(model1, [_sample(1, target1)], _initial_bank())

    assert loss0 != loss1
    assert torch.equal(bank0["non_cond_frame_outputs"][1]["pred_masks"],
                       bank1["non_cond_frame_outputs"][1]["pred_masks"])
    assert torch.equal(bank0["non_cond_frame_outputs"][1]["maskmem_features"],
                       bank1["non_cond_frame_outputs"][1]["maskmem_features"])


def test_xmem_clip_rejects_nonconsecutive_frames():
    model = _Model()
    samples = [_sample(1, torch.zeros((1, 1, 4, 4))),
               _sample(3, torch.zeros((1, 1, 4, 4)))]
    try:
        training.xmem_clip_loss(model, samples, _initial_bank())
    except ValueError as error:
        assert "consecutive" in str(error)
    else:
        raise AssertionError("nonconsecutive clip was accepted")


def test_xmem_clip_rejects_ground_truth_prompts_after_the_start():
    sample = _sample(1, torch.zeros((1, 1, 4, 4)))
    sample["mask_inputs"] = torch.ones((1, 1, 4, 4))
    try:
        training.xmem_clip_loss(_Model(), [sample], _initial_bank())
    except ValueError as error:
        assert "ground-truth prompts" in str(error)
    else:
        raise AssertionError("post-prompt ground-truth input was accepted")

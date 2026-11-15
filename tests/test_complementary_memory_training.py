import torch
from torch import nn

from modeling.complementary_memory import training


class _Decoder(nn.Module):
    def forward(self, logits):
        return logits, logits.new_ones((logits.shape[0], 1))


class _Model:
    def __init__(self):
        self.sam_mask_decoder = _Decoder()
        self.seen = []

    def track_step(self, **kwargs):
        assert kwargs["mask_inputs"] is None
        assert kwargs["point_inputs"] is None
        bank = kwargs["output_dict"]
        self.seen.append(sorted(
            set(bank["cond_frame_outputs"]) | set(bank["non_cond_frame_outputs"])
        ))
        frame = kwargs["frame_idx"]
        logits = torch.full((1, 1, 4, 4), float(frame))
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
        "num_frames": 3,
        "vision": (),
        "positions": (),
        "sizes": (),
        "target": target,
    }


def _initial_bank():
    return {
        "cond_frame_outputs": {0: {"pred_masks": torch.zeros((1, 1, 4, 4))}},
        "non_cond_frame_outputs": {},
    }


def test_clip_writes_its_predictions_before_the_next_read(monkeypatch):
    monkeypatch.setattr(training, "present_mask_loss", lambda logits, target, fn: logits.mean())
    monkeypatch.setattr(training, "object_presence_loss", lambda score, target: score.mean() * 0)
    model = _Model()
    target = torch.zeros((1, 1, 4, 4))

    _, bank, writes = training.reader_clip_loss(
        model, [_sample(1, target), _sample(2, target)], _initial_bank()
    )

    assert model.seen == [[0], [0, 1]]
    assert torch.equal(bank["non_cond_frame_outputs"][1]["pred_masks"], torch.ones((1, 1, 4, 4)))
    assert torch.equal(bank["non_cond_frame_outputs"][2]["pred_masks"], torch.full((1, 1, 4, 4), 2.0))
    assert all(not row["target_used_for_memory"] for row in writes)
    assert bank["non_cond_frame_outputs"][1]["maskmem_features"].dtype == torch.bfloat16


def test_gt_targets_change_loss_only_not_memory_writes(monkeypatch):
    monkeypatch.setattr(training, "present_mask_loss", lambda logits, target, fn: (logits - target.mean()).square().mean())
    monkeypatch.setattr(training, "object_presence_loss", lambda score, target: score.mean() * 0)
    first = torch.zeros((1, 1, 4, 4))
    second = torch.ones((1, 1, 4, 4))

    model_a = _Model()
    loss_a, bank_a, _ = training.reader_clip_loss(model_a, [_sample(1, first)], _initial_bank())
    model_b = _Model()
    loss_b, bank_b, _ = training.reader_clip_loss(model_b, [_sample(1, second)], _initial_bank())

    assert loss_a != loss_b
    assert torch.equal(
        bank_a["non_cond_frame_outputs"][1]["pred_masks"],
        bank_b["non_cond_frame_outputs"][1]["pred_masks"],
    )
    assert torch.equal(
        bank_a["non_cond_frame_outputs"][1]["maskmem_features"],
        bank_b["non_cond_frame_outputs"][1]["maskmem_features"],
    )


def test_detach_tensors_breaks_graph_without_changing_values():
    value = torch.ones((2,), requires_grad=True) * 3
    bank = {"features": [value], "positions": (value + 1,)}
    detached = training.detach_tensors(bank)

    assert torch.equal(detached["features"][0], value.detach())
    assert torch.equal(detached["positions"][0], (value + 1).detach())
    assert not detached["features"][0].requires_grad
    assert not detached["positions"][0].requires_grad

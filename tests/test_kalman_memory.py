"""Guards for the Kalman memory update built from RecursiveKalmanNet's official code (learned noise,
detector-corrected presence, reliability, data-association update) and its loss."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from modeling.kalman_memory import RELIABILITY_BIAS, RKNMemoryUpdate, box_iou, box_map, mask_box  # noqa: E402
from Algo.KalmanFilter import KalmanFilter  # noqa: E402  (official, put on the path by kalman_memory)
from Algo.RecursiveKalmanNet import GRUNetwork  # noqa: E402


def cues(association=1.0, mask=0.9, change=0.05, size=(8, 8)):
    return {"mask_probability": torch.full((1, 1, *size), mask), "feature_change": torch.full((1, 1, *size), change),
            "prompt_memory": torch.zeros(1, 4, *size), "detection_box": torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            "detection_confidence": torch.tensor([[0.9]]), "mask_box_iou": torch.tensor([[0.8]]),
            "association": torch.tensor([[association]])}


def test_data_association_update_on_the_official_kalman_step():
    torch.manual_seed(0)
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8)
    assert isinstance(update.filter, KalmanFilter)
    assert all(isinstance(network, GRUNetwork) for network in update.networks())
    mean, candidate = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    for beta in (1.0, 0.3):
        variance = update.start(mean)
        out = update.update(mean, variance, candidate, cues(beta))
        q, r = out["process_noise"], out["measurement_noise"]  # per pixel, shared over channels
        prior = variance.reshape(1, 4, 8, 8) + q
        gain = prior / (prior + r)
        correction = gain * (candidate - mean)
        posterior = (1 - gain).pow(2) * prior + gain.pow(2) * r  # official Joseph form
        assert torch.allclose(out["mean"], mean + beta * correction, atol=1e-5)
        expected = beta * posterior + (1 - beta) * prior + beta * (1 - beta) * correction.pow(2)
        assert torch.allclose(out["covariance"].reshape(1, 4, 8, 8), expected, atol=1e-5)
        assert torch.allclose(out["gain"], (beta * gain)[:, :1], atol=1e-5)


def test_an_absent_frame_only_predicts_whatever_the_noise():
    torch.manual_seed(0)
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8, weight_factor=1.0)
    with torch.no_grad():
        update.filter.rnn_Q.output_layer.bias.fill_(10.0)  # a huge learned Q would make K ~ 1
    mean, candidate = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    out = update.update(mean, update.start(mean), candidate, cues(association=0.0))
    assert torch.allclose(out["mean"], mean) and float(out["gain"].max()) == 0.0


def test_presence_only_ablation_writes_every_frame():
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8, learn_memory=False)
    assert update.trainable_networks() == (update.rnn_presence,)
    mean, candidate = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    out = update.update(mean, update.start(mean), candidate, cues(association=0.0))
    assert float(out["gain"].mean()) > 0.3  # beta = 1 regardless


def test_fixed_gain_control_keeps_the_same_association_gate():
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8)
    update.fixed_gain = 0.25
    mean, candidate = torch.zeros(1, 4, 8, 8), torch.ones(1, 4, 8, 8)
    covariance = update.start(mean)
    out = update.update(mean, covariance, candidate, cues(association=0.4))
    assert torch.allclose(out["mean"], torch.full_like(mean, 0.1))
    assert torch.allclose(out["gain"], torch.full((1, 1, 1, 1), 0.1))
    assert out["covariance"] is covariance


def test_untrained_heads_keep_medsam2_and_see_their_inputs():
    torch.manual_seed(0)
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8)
    update.start(torch.zeros(1, 4, 8, 8))
    rho = torch.sigmoid(update.reliability(torch.tensor([[0.9]]), torch.tensor([[0.8]]), torch.tensor([[1.0]])))
    assert abs(float(rho) - float(torch.sigmoid(torch.tensor(RELIABILITY_BIAS)))) < 0.02
    fused = []
    for confidence, iou in ((0.9, 0.8), (0.0, 0.0)):
        update.start(torch.zeros(1, 4, 8, 8))
        fused.append(update.presence(torch.tensor([[3.0]]), {"confidence": torch.tensor([[confidence]])},
                                     torch.tensor([[iou]])))
    assert abs(float(fused[0]) - 3.0) < 1.0 and float(fused[0]) != float(fused[1])


def test_boxes():
    mask = torch.zeros(1, 10, 10)
    mask[0, 2:6, 4:8] = 1
    assert torch.allclose(mask_box(mask), torch.tensor([[0.4, 0.2, 0.8, 0.6]]))
    assert float(box_iou(mask_box(mask), torch.tensor([[0.4, 0.2, 0.8, 0.6]]))) == 1.0
    assert float(box_iou(torch.zeros(1, 4), torch.tensor([[0.4, 0.2, 0.8, 0.6]]))) == 0.0
    assert torch.isclose(box_map(torch.tensor([[0.0, 0.0, 0.5, 0.5]]), torch.tensor([[0.7]]), (4, 4)).sum(),
                         torch.tensor(0.7 * 4))


def test_loss_targets_and_gradients():
    from training.memory_trainer import RKNLoss

    torch.manual_seed(0)
    update = RKNMemoryUpdate(memory_channels=4, image_channels=8, weight_factor=1.0)
    state = torch.randn(1, 4, 8, 8)
    covariance = update.start(state)
    mask = torch.zeros(32, 32)
    mask[8:24, 8:24] = 1
    logits = (mask * 20 - 10)[None, None]
    frames = []
    for target in (mask, torch.zeros_like(mask)):  # a polyp frame, then an empty frame with a false positive
        score = update.presence(torch.tensor([[2.0]]), {"confidence": torch.tensor([[0.9]])}, torch.tensor([[0.8]]))
        rho_logit = update.reliability(torch.tensor([[0.9]]), torch.tensor([[0.8]]), torch.tensor([[1.0]]))
        out = update.update(state, covariance, torch.randn(1, 4, 8, 8), cues(float(torch.sigmoid(score) * torch.sigmoid(rho_logit))))
        out["target"] = torch.randn(1, 4, 8, 8)
        frames.append({"memory": out, "multistep_object_score_logits": [score], "reliability_logit": rho_logit,
                       "pred_masks_high_res": logits.clone().requires_grad_(), "tracker_masks_high_res": logits,
                       "ungated_mask": logits[..., ::4, ::4]})
        state, covariance = out["mean"], out["covariance"]
    loss, parts = RKNLoss()([None, *frames], None, torch.stack([mask, mask, torch.zeros_like(mask)]))
    assert set(parts) >= {"mask", "presence", "reliability", "nll", "tracker_iou"}
    assert abs(parts["tracker_iou"] - 0.5) < 1e-6  # IoU 1 on the polyp frame, 0 on the false positive
    loss.backward()
    for network in update.networks():
        assert network.output_layer.weight.grad.abs().sum() > 0

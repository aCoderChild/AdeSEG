"""Memory update: a Kalman filter with learned noise, built from RecursiveKalmanNet's official code
(external/RecursiveKalmanNet, Mortada et al., EUSIPCO 2025), with a polyp detector as an
observation independent of MedSAM2.

The recurrent state is the object's appearance memory. The filter is the official
``KalmanFilter`` (``step``: predict, innovation, gain K = P- H^T S^-1, Joseph-form covariance) with
F = H = 1 on every element of MedSAM2's 64x32x32 memory; the measurement z is a frame memory.
Each update is weighted by the probability beta that the measurement really is the prompted
object (probabilistic data association, Bar-Shalom):

    x = x- + beta K (z - x-),   P = beta P+ + (1 - beta) P- + beta (1 - beta) (K (z - x-))^2,

so an absent or unreliable frame only predicts (x kept, P grows by Q), whatever Q and R are.
Four official ``GRUNetwork``s (RKN's default configuration) learn:

- presence p (per frame, inside MedSAM2's decoder): a correction of MedSAM2's object score from
  the score, the detector's top-box confidence and its box IoU with MedSAM2's mask; the fused
  score gates MedSAM2's output mask, object pointer and memory, as MedSAM2's own score does;
- reliability rho (per frame, before the memory is written): the expected IoU of the tracker's
  mask, from the detector confidence, box IoU and the object pointer's similarity to the
  prompt's. ``modeling.recurrent_memory`` uses it for beta = p rho and to gate the object pointer.
  When a confident detection disagrees with the mask (or the gate is closed), the frame is
  re-detected: decoded from the detector box on memory-free features, and that mask is output
  and written, with beta = its presence x detector confidence;
- process noise Q (per pixel): from the image-feature change since the previous frame;
- measurement noise R (per pixel): where in the frame to trust, from MedSAM2's mask probability,
  the detector box and the new memory's distance to the prompt memory.

No input compares the new memory with the state itself: a corrupted state would vouch for the
wrong frames that resemble it. ``training.memory_trainer.RKNLoss`` trains the networks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "external" / "RecursiveKalmanNet"
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.append(str(OFFICIAL_ROOT))  # the official code imports its package as `Algo`

from Algo.KalmanFilter import KalmanFilter  # noqa: E402
from Algo.RecursiveKalmanNet import GRUNetwork  # noqa: E402

# RKN.load_or_init_rnns' default GRU configuration.
GRU_CONFIGURATION = {"nb_layer_FC1": 1, "FC1_mult": 10, "nbr_GRU": 1, "hidden_size_mult": 10,
                     "nb_layer_FC2": 1, "FC2_mult": 20}
PRESENCE_INPUTS = 3  # object score, detector confidence, detector-mask box IoU
RELIABILITY_INPUTS = 3  # detector confidence, box IoU, pointer similarity
PROCESS_INPUTS = 1  # image-feature change
MEASUREMENT_INPUTS = 3  # mask probability, detector box, distance to the prompt memory
PRESENCE_SCALE = 10.0  # the presence network's output, in object-score logits
RELIABILITY_BIAS = 3.0  # rho starts at sigmoid(3) = 0.95: untrained, pointers are kept and nothing is re-detected


def _network(inputs: int, weight_factor: float) -> GRUNetwork:
    """An official GRUNetwork (1-D state and observation, one output) with ``inputs`` features."""
    network = GRUNetwork(1, 1, 1, GRU_CONFIGURATION, weight_factor=weight_factor)
    if inputs != network.input_size:  # resize the input layer, initialised as GRUNetwork does
        first = network.fc1[0]
        wider = nn.Linear(inputs, first.out_features)
        with torch.no_grad():
            wider.weight.mul_(weight_factor)
            wider.bias.mul_(weight_factor)
        network.fc1[0] = wider
    return network


class _LearnedNoiseKalmanFilter(KalmanFilter, nn.Module):
    """The official KalmanFilter with Q and R set each step from the noise networks (as ``RKN``
    combines KalmanFilter with nn.Module). ``predict_cov`` and ``calc_innov_cov`` build Q_t and
    R_t in one tensor operation instead of the official per-filter loop; the values are the same."""

    def __init__(self, initial_variance: float, weight_factor: float):
        nn.Module.__init__(self)
        self.register_buffer("one", torch.ones(1, 1))  # F = H = 1, on the module's device
        KalmanFilter.__init__(self, [lambda t: self.one, None, lambda t: self.one, None], torch.zeros(1, 1),
                              torch.full((1, 1), initial_variance))
        self.rnn_Q = _network(PROCESS_INPUTS, weight_factor)
        self.rnn_R = _network(MEASUREMENT_INPUTS, weight_factor)
        self.Q_values = self.R_values = None

    def predict_cov(self, t):
        self.Q_t = self.Q_values
        self.P_prior = self.F_t @ self.P @ self.F_t.transpose(1, 2) + self.Q_t

    def calc_innov_cov(self, t):
        self.R_t = self.R_values
        self.S = self.H_t @ self.P_prior @ self.H_t.transpose(1, 2) + self.R_t

    def calc_gain(self):
        """The official K = P- H^T S^-1 with S^-1 = 1 / S (S is 1x1; torch.linalg.inv of 65k 1x1
        matrices takes about 11 s on MPS)."""
        self.K = self.P_prior @ self.H_t.transpose(1, 2) * self.S.reciprocal()


def _elements(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B*C*H*W, 1, 1]: one scalar filter per memory element."""
    return x.reshape(-1, 1, 1)


def _pixels(x: torch.Tensor) -> torch.Tensor:
    """[B, F, H, W] -> [1, B*H*W, F]: one GRU sequence step per memory pixel."""
    return x.permute(0, 2, 3, 1).reshape(1, -1, x.size(1))


def _frame(x: torch.Tensor) -> torch.Tensor:
    """[B, F] -> [1, B, F]: one GRU sequence step per video."""
    return x.unsqueeze(0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of normalised [B, 4] x1, y1, x2, y2 boxes -> [B, 1]; 0 when either is empty."""
    width = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp_min(0)
    height = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp_min(0)
    area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    union = area(a) + area(b) - width * height
    return torch.where(union > 0, width * height / union.clamp_min(1e-12), torch.zeros_like(union)).unsqueeze(1)


def mask_box(masks: torch.Tensor) -> torch.Tensor:
    """Normalised box of each [B, H, W] mask's foreground (> 0) -> [B, 4]; zeros when empty."""
    boxes = []
    for mask in masks:
        points = torch.nonzero(mask > 0)
        if len(points) == 0:
            boxes.append(mask.new_zeros(4))
            continue
        (y1, x1), (y2, x2) = points.min(0).values, points.max(0).values
        height, width = mask.shape
        boxes.append(torch.stack([x1 / width, y1 / height, (x2 + 1) / width, (y2 + 1) / height]).float())
    return torch.stack(boxes)


def box_map(boxes: torch.Tensor, values: torch.Tensor, size) -> torch.Tensor:
    """[B, 1, H, W] map holding ``values`` [B, 1] inside each normalised box, 0 outside."""
    height, width = size
    ys = (torch.arange(height, device=boxes.device) + 0.5) / height
    xs = (torch.arange(width, device=boxes.device) + 0.5) / width
    inside_y = (ys[None] >= boxes[:, 1:2]) & (ys[None] <= boxes[:, 3:4])
    inside_x = (xs[None] >= boxes[:, 0:1]) & (xs[None] <= boxes[:, 2:3])
    return (inside_y[:, :, None] & inside_x[:, None, :]).float().unsqueeze(1) * values[..., None, None]


class RKNMemoryUpdate(nn.Module):
    """Official Kalman filter on the appearance memory, with learned noise, presence and write reliability."""

    CHECKPOINT_KIND = "rkn_detector"
    uses_detector = True

    def __init__(self, memory_channels: int, image_channels: int, initial_variance: float = 1.0,
                 weight_factor: float = 0.1, learn_memory: bool = True):
        super().__init__()
        self.memory_channels = memory_channels
        self.image_channels = image_channels
        # learn_memory=False (presence-only ablation): Q, R and rho keep their initial networks, beta = 1,
        # pointers are not gated and nothing is re-detected; only the presence correction is learned.
        self.config = {"initial_variance": initial_variance, "weight_factor": weight_factor, "learn_memory": learn_memory}
        self.filter = _LearnedNoiseKalmanFilter(initial_variance, weight_factor)
        self.rnn_presence = _network(PRESENCE_INPUTS, weight_factor)
        self.rnn_reliability = _network(RELIABILITY_INPUTS, weight_factor)
        self.fixed_gain = None  # Evaluation-only EMA control; set after loading the same RKN checkpoint.

    def networks(self):
        return self.rnn_presence, self.rnn_reliability, self.filter.rnn_Q, self.filter.rnn_R

    def trainable_networks(self):
        return self.networks() if self.config["learn_memory"] else (self.rnn_presence,)

    def start(self, memory: torch.Tensor) -> torch.Tensor:
        """Reset the GRU states for a new video; returns P0 for every element."""
        batch, _, height, width = memory.shape
        for network in (self.rnn_presence, self.rnn_reliability):
            network.reset_hidden_state(batch)
        for network in (self.filter.rnn_Q, self.filter.rnn_R):
            network.reset_hidden_state(batch * height * width)
        return self.filter.P0.to(memory.device).expand(memory.numel(), -1, -1)

    def presence(self, object_score: torch.Tensor, detection: dict, mask_box_iou: torch.Tensor) -> torch.Tensor:
        """MedSAM2's object score [B, 1] corrected by the detector evidence."""
        inputs = torch.cat([object_score / 10.0, detection["confidence"], mask_box_iou], dim=1).detach()
        return object_score + PRESENCE_SCALE * self.rnn_presence(_frame(inputs.float())).squeeze(0)

    def reliability(self, detection_confidence, mask_box_iou, pointer_similarity) -> torch.Tensor:
        """Logit of rho [B, 1], the expected IoU of the tracker's mask for this frame."""
        inputs = torch.cat([detection_confidence, mask_box_iou, pointer_similarity], dim=1).detach()
        return self.rnn_reliability(_frame(inputs.float())).squeeze(0) + RELIABILITY_BIAS

    def noise(self, network, inputs, channels):
        """Per-pixel noise variance from a GRU network, repeated over the memory channels."""
        batch, _, height, width = inputs.shape
        log_variance = network(_pixels(inputs.detach())).reshape(batch, height, width, 1).permute(0, 3, 1, 2)
        return torch.exp(log_variance).expand(-1, channels, -1, -1)

    def update(self, state, covariance, memory, reliability):
        """Kalman update weighted by ``reliability["association"]`` (beta [B, 1])."""
        if self.fixed_gain is not None:
            beta = reliability["association"] if self.config["learn_memory"] else torch.ones_like(reliability["association"])
            gain = beta[..., None, None] * self.fixed_gain
            return {
                "mean": state + gain * (memory - state),
                "covariance": covariance,
                "gain": gain.detach(),
                "association": beta.detach(),
            }
        kf, (batch, channels, height, width) = self.filter, state.shape
        to_prompt = torch.log((memory - reliability["prompt_memory"]).pow(2).mean(1, keepdim=True) + 1e-6)
        detection_map = box_map(reliability["detection_box"], reliability["detection_confidence"], (height, width))
        change = reliability.get("feature_change")
        change = state.new_zeros(batch, 1, height, width) if change is None else change
        kf.Q_values = _elements(self.noise(kf.rnn_Q, change, channels))
        kf.R_values = _elements(self.noise(kf.rnn_R, torch.cat([
            reliability["mask_probability"], detection_map, to_prompt,
        ], dim=1), channels))
        kf.x, kf.P = _elements(state), covariance
        kf.step(_elements(memory), 0, None)
        beta = reliability["association"] if self.config["learn_memory"] else torch.ones_like(reliability["association"])
        beta = _elements(beta[..., None, None].expand(-1, channels, height, width))
        correction = kf.K @ kf.y  # K (z - x-), the full Kalman correction
        mean = kf.x_prior + beta * correction
        covariance = beta * kf.P + (1 - beta) * kf.P_prior + beta * (1 - beta) * correction.pow(2)
        gain = (beta * kf.K).reshape(state.shape)[:, :1]
        return {
            "mean": mean.reshape(state.shape),
            "covariance": covariance,
            "gain": gain.detach(),
            "association": beta.reshape(state.shape)[:, :1, :1, :1].reshape(batch, 1).detach(),
            "process_noise": kf.Q_t.reshape(state.shape)[:, :1].detach(),
            "measurement_noise": kf.R_t.reshape(state.shape)[:, :1].detach(),
        }

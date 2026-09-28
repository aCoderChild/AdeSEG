"""Reliability-gated recurrent spatial memory for MedSAM2 experiments.

The predictor keeps one spatial mask-memory state.  It deliberately leaves
MedSAM2 object-pointer selection unchanged: the prompt-frame pointer and the
native recent-pointer history are still supplied to memory attention.

The temporal compression primitive follows the two-frame ``Conv3d`` pattern
used by RDE-VOS' public ``MemCrompress`` implementation.  It is an adaptation
to MedSAM2 mask-memory features, not an RDE-VOS implementation.
"""

from __future__ import annotations

import torch
from torch import nn

from modeling.native_pointers import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor


def _logit(probability: float) -> float:
    value = torch.tensor(probability, dtype=torch.float32)
    return float(torch.logit(value).item())


class ReliabilityGatedFusion(nn.Module):
    """Fuse two spatial states and learn one update weight per video object.

    The fusion learns a two-frame spatial compression and a scalar reliability
    gate. It consumes the decoder's predicted IoU; it never receives
    ground-truth IoU.
    """

    def __init__(
        self,
        feature_channels: int,
        gate_hidden_channels: int | None = None,
        initial_gate: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1).")
        self.feature_channels = feature_channels
        self.initial_gate = initial_gate
        self.last_gate: torch.Tensor | None = None
        # RDE-VOS MemCrompress uses a two-frame Conv3d with this kernel.
        self.temporal_compression = nn.Conv3d(
            feature_channels,
            feature_channels,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1),
            bias=True,
        )
        self._initialize_candidate_identity()
        hidden = gate_hidden_channels or max(32, feature_channels // 4)
        self.reliability_gate = nn.Sequential(
            nn.Linear(1 + feature_channels * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # The initial model is exactly EMA(initial_gate), because compression
        # initially returns the unmodified candidate.
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, _logit(initial_gate))

    def _initialize_candidate_identity(self) -> None:
        """Make the temporal compressor return candidate state at step zero."""
        assert self.temporal_compression is not None
        with torch.no_grad():
            self.temporal_compression.weight.zero_()
            self.temporal_compression.bias.zero_()
            center = self.temporal_compression.kernel_size[1] // 2
            for channel in range(self.feature_channels):
                # Temporal index 1 is the second input: the current candidate.
                self.temporal_compression.weight[channel, channel, 1, center, center] = 1.0

    @staticmethod
    def _normalize_predicted_iou(predicted_iou: torch.Tensor, batch_size: int) -> torch.Tensor:
        if predicted_iou.ndim == 1:
            predicted_iou = predicted_iou.unsqueeze(1)
        if predicted_iou.ndim != 2 or predicted_iou.shape != (batch_size, 1):
            raise ValueError("predicted_iou must be [B] or [B, 1].")
        return predicted_iou

    def _candidate_state(self, previous: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        temporal_pair = torch.stack((previous, candidate), dim=2)
        return self.temporal_compression(temporal_pair).squeeze(2)

    def _gate(self, previous: torch.Tensor, candidate: torch.Tensor, predicted_iou: torch.Tensor) -> torch.Tensor:
        pooled_previous = previous.mean(dim=(-2, -1))
        pooled_candidate = candidate.mean(dim=(-2, -1))
        pooled_difference = (previous - candidate).abs().mean(dim=(-2, -1))
        quality = self._normalize_predicted_iou(predicted_iou, previous.size(0)).to(previous)
        return torch.sigmoid(
            self.reliability_gate(
                torch.cat((quality, pooled_previous, pooled_candidate, pooled_difference), dim=1)
            )
        )

    def forward(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        predicted_iou: torch.Tensor,
    ) -> torch.Tensor:
        if previous.ndim != 4 or previous.shape != candidate.shape:
            raise ValueError("previous and candidate must be matching [B, C, H, W] tensors.")
        candidate_state = self._candidate_state(previous, candidate)
        gate = self._gate(previous, candidate, predicted_iou)
        self.last_gate = gate.detach()
        return (1.0 - gate[:, :, None, None]) * previous + gate[:, :, None, None] * candidate_state


class _RGMSpatialState:
    """Inference-only holder for one recurrent spatial mask-memory tensor."""

    def __init__(self, frame_idx: int, output: dict):
        features = output.get("maskmem_features")
        positions = output.get("maskmem_pos_enc")
        if features is None or positions is None:
            raise RuntimeError("RGM requires prompt-frame mask-memory features.")
        self.features = features.detach().to(device=output["obj_ptr"].device, dtype=torch.float32)
        self.position = positions[-1].detach().to(device=self.features.device, dtype=torch.float32)
        self.last_frame_idx = frame_idx
        self.updates = 0

    def update(
        self,
        frame_idx: int,
        candidate: torch.Tensor,
        predicted_iou: torch.Tensor,
        fusion: ReliabilityGatedFusion,
    ) -> None:
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("RGM state requires consecutive forward updates.")
        candidate = candidate.to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Mask-memory shape changed within the sequence.")
        self.features = fusion(self.features, candidate, predicted_iou.to(self.features))
        self.last_frame_idx = frame_idx
        self.updates += 1


class ReliabilityGatedMemoryVideoPredictor(SAM2VideoPredictor):
    """MedSAM2 predictor with one spatial state and native pointer history."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Attached after the base MedSAM2 checkpoint loads.
        self.state_fusion: ReliabilityGatedFusion | None = None

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("rgm_enabled", False):
            raise ValueError("RGM propagation must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("RGM currently supports exactly one object.")
        if self.state_fusion is None:
            raise RuntimeError("RGM requires a loaded fusion checkpoint.")
        super().propagate_in_video_preflight(inference_state)
        outputs = inference_state["output_dict"]
        anchor_idx = inference_state["rgm_anchor_frame_idx"]
        if set(outputs["cond_frame_outputs"]) != {anchor_idx}:
            raise ValueError("RGM supports one initial prompt frame.")
        anchor = outputs["cond_frame_outputs"].get(anchor_idx)
        if anchor is None:
            raise RuntimeError("VOS preflight did not encode the prompted frame.")
        outputs["rgm_state"] = _RGMSpatialState(anchor_idx, anchor)
        if inference_state.get("rgm_capture_tensors", False):
            outputs["rgm_frame_trace"] = [{
                "frame_idx": anchor_idx,
                "mask_logits": anchor["pred_masks"].detach().float().cpu(),
                "predicted_iou": anchor["iou_predictions"].detach().float().cpu(),
                "object_pointer": anchor["obj_ptr"].detach().float().cpu(),
                "candidate": outputs["rgm_state"].features.detach().cpu(),
                "state": outputs["rgm_state"].features.detach().cpu(),
                "gate": None,
            }]
        # Keep the pointer but release the duplicated prompt-frame spatial tensor.
        anchor["maskmem_features"] = None
        anchor["maskmem_pos_enc"] = None

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("rgm_state", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        output_dict = kwargs["output_dict"]
        frame_idx = kwargs["frame_idx"]
        state = output_dict.get("rgm_state")
        if state is None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)
        if kwargs["reverse"]:
            raise ValueError("RGM supports forward propagation only.")

        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        candidate = current_out.get("maskmem_features")
        predicted_iou = current_out.get("iou_predictions")
        if candidate is None or predicted_iou is None:
            raise RuntimeError("RGM requires candidate memory and predicted IoU.")
        assert self.state_fusion is not None
        previous_state = state.features.detach().clone()
        state.update(frame_idx, candidate, predicted_iou, self.state_fusion)
        if kwargs["inference_state"].get("rgm_capture_tensors", False):
            output_dict.setdefault("rgm_frame_trace", []).append({
                "frame_idx": frame_idx,
                "mask_logits": current_out["pred_masks"].detach().float().cpu(),
                "predicted_iou": predicted_iou.detach().float().cpu(),
                "object_pointer": current_out["obj_ptr"].detach().float().cpu(),
                "candidate": candidate.detach().float().cpu(),
                "previous_state": previous_state.detach().float().cpu(),
                "state": state.features.detach().float().cpu(),
                "gate": self.state_fusion.last_gate.detach().float().cpu(),
            })
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["rgm_trace"] = {
            "updates": state.updates,
            "gate": [float(value) for value in self.state_fusion.last_gate.flatten().cpu()],
        }
        return current_out, pred_masks

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        if is_init_cond_frame:
            return super()._prepare_memory_conditioned_features(
                frame_idx, True, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse,
            )
        if track_in_reverse:
            raise ValueError("RGM supports forward propagation only.")
        state = output_dict["rgm_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("RGM state was read out of order.")
        dtype = current_vision_feats[-1].dtype
        memory_chunks = [state.features.to(dtype).flatten(2).permute(2, 0, 1)]
        position_chunks = [
            (state.position + self.maskmem_tpos_enc[0].view(1, -1, 1, 1))
            .to(dtype).flatten(2).permute(2, 0, 1)
        ]
        num_obj_ptr_tokens = append_native_object_pointers(
            self, frame_idx, output_dict, num_frames, track_in_reverse,
            current_vision_feats[-1].device, current_vision_feats[-1].size(1),
            memory_chunks, position_chunks,
        )
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory_chunks, dim=0),
            memory_pos=torch.cat(position_chunks, dim=0),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        batch_size = current_vision_feats[-1].size(1)
        return fused.permute(1, 2, 0).reshape(batch_size, self.hidden_dim, *feat_sizes[-1])

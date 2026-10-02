"""Recurrent MedSAM2 memory with LiVOS gating and RDE-VOS fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from modeling.native_pointers import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor


class LiVOSGate(nn.Module):
    def __init__(
        self,
        image_channels: int,
        state_channels: int,
        fixed_gate: float | None = None,
    ):
        super().__init__()
        if fixed_gate is not None and not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("fixed_gate must be in [0, 1].")
        self.state_channels = state_channels
        self.fixed_gate = fixed_gate
        self.layer = nn.Conv2d(image_channels, state_channels, kernel_size=1)

    def forward(self, image_feature: torch.Tensor) -> torch.Tensor:
        if image_feature.ndim != 4:
            raise ValueError("image_feature must be [B, C, H, W].")
        if self.fixed_gate is not None:
            return image_feature.new_full(
                (image_feature.size(0), self.state_channels), self.fixed_gate
            )
        return torch.sigmoid(self.layer(image_feature)).mean(dim=(-2, -1))


class _NonLocal3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, channels // 2)
        pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.g = nn.Sequential(nn.Conv3d(channels, hidden, 1), pool)
        self.theta = nn.Conv3d(channels, hidden, 1)
        self.phi = nn.Sequential(nn.Conv3d(channels, hidden, 1), nn.MaxPool3d((1, 2, 2)))
        self.out = nn.Conv3d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        g = self.g(x).flatten(2).transpose(1, 2)
        theta = self.theta(x).flatten(2).transpose(1, 2)
        phi = self.phi(x).flatten(2)
        affinity = torch.matmul(theta, phi) / phi.size(-1)
        y = torch.matmul(affinity, g).transpose(1, 2).contiguous()
        y = y.view(batch, -1, *x.shape[2:])
        return self.out(y) + x


class _ASPP3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 4)
        self.branches = nn.ModuleList([
            nn.Conv3d(channels, mid, 1, bias=False),
            nn.Conv3d(channels, mid, (1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2), bias=False),
            nn.Conv3d(channels, mid, (1, 3, 3), padding=(0, 4, 4), dilation=(1, 4, 4), bias=False),
            nn.Conv3d(channels, mid, (1, 3, 3), padding=(0, 6, 6), dilation=(1, 6, 6), bias=False),
        ])
        self.project = nn.Conv3d(mid * 4, channels, (1, 3, 3), padding=(0, 1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.relu(branch(x), inplace=True) for branch in self.branches], dim=1)
        return F.relu(self.project(x), inplace=True)


class RDEFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.extract = _NonLocal3D(channels)
        self.enhance = _ASPP3D(channels)
        self.squeeze = nn.Conv3d(
            channels,
            channels,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1),
        )

    def forward(self, previous: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        x = torch.stack((previous, candidate), dim=2)
        x = self.extract(x)
        x = x + self.enhance(x)
        return self.squeeze(x).squeeze(2)


class ReliabilityGatedFusion(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        fixed_gate: float | None = None,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.image_channels = image_channels
        self.fixed_gate = fixed_gate
        self.last_gate: torch.Tensor | None = None
        self.register_buffer("livos_rde_version", torch.tensor(1, dtype=torch.int8))
        self.gate_projector = LiVOSGate(image_channels, feature_channels, fixed_gate)
        self.rde_fusion = RDEFusion(feature_channels)

    def forward(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        image_feature: torch.Tensor,
    ) -> torch.Tensor:
        if previous.ndim != 4 or previous.shape != candidate.shape:
            raise ValueError("previous and candidate must be matching [B, C, H, W] tensors.")
        if image_feature.size(0) != previous.size(0):
            raise ValueError("image_feature batch size must match the recurrent state.")
        gate = self.gate_projector(image_feature)
        self.last_gate = gate.detach()
        retained = previous * gate[:, :, None, None]
        return self.rde_fusion(retained, candidate)


class _RGMSpatialState:
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
        image_feature: torch.Tensor,
        fusion: ReliabilityGatedFusion,
    ) -> None:
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("RGM state requires consecutive forward updates.")
        candidate = candidate.to(self.features)
        image_feature = image_feature.to(device=self.features.device, dtype=torch.float32)
        if candidate.shape != self.features.shape:
            raise ValueError("Mask-memory shape changed within the sequence.")
        self.features = fusion(self.features, candidate, image_feature)
        self.last_frame_idx = frame_idx
        self.updates += 1


class ReliabilityGatedMemoryVideoPredictor(SAM2VideoPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        _, backbone_out, _, _, _ = self._get_image_feature(
            kwargs["inference_state"], frame_idx, kwargs["batch_size"]
        )
        image_feature = backbone_out["backbone_fpn"][-1]
        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        candidate = current_out.get("maskmem_features")
        predicted_iou = current_out.get("iou_predictions")
        if candidate is None or predicted_iou is None:
            raise RuntimeError("RGM requires candidate memory and predicted IoU.")
        assert self.state_fusion is not None
        previous_state = state.features.detach().clone()
        state.update(frame_idx, candidate, image_feature, self.state_fusion)
        gate = self.state_fusion.last_gate
        if gate is None:
            raise RuntimeError("RGM gate was not computed.")
        if kwargs["inference_state"].get("rgm_capture_tensors", False):
            output_dict.setdefault("rgm_frame_trace", []).append({
                "frame_idx": frame_idx,
                "mask_logits": current_out["pred_masks"].detach().float().cpu(),
                "predicted_iou": predicted_iou.detach().float().cpu(),
                "object_pointer": current_out["obj_ptr"].detach().float().cpu(),
                "candidate": candidate.detach().float().cpu(),
                "previous_state": previous_state.detach().float().cpu(),
                "state": state.features.detach().float().cpu(),
                "gate": gate.detach().float().cpu(),
            })
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        gate_cpu = gate.detach().float().cpu()
        current_out["rgm_trace"] = {
            "updates": state.updates,
            "gate_mean": float(gate_cpu.mean()),
            "gate_min": float(gate_cpu.min()),
            "gate_max": float(gate_cpu.max()),
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

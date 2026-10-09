"""Recurrent memory for frozen MedSAM2: the prompt memory plus one recurrent state.

MedSAM2 natively keeps a bank of the last 7 frame memories. Here that bank is replaced by
the prompt frame's memory (the anchor) and one state that a memory update rewrites each
frame from the newest frame memory. This module is the MedSAM2 side only; the update rules are

- ``modeling.kalman_memory.RKNMemoryUpdate`` (a modified RecursiveKalmanNet with detector observations),
- ``modeling.rde_memory.RDEMemoryUpdate`` (RDE-VOS, from its official code).

An update implements ``start(memory) -> covariance`` (``None`` if it keeps none) and
``update(state, covariance, memory, reliability) -> {"mean", "covariance", ["gain"]}`` (reliability:
MedSAM2's object-score logit [B, 1], the cosine similarity of the frame's
object pointer to the prompt frame's [B, 1], mask probability and image-feature change since the
previous frame at memory resolution [B, 1, H, W], and the prompt memory), and may define
``updates_at(frame_idx)`` (which frames are written), ``memory_slots(...)`` (which memories
are read) and ``keeps_recent`` (whether the newest written memory is kept for reading).

Detector observations (``detections``, {frame index: {"box", "confidence"}}, set per video) drive
re-detection (below) for an update with ``uses_detector``, or for any memory (MedSAM2's native
bank, RDE-VOS) with ``redetect_enabled``. An update with ``uses_detector`` also corrects MedSAM2's
object score in its decoder (``presence``, through MedSAM2's ``object_score_hook``) and rates the
tracker's mask before it is written (``reliability`` -> rho). Then, on a tracked frame:

- re-detection: if the detector is confident (>= ``REDETECT_CONFIDENCE``) about a box that disagrees
  with the mask (IoU < ``REDETECT_IOU``; a closed gate's empty mask disagrees with any box), the
  frame is decoded from that box on memory-free features, as on a prompt frame, and that mask is
  the frame's output and memory, gated by its own object score. Set on the training videos
  (seq1-12, with the first trained model): this fires on 39% of polyp frames, 65% of them lost,
  catches 82% of lost frames, and fires on 3.4% of empty frames. Neither the corrected presence
  (it closes the gate on lost frames) nor rho (it never fell below 0.5 there) gates it;
- the write is weighted by beta = presence x rho (presence x detector confidence when re-detected);
- the object pointer is gated: ptr <- no_obj_ptr + w (ptr - no_obj_ptr), w = rho (confidence when
  re-detected);
- the state is read at its effective age, which grows by one frame and shrinks with each write
  in proportion to the effective gain beta K.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from modeling.kalman_memory import box_iou, mask_box
from modeling.native_pointers import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor

CHECKPOINT_FORMAT = "adseg_kalman_memory_v6"  # name kept so existing checkpoints load
REDETECT_CONFIDENCE = 0.5  # fixed before any run with re-detection; checked on training videos only
REDETECT_IOU = 0.5
RDE_CHECKPOINT_FORMATS = ("adseg_kalman_memory_v5",)  # earlier RDE checkpoints; same parameters


def vision_feature_map(current_vision_feats, feat_sizes) -> torch.Tensor:
    feature = current_vision_feats[-1]
    return feature.permute(1, 2, 0).reshape(feature.size(1), -1, *feat_sizes[-1])


class MemoryState:
    """The prompt memory (anchor) and the recurrent state written by the memory update."""

    def __init__(self, anchor, position, pointer, frame_idx, update):
        self.anchor = anchor.float()
        self.position = position.float()
        self.pointer = pointer.float()  # the prompt frame's object pointer
        self.mean = self.anchor
        self.covariance = update.start(self.mean)
        self.last_frame_idx = frame_idx
        self.previous_feature = None  # image features of the previous frame, for the camera-motion measure
        self.recent = None  # newest memory written to the state, for updates that also read it (RDE)
        self.recent_frame_idx = None
        self.age = 0.0  # frames since the state's content was observed, weighted by the effective gain

    def tensors(self) -> list[torch.Tensor]:
        extra = [t for t in (self.covariance, self.recent) if isinstance(t, torch.Tensor)]
        return [self.anchor, self.position, self.pointer, self.mean, *extra]


class RecurrentMemory:
    """Put before a ``SAM2Base`` subclass; inactive while ``memory_update`` is None or
    ``recurrent_memory_enabled`` is False (then MedSAM2's native bank is used)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_update = None
        self.recurrent_memory_enabled = True
        self.last_memory_step = None
        self._memory_step = None
        self.detections = None
        self.redetect_enabled = False  # re-detection for a memory without detector heads (native, RDE-VOS)
        self.last_frame_cues = None
        self._detection = None  # the current frame's detector cue, while its decoder runs
        self.object_score_hook = self._presence_hook

    def _uses_detector(self) -> bool:
        return self._recurrent_memory_active() and getattr(self.memory_update, "uses_detector", False)

    def _redetection_on(self) -> bool:
        if self._uses_detector():
            return self.memory_update.config.get("learn_memory", True)
        return self.redetect_enabled

    def _presence_hook(self, object_score_logits, ious, low_res_multimasks):
        """MedSAM2's object score corrected by the memory update's presence network (tracked frames)."""
        detection = self._detection
        if detection is None:
            return object_score_logits
        batch = torch.arange(ious.size(0), device=ious.device)
        best = low_res_multimasks[batch, ious.argmax(-1)].detach()
        detection["ungated_mask"] = best.unsqueeze(1)  # the output candidate before the presence gate
        detection["mask_box_iou"] = box_iou(mask_box(best), detection["box"])
        if not self._uses_detector():
            return object_score_logits
        fused = self.memory_update.presence(object_score_logits.float(), detection, detection["mask_box_iou"])
        return fused.to(object_score_logits.dtype)

    def _recurrent_memory_active(self) -> bool:
        return self.recurrent_memory_enabled and self.memory_update is not None

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
        step = {"output_dict": output_dict, "frame_idx": frame_idx, "init": is_init_cond_frame}
        self._detection = None
        if (self._uses_detector() or self.redetect_enabled) and not is_init_cond_frame:
            if self.detections is None or frame_idx not in self.detections:
                raise ValueError(f"The memory update needs a detection for frame {frame_idx}.")
            self._detection = dict(self.detections[frame_idx])
        step["detection"] = self._detection
        if not self._recurrent_memory_active() or is_init_cond_frame:
            step["pix_feat"] = super()._prepare_memory_conditioned_features(
                frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse,
            )
            self._memory_step = step
            return step["pix_feat"]
        if track_in_reverse:
            raise ValueError("Recurrent memory supports forward propagation only.")
        state = output_dict["memory_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Recurrent memory requires consecutive forward frames.")
        batch = current_vision_feats[-1].size(1)
        feature = vision_feature_map(current_vision_feats, feat_sizes).float()
        change = None
        if state.previous_feature is not None:
            change = 1.0 - F.cosine_similarity(feature, state.previous_feature, dim=1).unsqueeze(1)
        state.previous_feature = feature
        step["feature_change"] = change  # camera motion, logged for the motion-stratified analysis
        memory_chunks, position_chunks = memory_tokens(
            self, memory_slots(self, state, frame_idx), state.position, current_vision_feats[-1].dtype,
        )
        num_obj_ptr_tokens = append_native_object_pointers(
            self, frame_idx, output_dict, num_frames, track_in_reverse,
            current_vision_feats[-1].device, batch, memory_chunks, position_chunks,
        )
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory_chunks, dim=0),
            memory_pos=torch.cat(position_chunks, dim=0),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        step["pix_feat"] = fused.permute(1, 2, 0).reshape(batch, self.hidden_dim, *feat_sizes[-1])
        self._memory_step = step
        return step["pix_feat"]

    def _redetect(self, current_vision_feats, feat_sizes, box):
        """Decode the detector box on memory-free features, as MedSAM2 does on a prompt frame."""
        batch, channels = current_vision_feats[-1].size(1), self.hidden_dim
        pix_feat = (current_vision_feats[-1] + self.no_mem_embed).permute(1, 2, 0).view(batch, channels, *feat_sizes[-1])
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *size)
            for x, size in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ] or None
        point_inputs = {
            "point_coords": (box.float() * self.image_size).view(batch, 2, 2),
            "point_labels": torch.tensor([[2, 3]], dtype=torch.int, device=box.device).expand(batch, -1),
        }
        self._detection = None  # the box decode is not a tracked decode: no presence correction
        return self._forward_sam_heads(
            backbone_features=pix_feat, point_inputs=point_inputs, mask_inputs=None,
            high_res_features=high_res_features, multimask_output=self._use_multimask(True, point_inputs),
        )

    def _assess(self, current_vision_feats, feat_sizes, high_res_masks, object_score_logits, current_out, step):
        """Rate the tracker's mask (an update with ``uses_detector``), re-detect if needed, and return
        the (possibly replaced) mask, score and the write's association beta and pointer weight
        (None without a reliability head)."""
        detection = step["detection"]
        confidence, mask_box_iou = detection["confidence"], detection["mask_box_iou"]
        rho = None
        step["cues"] = {"detection_confidence": confidence, "mask_box_iou": mask_box_iou}
        if self._uses_detector() and "memory_state" in step["output_dict"]:
            state = step["output_dict"]["memory_state"]
            similarity = F.cosine_similarity(current_out["obj_ptr"].float(), state.pointer).unsqueeze(1)
            rho_logit = self.memory_update.reliability(confidence, mask_box_iou, similarity)
            rho = torch.sigmoid(rho_logit)
            current_out["reliability_logit"] = rho_logit
            step["cues"]["rho"] = rho
        current_out["tracker_masks_high_res"] = high_res_masks.detach()
        current_out["ungated_mask"] = detection["ungated_mask"]
        redetect = bool(
            self._redetection_on()
            and (confidence >= REDETECT_CONFIDENCE).all() and (mask_box_iou < REDETECT_IOU).all()
        )
        step["redetected"] = redetect
        self.last_frame_cues = {"redetected": redetect, **step["cues"]}
        if not redetect:
            if rho is None:
                return high_res_masks, object_score_logits, None, None
            return high_res_masks, object_score_logits, torch.sigmoid(object_score_logits.float()) * rho, rho
        _, _, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self._redetect(
            current_vision_feats, feat_sizes, detection["box"],
        )
        current_out.update(pred_masks=low_res_masks, pred_masks_high_res=high_res_masks, obj_ptr=obj_ptr,
                           ungated_mask=low_res_masks.detach())
        if "object_score_logits" in current_out:
            current_out["object_score_logits"] = object_score_logits
            current_out["iou_predictions"] = ious.max(dim=-1, keepdim=True).values
        for key, value in (("multistep_pred_masks_high_res", high_res_masks), ("multistep_object_score_logits", object_score_logits)):
            if key in current_out:  # SAM2Train's record of the final decode
                if isinstance(current_out[key], list):
                    current_out[key][-1] = value
                else:
                    current_out[key] = value
        if rho is None:
            return high_res_masks, object_score_logits, None, None
        return high_res_masks, object_score_logits, torch.sigmoid(object_score_logits.float()) * confidence, confidence

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        pending, association, pointer_weight = self._memory_step, None, None
        if pending is not None and not pending["init"] and pending.get("detection") is not None:
            high_res_masks, object_score_logits, association, pointer_weight = self._assess(
                current_vision_feats, feat_sizes, high_res_masks, object_score_logits, current_out, pending,
            )
        super()._encode_memory_in_output(
            current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
            high_res_masks, object_score_logits, current_out,
        )
        if current_out["maskmem_features"] is not None:
            # MedSAM2's video predictor stores frame memories in bf16; round here so training matches it.
            features = current_out["maskmem_features"]
            current_out["maskmem_features"] = features.to(torch.bfloat16).to(features.dtype)
        step, self._memory_step = self._memory_step, None
        if step is None:
            return
        current_out["pix_feat_with_mem"] = step["pix_feat"]
        candidate = current_out.get("maskmem_features")
        if not self._recurrent_memory_active() or candidate is None:
            return
        output_dict, frame_idx = step["output_dict"], step["frame_idx"]
        if step["init"]:
            output_dict["memory_state"] = MemoryState(
                candidate, current_out["maskmem_pos_enc"][-1], current_out["obj_ptr"], frame_idx, self.memory_update,
            )
            output_dict["memory_state"].previous_feature = vision_feature_map(current_vision_feats, feat_sizes).float()
            return
        state = output_dict["memory_state"]
        updates_at = getattr(self.memory_update, "updates_at", None)
        if updates_at is not None and not updates_at(frame_idx):
            # Not a memory frame (RDE ``mem_every``): the state is read but not written.
            status, updated = "held", {"mean": state.mean, "covariance": state.covariance}
        else:
            status = "updated"
            reliability = {
                "object_score": object_score_logits.float(),
                "pointer_similarity": F.cosine_similarity(current_out["obj_ptr"].float(), state.pointer).unsqueeze(1),
                "prompt_memory": state.anchor,
                "mask_probability": F.interpolate(torch.sigmoid(current_out["pred_masks"].float()),
                                                  size=candidate.shape[-2:], mode="area"),
            }
            if step.get("feature_change") is not None:
                reliability["feature_change"] = F.interpolate(step["feature_change"], size=candidate.shape[-2:], mode="area")
            if step["detection"] is not None:
                reliability.update(detection_box=step["detection"]["box"],
                                   detection_confidence=step["detection"]["confidence"],
                                   mask_box_iou=step["detection"]["mask_box_iou"], association=association)
            updated = self.memory_update.update(state.mean, state.covariance, candidate.float(), reliability)
            if pointer_weight is not None and self.memory_update.config.get("learn_memory", True):
                # an unreliable frame's pointer moves toward MedSAM2's no-object pointer
                pointer = current_out["obj_ptr"]
                current_out["obj_ptr"] = self.no_obj_ptr + pointer_weight.to(pointer.dtype) * (pointer - self.no_obj_ptr)
            if "gain" in updated:
                state.age = (1.0 - float(updated["gain"].mean())) * (state.age + 1.0)
        if not torch.isfinite(updated["mean"]).all():
            raise FloatingPointError(f"Recurrent memory became non-finite at frame {frame_idx}.")
        state.mean, state.covariance = updated["mean"], updated["covariance"]
        state.last_frame_idx = frame_idx
        if status == "updated" and getattr(self.memory_update, "keeps_recent", False):
            state.recent, state.recent_frame_idx = candidate.float(), frame_idx
        updated.update(status=status, frame_idx=frame_idx, presence=torch.sigmoid(object_score_logits.float()),
                       redetected=step.get("redetected", False), age=state.age, **step.get("cues", {}))
        if step.get("feature_change") is not None:
            updated["feature_change"] = step["feature_change"]
        current_out["memory"] = updated
        self.last_memory_step = updated


def _wait_for_offload(device: torch.device) -> None:
    # MedSAM2 offloads with non_blocking=True; on MPS the host copy can still be in flight.
    if device.type == "mps":
        torch.mps.synchronize()


class RecurrentMemoryVideoPredictor(RecurrentMemory, SAM2VideoPredictor):
    """MedSAM2 video predictor with the prompt memory plus one recurrent state."""

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("recurrent_memory", False):
            if self.redetect_enabled:  # MedSAM2's native bank with re-detection
                return super().propagate_in_video_preflight(inference_state)
            raise ValueError("Recurrent memory must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Recurrent memory currently supports exactly one object.")
        if self.memory_update is None:
            raise RuntimeError("Recurrent memory requires a loaded memory update.")
        super().propagate_in_video_preflight(inference_state)
        outputs = inference_state["output_dict"]
        anchor_idx = inference_state["memory_anchor_frame_idx"]
        if set(outputs["cond_frame_outputs"]) != {anchor_idx}:
            raise ValueError("Recurrent memory supports one initial prompt frame.")
        anchor = outputs["cond_frame_outputs"][anchor_idx]
        device = anchor["obj_ptr"].device
        _wait_for_offload(device)
        features = anchor["maskmem_features"].to(device)
        outputs["memory_state"] = MemoryState(
            features, anchor["maskmem_pos_enc"][-1].to(device), anchor["obj_ptr"], anchor_idx, self.memory_update,
        )
        _, _, vision_feats, _, feat_sizes = self._get_image_feature(inference_state, anchor_idx, 1)
        outputs["memory_state"].previous_feature = vision_feature_map(vision_feats, feat_sizes).float()
        anchor["maskmem_features"] = None
        anchor["maskmem_pos_enc"] = None

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("memory_state", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        self.last_memory_step = self.last_frame_cues = None
        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        step = self.last_memory_step
        if step is None and self.last_frame_cues is not None and not kwargs["is_init_cond_frame"]:
            current_out["memory_trace"] = {key: float(value.mean()) if torch.is_tensor(value) else value
                                           for key, value in self.last_frame_cues.items()}
        if step is not None and not kwargs["is_init_cond_frame"]:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None
            current_out["memory_trace"] = {
                "memory_status": step["status"],
                "presence": float(step["presence"].mean()),
                "gain_mean": float(step["gain"].mean()) if "gain" in step else float("nan"),
                "redetected": bool(step.get("redetected", False)),
                **{key: float(step[key].mean()) for key in ("rho", "detection_confidence", "mask_box_iou") if key in step},
                "feature_change_mean": float(step["feature_change"].mean()) if "feature_change" in step else float("nan"),
            }
        return current_out, pred_masks


def memory_slots(model, state, frame_idx):
    """(memory map, frames ago) for every memory slot read at ``frame_idx``. By default the
    prompt anchor and the recurrent state, read at the state's effective age (at least one frame;
    not detached: training reaches each write through later frames' segmentation); an update may
    define its own (``RDEMemoryUpdate.memory_slots``)."""
    slots = getattr(model.memory_update, "memory_slots", None)
    if slots is None:
        return [(state.anchor, 0), (state.mean, max(1, round(state.age)))]
    return slots(state, frame_idx, model.num_maskmem)


def memory_tokens(model, slots, position, dtype=None):
    """Memory and position tokens; a slot ``t`` frames ago gets MedSAM2's temporal encoding of
    that distance (0 = prompt frame, capped at the oldest slot)."""
    dtype = dtype or slots[0][0].dtype
    tokens = lambda tensor: tensor.to(dtype).flatten(2).permute(2, 0, 1)
    memory, positions = [], []
    for features, frames_ago in slots:
        frames_ago = min(frames_ago, model.num_maskmem - 1)
        memory.append(tokens(features))
        positions.append(tokens(position + model.maskmem_tpos_enc[model.num_maskmem - frames_ago - 1].view(1, -1, 1, 1)))
    return memory, positions


def save_memory_update(update, path) -> None:
    """Save a trained memory update (modified RKN or RDE-VOS)."""
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "kind": update.CHECKPOINT_KIND,
        "memory_channels": update.memory_channels,
        "image_channels": update.image_channels,
        "update_config": update.config,
        "state_dict": update.state_dict(),
    }, path)


def load_memory_update(path, model, device):
    """Memory update from ``save_memory_update``, checked against ``model``'s MedSAM2 widths."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_format, kind = checkpoint.get("format"), checkpoint.get("kind")
    if (checkpoint["memory_channels"], checkpoint["image_channels"]) != (model.mem_dim, model.hidden_dim):
        raise ValueError(f"{path} does not match this MedSAM2 model.")
    config, state_dict = checkpoint["update_config"], checkpoint["state_dict"]
    if kind == "rde" and checkpoint_format in (CHECKPOINT_FORMAT, *RDE_CHECKPOINT_FORMATS):
        from modeling.rde_memory import RDEMemoryUpdate

        update_class = RDEMemoryUpdate
        config, state_dict = RDEMemoryUpdate.upgrade(config, state_dict)
    elif kind == "rkn_detector" and checkpoint_format == CHECKPOINT_FORMAT:
        from modeling.kalman_memory import RKNMemoryUpdate

        update_class = RKNMemoryUpdate
    else:
        raise ValueError(f"{path} is not a modified-RKN or RDE-VOS memory-update checkpoint.")
    update = update_class(model.mem_dim, model.hidden_dim, **config).to(device)
    update.load_state_dict(state_dict)
    return update.eval()

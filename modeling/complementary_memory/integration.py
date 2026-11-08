"""One residual after SAM2's complete native memory attention.

No global SDPA patch, native weight mutation, decoder replacement or second
prediction pass. Fail closed if labels and actual native memory order disagree.
"""

import inspect
import hashlib
import json

import torch
import torch.nn.functional as F

from modeling.native_memory_layout import selected_memory_frames


def _tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode())
    hasher.update(str(tuple(tensor.shape)).encode())
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _memory_alignment_diagnostic(context, span, output, actual, expected):
    """Describe a failed native-bank equality check without masking it.

    The raw-byte comparison distinguishes a real value/order mismatch from
    ``torch.equal`` returning false for matching NaN payloads. Non-finite
    memory still fails the check; the extra fields make the failure actionable.
    """
    report = {
        "current_frame": context["frame"],
        "object_id": context["object_id"],
        "stored_frame": span["frame"],
        "temporal_position": span["t_pos"],
        "memory_slice": [span["start"], span["stop"]],
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "stored_dtype": str(output["maskmem_features"].dtype),
        "expected_dtype_after_cast": str(expected.dtype),
        "value_equal": bool(torch.equal(actual, expected)),
    }
    if actual.shape == expected.shape and actual.dtype == expected.dtype:
        actual_bytes = actual.detach().contiguous().view(torch.uint8)
        expected_bytes = expected.detach().contiguous().view(torch.uint8)
        report["raw_bytes_equal"] = bool(torch.equal(actual_bytes, expected_bytes))
        actual_finite = torch.isfinite(actual)
        expected_finite = torch.isfinite(expected)
        report["actual_finite_count"] = int(actual_finite.sum().item())
        report["expected_finite_count"] = int(expected_finite.sum().item())
        finite_pairs = actual_finite & expected_finite
        if bool(finite_pairs.any()):
            difference = (actual.float() - expected.float()).abs()
            report["max_abs_finite_difference"] = float(difference[finite_pairs].max().item())
        else:
            report["max_abs_finite_difference"] = None
    return json.dumps(report, sort_keys=True)


class NativeComplementaryRead:
    def __init__(self, model, reader, mode="regional", *, inference_state=None,
                 trace_only=False, record_update_norm=False):
        if mode not in ("regional", "generic"):
            raise ValueError("Expected regional or generic")
        if trace_only and reader is not None:
            raise ValueError("Trace-only mode cannot alter the native read")
        self.model, self.reader, self.mode = model, reader, mode
        self.inference_state = inference_state
        self.trace_only = trace_only
        self.record_update_norm = record_update_norm
        self.calls, self.handles = [], []
        self.context = None

    def _prepare(self, *args, **kwargs):
        data = self.signature.bind(*args, **kwargs)
        data.apply_defaults()
        data = data.arguments
        if self.context is not None:
            raise RuntimeError("Nested memory preparation")
        if data["is_init_cond_frame"]:
            return self.original(*args, **kwargs)
        if data["track_in_reverse"]:
            raise ValueError("This prospective experiment uses forward propagation only")
        bank, frame = data["output_dict"], int(data["frame_idx"])
        spans = selected_memory_frames(
            self.model, frame, bank, data["num_frames"], track_in_reverse=False
        )
        owner = None
        if self.inference_state is not None:
            owners = [i for i, item in self.inference_state["output_dict_per_obj"].items()
                      if item is bank]
            if len(owners) != 1:
                raise RuntimeError("Bank must belong to one persistent object")
            owner = self.inference_state["obj_idx_to_id"][owners[0]]
        outputs, masks = [], []
        for span in spans:
            index = span["frame"]
            if index >= frame:
                raise RuntimeError("Current/future mask entered memory summaries")
            output = bank["cond_frame_outputs"].get(index)
            if output is None:
                output = bank["non_cond_frame_outputs"].get(index)
            if output is None:
                raise RuntimeError(f"Selected memory frame {index} has no stored output")
            feature = output["maskmem_features"]
            mask = output["pred_masks"]
            if (feature.ndim != 4 or feature.shape[1] != 64
                    or mask.shape != (feature.shape[0], 1, 256, 256)
                    or span["stop"] - span["start"] != feature.shape[-2] * feature.shape[-1]):
                raise RuntimeError("Stored mask and memory batch disagree")
            masks.append(F.adaptive_avg_pool2d(mask.float().sigmoid(), feature.shape[-2:])[:, 0])
            outputs.append(output)
        self.context = {"frame": frame, "object_id": owner, "spans": spans,
                        "outputs": outputs, "probabilities": torch.stack(masks, dim=1),
                        "layers": []}
        before = len(self.calls)
        try:
            result = self.original(*args, **kwargs)
            if len(self.calls) != before + 1:
                raise RuntimeError("Expected exactly one complementary read per frame/object")
            return result
        finally:
            self.context = None

    def _layer(self, index):
        def observed(module, args, kwargs, output):
            if self.context is not None:
                self.context["layers"].append(index)
        return observed

    def _read(self, module, args, kwargs, native):
        context = self.context
        if context is None:  # native prompt path
            return native
        if args or context["layers"] != [0, 1, 2, 3]:
            raise RuntimeError("Expected native keyword call and all four layers")
        memory, positions = kwargs["memory"], kwargs["memory_pos"]
        pointers = int(kwargs["num_obj_ptr_tokens"])
        spatial = sum(span["stop"] - span["start"] for span in context["spans"])
        if memory.shape != positions.shape or memory.shape[0] != spatial + pointers:
            raise RuntimeError("Native spatial/pointer count differs from selected spans")
        if context["layers"] != [0, 1, 2, 3]:
            raise RuntimeError("Expected one complete native four-layer memory read")
        pointer_memory = memory[spatial:]
        pointer_positions = positions[spatial:]
        call = {
            "frame": context["frame"], "object_id": context["object_id"],
            "layers": context["layers"].copy(), "memory_frames": context["spans"],
            "spatial_keys": spatial, "pointer_keys": pointers,
            "pointer_input_sha256": _tensor_sha256(pointer_memory),
            "pointer_position_sha256": _tensor_sha256(pointer_positions),
            "stored_mask_alignment": True, "native_positions_verified": True,
        }
        values, positional = [], []
        for span, output in zip(context["spans"], context["outputs"]):
            feature = output["maskmem_features"].to(memory.device)
            expected = feature.flatten(2).permute(2, 0, 1).to(memory.dtype)
            actual = memory[span["start"]:span["stop"]]
            if not torch.equal(actual, expected):
                details = _memory_alignment_diagnostic(
                    context, span, output, actual, expected
                )
                raise RuntimeError(
                    "Stored-mask order differs from actual native memory: " + details
                )
            expected_pos = output["maskmem_pos_enc"][-1].to(positions.device).flatten(2).permute(2, 0, 1)
            expected_pos = expected_pos + self.model.maskmem_tpos_enc[self.model.num_maskmem - span["t_pos"] - 1]
            actual_pos = positions[span["start"]:span["stop"]]
            if not torch.equal(actual_pos, expected_pos.to(positions.dtype)):
                raise RuntimeError("Native spatial/temporal positions differ")
            values.append(actual.permute(1, 2, 0).reshape_as(feature))
            positional.append(actual_pos.permute(1, 2, 0).reshape_as(feature))
        if self.trace_only:
            self.calls.append(call)
            return native
        query_pos = kwargs["curr_pos"]
        if isinstance(query_pos, list):
            query_pos = query_pos[0]
        updated = self.reader(native.transpose(0, 1), query_pos.transpose(0, 1),
                              torch.stack(values, 1), torch.stack(positional, 1),
                              context["probabilities"].to(memory.device), self.mode)
        call["summary_tokens"] = len(values) * self.reader.summaries
        if self.record_update_norm:
            delta = updated - native.transpose(0, 1)
            call["reader_update_l1"] = float(delta.detach().float().abs().sum())
            call["finite_reader_update"] = bool(torch.isfinite(delta).all())
        self.calls.append(call)
        return updated.transpose(0, 1)

    def __enter__(self):
        if (any(p.requires_grad for p in self.model.parameters()) or self.model.training
                or len(self.model.memory_attention.layers) != 4 or self.model.num_maskmem <= 0):
            raise RuntimeError("Requires frozen eval SAM2 with four layers and enabled memory")
        if getattr(self.model, "_complementary_reader_active", False):
            raise RuntimeError("Complementary reader already installed")
        self.model._complementary_reader_active = True
        self.original = self.model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.original)
        self.model._prepare_memory_conditioned_features = self._prepare
        for index, layer in enumerate(self.model.memory_attention.layers):
            self.handles.append(layer.register_forward_hook(self._layer(index), with_kwargs=True))
        self.handles.append(self.model.memory_attention.register_forward_hook(self._read, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model._prepare_memory_conditioned_features = self.original
        del self.model._complementary_reader_active
        self.context = None

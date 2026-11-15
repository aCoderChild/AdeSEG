"""Shadow-measure attention changes from pinned token mergers.

This observer computes candidate compressed-attention outputs beside the
native SAM2 read, records their distortion, and always returns the native
output to the model. It reuses the existing ToMe/PiToMe adapters; it is not a
new merger or a segmentation method.
"""

import hashlib
from pathlib import Path

import torch

from modeling.author_token_matching import use_matching
from modeling.pitome_matching import use_pitome
from modeling.region_constrained_tome import (
    NativeRegionConstrainedToMe,
    compress_memory_kv,
)


DEFAULT_CONDITIONS = (
    {"matcher": "tome", "mode": "regional", "merge_divisor": 2},
    {"matcher": "tome", "mode": "global", "merge_divisor": 2},
    {"matcher": "tome", "mode": "regional", "merge_divisor": 4},
    {"matcher": "tome", "mode": "global", "merge_divisor": 4},
    {"matcher": "pitome", "mode": "regional", "merge_divisor": 4},
    {"matcher": "pitome", "mode": "global", "merge_divisor": 4},
)


def _sha256_tensor(tensor):
    array = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _norm_summary(tensor, *, weights=None):
    values = tensor.detach().float()
    norms = values.norm(dim=-1).reshape(-1)
    if weights is None:
        weighted_mean = norms.mean()
    else:
        mass = torch.as_tensor(weights, device=norms.device, dtype=norms.dtype).reshape(-1)
        token_count = values.shape[-2]
        if mass.numel() != token_count:
            raise ValueError("Norm weights must align with the token dimension")
        mass = mass.repeat(norms.numel() // token_count)
        if not torch.isfinite(mass).all() or (mass < 0).any():
            raise ValueError("Norm weights must be finite, nonnegative and token-aligned")
        weighted_mean = (norms * mass).sum() / mass.sum().clamp_min(1)
    return {
        "mean_l2": float(norms.mean()),
        "rms_l2": float(values.square().sum(dim=-1).mean().sqrt()),
        "mass_weighted_mean_l2": float(weighted_mean),
    }


def _output_distortion(native, shadow):
    # Keep FP64 references precise; FP16/BF16/FP32 retain FP32 accumulation.
    dtype = torch.float64 if torch.float64 in (native.dtype, shadow.dtype) else torch.float32
    native32, shadow32 = native.detach().to(dtype), shadow.detach().to(dtype)
    difference = shadow32 - native32
    native_energy = native32.square().sum()
    shadow_energy = shadow32.square().sum()
    denominator = torch.sqrt(native_energy * shadow_energy)
    cosine = None if denominator == 0 else float((native32 * shadow32).sum() / denominator)
    return {
        "relative_squared_error": (
            None if native_energy == 0 else float(difference.square().sum() / native_energy)
        ),
        "zero_native_reference": bool(native_energy == 0),
        "cosine_similarity": cosine,
        "zero_cosine_denominator": bool(denominator == 0),
        "max_absolute_difference": float(difference.abs().max()),
    }


class AttentionDistortionObserver(NativeRegionConstrainedToMe):
    """Run the frozen mergers as shadows and preserve every native SDPA output."""

    def __init__(self, model, *, pitome_repository, conditions=DEFAULT_CONDITIONS,
                 calibration_shadow=False):
        super().__init__(model, mode="regional", passthrough=True)
        self.pitome_repository = Path(pitome_repository)
        self.conditions = tuple(dict(condition) for condition in conditions)
        self.calibration_shadow = calibration_shadow
        if not self.conditions:
            raise ValueError("At least one shadow condition is required")
        for condition in self.conditions:
            if condition.get("matcher") not in ("tome", "pitome"):
                raise ValueError("Shadow matcher must be a pinned ToMe or PiToMe adapter")
            if condition.get("mode") not in ("regional", "global"):
                raise ValueError("Shadow mode must be regional or global")
            if type(condition.get("merge_divisor")) is not int or condition["merge_divisor"] < 2:
                raise ValueError("Shadow merge divisor must be an integer >= 2")
            if condition["matcher"] == "pitome" and condition["merge_divisor"] > 4:
                raise ValueError("Pinned PiToMe adapter is verified only through quarter removal")
        self.records = []
        self.sdpa_calls_by_target = {}

    def _backend_context(self, matcher):
        if matcher == "tome":
            return use_matching("tome")
        return use_pitome(self.pitome_repository)

    @staticmethod
    def _shadow_sdpa(query, key, value, mass_bias, original_sdpa, kwargs):
        if key.shape[-2] == mass_bias.shape[-1] and torch.equal(
            mass_bias, torch.zeros_like(mass_bias)
        ):
            return original_sdpa(query, key, value, **kwargs)
        return original_sdpa(
            query,
            key,
            value,
            attn_mask=mass_bias.to(dtype=query.dtype),
            dropout_p=0.0,
            is_causal=False,
            scale=kwargs.get("scale"),
            enable_gqa=kwargs.get("enable_gqa", False),
        )

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.0) != 0.0:
            raise RuntimeError("Unsupported SDPA options in the SAM2 shadow observer")
        if kwargs.get("is_causal", False):
            raise RuntimeError("SAM2 memory attention is expected to be noncausal")
        if self.memberships is None or not self.spans:
            raise RuntimeError("Native memory-frame roles were not prepared for this read")

        layer, pointers = self.active
        native = self.original_sdpa(query, key, value, *args, **kwargs)
        before = tuple(tensor.detach().clone() for tensor in (query, key, value))
        spatial_count = key.shape[-2] - pointers
        if spatial_count != self.memberships.shape[1]:
            raise RuntimeError("Native memory roles do not align with projected attention keys")
        target_key = (self.frame, self.current_object_id)
        target_calls = self.sdpa_calls_by_target.setdefault(target_key, [])
        if layer in target_calls:
            raise RuntimeError(f"Layer {layer} ran more than once for target {target_key}")

        key_spatial = key[..., :spatial_count, :]
        value_spatial = value[..., :spatial_count, :]
        pointer_keys = key[..., spatial_count:, :]
        pointer_values = value[..., spatial_count:, :]
        condition_records = []
        for condition in self.conditions:
            with self._backend_context(condition["matcher"]):
                compressed_key, compressed_value, mass_bias, audit = compress_memory_kv(
                    key,
                    value,
                    self.memberships.to(device=key.device),
                    self.spans,
                    mode=condition["mode"],
                    pointers=pointers,
                    merge_divisor=condition["merge_divisor"],
                )
            shadow = self._shadow_sdpa(
                query,
                compressed_key,
                compressed_value,
                mass_bias,
                self.original_sdpa,
                kwargs,
            )
            if not torch.isfinite(shadow).all():
                raise RuntimeError("Shadow attention produced nonfinite output")
            merged_spatial = compressed_key.shape[-2] - pointers
            # SDPA receives log(cluster_size); norm weighting needs cluster_size.
            masses = mass_bias.reshape(-1)[:merged_spatial].float().exp()
            if pointers and (
                not torch.equal(compressed_key[..., -pointers:, :], pointer_keys)
                or not torch.equal(compressed_value[..., -pointers:, :], pointer_values)
            ):
                raise RuntimeError("Shadow compressor changed native object-pointer tensors")
            condition_records.append({
                **condition,
                "spatial_keys_before": spatial_count,
                "spatial_keys_after": merged_spatial,
                "pointer_keys": pointers,
                "key_norm_before": _norm_summary(key_spatial),
                "key_norm_after_mass_weighted": _norm_summary(compressed_key[..., :merged_spatial, :], weights=masses),
                "value_norm_before": _norm_summary(value_spatial),
                "value_norm_after_mass_weighted": _norm_summary(compressed_value[..., :merged_spatial, :], weights=masses),
                "pointer_key_norm_before": _norm_summary(pointer_keys) if pointers else None,
                "pointer_key_norm_after": _norm_summary(compressed_key[..., -pointers:, :]) if pointers else None,
                "pointer_value_norm_before": _norm_summary(pointer_values) if pointers else None,
                "pointer_value_norm_after": _norm_summary(compressed_value[..., -pointers:, :]) if pointers else None,
                "output_distortion": _output_distortion(native, shadow),
                "native_pointer_key_sha256": _sha256_tensor(pointer_keys),
                "shadow_pointer_key_sha256": _sha256_tensor(compressed_key[..., -pointers:, :]) if pointers else None,
                "native_pointer_value_sha256": _sha256_tensor(pointer_values),
                "shadow_pointer_value_sha256": _sha256_tensor(compressed_value[..., -pointers:, :]) if pointers else None,
                "memory_audit": audit,
            })
            if self.calibration_shadow:
                # Isolate scaling on the SAME actual merged tensors, without a
                # second matcher call or a separate propagated memory history.
                from modeling.stat_calibrated_tome import calibrate_merged_kv

                merged_inputs = (compressed_key, compressed_value, mass_bias)
                fingerprints = [_sha256_tensor(t) for t in merged_inputs]
                calibrated_key, calibrated_value, calibrated_bias, coefficient = (
                    calibrate_merged_kv(
                        *merged_inputs, spatial_before=spatial_count,
                        spatial_after=merged_spatial,
                    )
                )
                calibrated_output = self._shadow_sdpa(
                    query, calibrated_key, calibrated_value, calibrated_bias,
                    self.original_sdpa, kwargs,
                )
                if not torch.isfinite(calibrated_output).all():
                    raise RuntimeError("Calibrated shadow produced nonfinite output")
                if fingerprints != [_sha256_tensor(t) for t in merged_inputs]:
                    raise RuntimeError("Calibration modified shared merged inputs")
                if pointers and (
                    not torch.equal(calibrated_key[..., -pointers:, :], pointer_keys)
                    or not torch.equal(calibrated_value[..., -pointers:, :], pointer_values)
                ):
                    raise RuntimeError("Calibration changed native pointers")
                condition_records[-1]["calibrated_shadow"] = {
                    "shared_merge_input_sha256": fingerprints,
                    "key_logit_and_value_scale": coefficient,
                    "output_distortion": _output_distortion(native, calibrated_output),
                    "returns_native_output": True,
                }

        for old, current in zip(before, (query, key, value), strict=True):
            if not torch.equal(old, current):
                raise RuntimeError("Shadow measurement modified native Q/K/V input")
        target_calls.append(layer)
        self.records.append({
            "frame": self.frame,
            "object_id": self.current_object_id,
            "layer": layer,
            "q_shape": list(query.shape),
            "k_shape": list(key.shape),
            "v_shape": list(value.shape),
            "native_output_sha256": _sha256_tensor(native),
            "source_frames": [
                {"frame": int(span["frame"]), "t_pos": int(span["t_pos"]),
                 "start": int(span["start"]), "stop": int(span["stop"])}
                for span in self.spans
            ],
            "conditions": condition_records,
        })
        return native

    def validate_complete(self, expected_targets):
        """Require exactly one native memory read at each of four layers/target."""
        expected = {tuple(target) for target in expected_targets}
        observed = set(self.sdpa_calls_by_target)
        if observed != expected:
            raise RuntimeError(f"Observed targets differ: expected={expected}, observed={observed}")
        for target, layers in self.sdpa_calls_by_target.items():
            if sorted(layers) != [0, 1, 2, 3]:
                raise RuntimeError(f"Expected layers 0..3 exactly once at {target}, got {layers}")
        return True

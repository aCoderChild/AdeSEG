"""Statistically calibrated ToMe reads for frozen SAM2 memory attention.

This adapts the fixed scaling equations from ``Statistically Calibrated
Scaling for Token Merging in Transformers`` (ICML 2026) to the existing
pinned-ToMe SAM2 wrapper. It changes only compressed spatial K/V at read time:
K is scaled to shrink merged logits, V is scaled to calibrate output energy,
and the additive log-mass bias is scaled toward the cluster-size prior.
Native pointer tokens and all pretrained SAM2 weights remain untouched.

The equations are a paper-based adaptation; the cited paper's linked code
repository is empty. This module is a research candidate, not a validated
segmentation improvement or a Seq5-specific recovery method.
"""

import math

import torch

from modeling.region_constrained_tome import NativeRegionConstrainedToMe


def calibrate_merged_kv(keys, values, log_mass_bias, *, spatial_before,
                        spatial_after):
    """Apply the paper's radial/angular calibration to compressed spatial K/V.

    ``keys`` and ``values`` are [B,H,N,D], with unchanged object pointers in
    the tail. ``log_mass_bias`` is the additive log(cluster-size) SDPA mask
    returned by the existing ToMe compressor. The paper's retention ratio is
    computed over the full spatial bank; pointer entries are excluded.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected aligned [B,H,N,D] compressed K/V tensors")
    if not 0 < spatial_after <= spatial_before:
        raise ValueError("Invalid spatial token counts")
    if spatial_after > keys.shape[-2]:
        raise ValueError("Spatial token count exceeds compressed K/V length")
    if log_mass_bias.shape[-1] != keys.shape[-2]:
        raise ValueError("Attention mass bias does not align with compressed K/V")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Compressed K/V must be finite")
    if not torch.isfinite(log_mass_bias).all():
        raise ValueError("Attention mass bias must be finite")

    if spatial_after == spatial_before:
        return keys, values, log_mass_bias, 1.0

    retention = spatial_after / float(spatial_before)
    coefficient = math.sqrt(retention)
    spatial_keys = keys[..., :spatial_after, :] * coefficient
    spatial_values = values[..., :spatial_after, :] * coefficient
    pointer_keys = keys[..., spatial_after:, :]
    pointer_values = values[..., spatial_after:, :]
    calibrated_keys = torch.cat((spatial_keys, pointer_keys), dim=-2)
    calibrated_values = torch.cat((spatial_values, pointer_values), dim=-2)
    # T(z, n) = sqrt(r) * z + (1-sqrt(r)) * log(n).
    # Scaling K implements sqrt(r)*z; this scales the existing log(n) term.
    calibrated_bias = log_mass_bias * (1.0 - coefficient)
    return calibrated_keys, calibrated_values, calibrated_bias, coefficient


class StatisticallyCalibratedToMe(NativeRegionConstrainedToMe):
    """Apply paper-derived calibration after the existing fixed-budget merge."""

    # The v1 runner regenerates matches on each arm's propagated features.
    # It cannot satisfy the protocol's exact cross-calibration assignment replay.
    # Keep real-frame calibration gates blocked until replay is implemented.
    supports_frozen_assignment_replay = False

    def _compress_memory(self, keys, values, memberships, spans, pointers,
                         merge_plan):
        merged_keys, merged_values, log_mass_bias, audit = super()._compress_memory(
            keys, values, memberships, spans, pointers, merge_plan
        )
        calibrated = calibrate_merged_kv(
            merged_keys, merged_values, log_mass_bias,
            spatial_before=audit["spatial_keys_before"],
            spatial_after=audit["spatial_keys_after"],
        )
        keys, values, bias, coefficient = calibrated
        audit = {
            **audit,
            "calibration": "icml2026_radial_angular",
            "spatial_retention_ratio": (
                audit["spatial_keys_after"] / audit["spatial_keys_before"]
            ),
            "key_logit_and_value_scale": coefficient,
            "log_mass_bias_scale": 1.0 - coefficient,
        }
        return keys, values, bias, audit

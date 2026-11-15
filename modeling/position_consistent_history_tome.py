"""Fixed-budget long-history ToMe with position-consistent SAM2 memory reads.

This candidate reuses the repository's pinned-author ToMe integration and
position-indexed SAM2 RoPE path. It is isolated from the active post-RoPE run.
It addresses spatial RoPE mixing; it does not claim to solve temporal-slot
aliasing or improve Dice.
"""

import math

import torch

from modeling.position_consistent_tome import PositionConsistentRegionToMe
from modeling.wider_history_tome import (
    MAP_TOKENS,
    MAX_NONCOND_MAPS,
    NATIVE_NONCOND_MAPS,
    allocate_history_budget,
)


def build_fixed_budget_merge_plan(
    spans,
    *,
    map_tokens=MAP_TOKENS,
    native_noncond_maps=NATIVE_NONCOND_MAPS,
    max_noncond_maps=MAX_NONCOND_MAPS,
    window_size=4,
    allocation="uniform",
):
    """Return exact ToMe merge counts per source frame and local window.

    The output format is accepted by ``PositionConsistentRegionToMe`` in
    generic mode. Quotas come from the frozen long-history allocation rule;
    integer remainders are spread over windows in raster order.
    """
    if not spans or window_size <= 0:
        raise ValueError("Expected spans and a positive window size")
    quotas = allocate_history_budget(
        spans,
        map_tokens=map_tokens,
        native_noncond_maps=native_noncond_maps,
        max_noncond_maps=max_noncond_maps,
        allocation=allocation,
    )

    plan = {}
    seen_frames = set()
    cursor = 0
    for span_index, span in enumerate(spans):
        frame = int(span["frame"])
        if frame in seen_frames:
            raise ValueError(f"Duplicate memory frame in spans: {frame}")
        seen_frames.add(frame)

        start, stop = int(span["start"]), int(span["stop"])
        source_tokens = stop - start
        side = math.isqrt(source_tokens)
        if (
            start != cursor
            or side * side != source_tokens
            or side % window_size
        ):
            raise ValueError("Spans must be contiguous square grids tiled by windows")
        cursor = stop
        target_tokens = int(quotas[span_index])
        if target_tokens <= 0 or target_tokens > source_tokens:
            raise ValueError("Per-frame target token quota is invalid")

        merge_count = source_tokens - target_tokens
        window_count = (side // window_size) ** 2
        base, remainder = divmod(merge_count, window_count)
        frame_plan = {
            window_index: base + int(
                ((window_index + 1) * remainder) // window_count
                > (window_index * remainder) // window_count
            )
            for window_index in range(window_count)
        }
        local_limit = window_size * window_size // 2
        if any(count > local_limit for count in frame_plan.values()):
            raise ValueError("Requested quota exceeds ToMe's local 50% merge limit")
        if sum(frame_plan.values()) != merge_count:
            raise RuntimeError("Window merge plan does not conserve the frame quota")
        plan[frame] = frame_plan

    if cursor != sum(int(span["stop"]) - int(span["start"]) for span in spans):
        raise RuntimeError("Memory spans are not contiguous")
    return plan


def extend_temporal_position_table(original, max_noncond_maps):
    """Extend SAM2's six learned age slots by reusing its oldest slot.

    The six native learned age embeddings are kept in their original order;
    only older positions reuse the native oldest embedding. This mirrors the
    active wider-history protocol so that the new candidate isolates the
    pre-RoPE spatial merge change.
    """
    if original.shape[0] != NATIVE_NONCOND_MAPS + 1:
        raise ValueError("Expected six native age rows and one conditioning row")
    if not NATIVE_NONCOND_MAPS <= max_noncond_maps <= MAX_NONCOND_MAPS:
        raise ValueError("Unsupported non-conditioning history length")
    older_count = max_noncond_maps - NATIVE_NONCOND_MAPS
    extended = torch.cat(
        (
            original[:NATIVE_NONCOND_MAPS],
            original[NATIVE_NONCOND_MAPS - 1 : NATIVE_NONCOND_MAPS].expand(
                older_count, *original.shape[1:]
            ),
            original[NATIVE_NONCOND_MAPS : NATIVE_NONCOND_MAPS + 1],
        ),
        dim=0,
    )
    if extended.shape[0] != max_noncond_maps + 1:
        raise RuntimeError("Extended temporal position table has an invalid length")
    return extended


class PositionConsistentLongHistoryToMe(PositionConsistentRegionToMe):
    """Position-consistent generic ToMe under the fixed native spatial budget.

    It mass-averages memory content, keeps each cluster's positional embedding
    from its ToMe destination, then applies RoPE at that same representative
    spatial coordinate. ``matching_mode='key_value'`` also uses frozen value
    projections when forming ToMe's matching metric. The object-pointer tail is
    left untouched. This is a diagnostic candidate, not a validated improvement.
    """

    def __init__(
        self,
        model,
        *,
        window_size=4,
        max_noncond_maps=10,
        allocation="uniform",
        position_mode="representative",
        matching_mode="key",
        object_id_by_output_dict=None,
        passthrough=False,
        verify_no_merge_parity=False,
        provenance_target=None,
        hash_tensors_target=None,
    ):
        if model.num_maskmem != NATIVE_NONCOND_MAPS + 1:
            raise ValueError("Expected native SAM2 num_maskmem=7 before extending history")
        if not NATIVE_NONCOND_MAPS <= max_noncond_maps <= MAX_NONCOND_MAPS:
            raise ValueError("Unsupported non-conditioning history length")
        if allocation not in ("uniform", "recent_dense"):
            raise ValueError("Expected uniform or recent_dense allocation")
        if position_mode not in ("mean", "representative"):
            raise ValueError("Expected position_mode='mean' or 'representative'")
        super().__init__(
            model,
            mode="global",
            window_size=window_size,
            position_mode=position_mode,
            matching_mode=matching_mode,
            object_id_by_output_dict=object_id_by_output_dict,
            passthrough=passthrough,
            verify_no_merge_parity=verify_no_merge_parity,
            provenance_target=provenance_target,
            hash_tensors_target=hash_tensors_target,
        )
        self.max_noncond_maps = int(max_noncond_maps)
        self.allocation = allocation
        self._original_num_maskmem = None
        self._original_tpos_data = None

    def _plan_for_target(self):
        if not self.spans:
            raise RuntimeError("SAM2 has not selected memory spans for this target")
        return build_fixed_budget_merge_plan(
            self.spans,
            max_noncond_maps=self.max_noncond_maps,
            window_size=self.window_size,
            allocation=self.allocation,
        )

    def _install_extended_time_slots(self):
        if self._original_num_maskmem is not None:
            raise RuntimeError("Extended temporal slots are already installed")
        original = self.model.maskmem_tpos_enc.data
        extended = extend_temporal_position_table(original, self.max_noncond_maps)
        self._original_num_maskmem = self.model.num_maskmem
        self._original_tpos_data = original
        self.model.maskmem_tpos_enc.data = extended
        self.model.num_maskmem = self.max_noncond_maps + 1

    def _restore_native_time_slots(self):
        if self._original_num_maskmem is None:
            return
        self.model.maskmem_tpos_enc.data = self._original_tpos_data
        self.model.num_maskmem = self._original_num_maskmem
        self._original_num_maskmem = None
        self._original_tpos_data = None

    def __enter__(self):
        self._install_extended_time_slots()
        try:
            return super().__enter__()
        except BaseException:
            self._restore_native_time_slots()
            raise

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self._restore_native_time_slots()

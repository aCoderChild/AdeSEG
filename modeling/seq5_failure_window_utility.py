"""Frozen-plan helpers for post-hoc Seq5 memory-group utility analysis."""

from __future__ import annotations

import torch

from modeling.within_frame_token_utility import WithinFrameTokenRead


TARGETS = (157, 184)
TILE_NAMES = tuple(f"tile_{i:02d}" for i in range(16))


class FailureWindowTokenRead(WithinFrameTokenRead):
    """Apply the existing one-source tile ablation to a declared failure target."""

    def __init__(self, model):
        super().__init__(model)
        self.capture_raw_decoder = False
        self.raw_decoder_outputs = []
        self.decoder_handle = None

    def _capture_decoder(self, _module, _args, output):
        if not self.capture_raw_decoder:
            return
        if not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeError("Unexpected SAM2 mask-decoder output")
        masks, iou, _tokens, object_score = output
        if masks.ndim != 4 or masks.shape[0] != 1 or iou.shape[:2] != masks.shape[:2]:
            raise RuntimeError("Unexpected mask-decoder candidate tensor shapes")
        selected = int(torch.argmax(iou[0]).item()) if masks.shape[1] > 1 else 0
        self.raw_decoder_outputs.append({
            "lowres_logits": masks[0, selected].detach().float().cpu(),
            "object_score_logit": float(object_score.detach().float().reshape(-1)[0].cpu()),
            "selected_mask_index": selected,
            "candidate_count": int(masks.shape[1]),
        })

    def __enter__(self):
        super().__enter__()
        self.decoder_handle = self.model.sam_mask_decoder.register_forward_hook(self._capture_decoder)
        return self

    def __exit__(self, *exc):
        if self.decoder_handle is not None:
            self.decoder_handle.remove()
            self.decoder_handle = None
        return super().__exit__(*exc)


def target_sources(trace):
    """Return only actual past spatial sources, protecting prompt and latest maps."""
    spans = trace.get("memory_frames")
    target = int(trace.get("frame", -1))
    if not isinstance(spans, list) or len(spans) < 3:
        raise RuntimeError("Expected prompt, at least one older source, and latest spatial memory")
    ids = [int(span["frame"]) for span in spans]
    if len(ids) != len(set(ids)) or target < 0 or any(frame >= target for frame in ids):
        raise RuntimeError("Memory source IDs must be unique and strictly earlier than target")
    if int(spans[0]["t_pos"]) != 0 or ids[-1] != target - 1:
        raise RuntimeError("Prompt/latest-source layout differs from frozen native schedule")
    return ids[1:-1]


def summarize_target(target_record):
    """Summarize paired native-vs-tile effects without treating tiles as samples."""
    native = target_record["native"]
    result = {
        "target_frame": target_record["target_frame"],
        "native_visible_dice": native["visible"]["scores"]["dice"],
        "native_raw_dice": native["raw"]["scores"]["dice"],
        "native_object_score_logit": native["object_score_logit"],
        "sources": [],
    }
    for source in target_record["sources"]:
        rows = source["tiles"]
        if tuple(row["tile"] for row in rows) != TILE_NAMES:
            raise RuntimeError("Incomplete or reordered tile intervention set")
        visible = [row["visible"]["scores"]["dice"] for row in rows]
        raw = [row["raw"]["scores"]["dice"] for row in rows]
        visible_delta = [value - result["native_visible_dice"] for value in visible]
        result["sources"].append({
            "source_frame": source["source_frame"],
            "tile_count": len(rows),
            "mean_tile_removed_visible_dice": sum(visible) / len(visible),
            "mean_visible_dice_change_vs_native": sum(visible_delta) / len(visible_delta),
            "oracle_best_visible_dice_change_vs_native": max(visible_delta),
            "mean_raw_dice_change_vs_native": sum(raw) / len(raw) - result["native_raw_dice"],
            "presence_positive_tile_count": sum(row["object_score_logit"] > 0 for row in rows),
            "tile_deltas": visible_delta,
        })
    result["warning"] = "Outcome-selected tile effects are exploratory oracle diagnostics, not a deployable selector or efficacy estimate."
    return result


def validate_target_sources(trace, expected_sources):
    actual = target_sources(trace)
    if actual != list(expected_sources):
        raise RuntimeError(f"Native memory source order changed: {actual} != {list(expected_sources)}")
    return actual

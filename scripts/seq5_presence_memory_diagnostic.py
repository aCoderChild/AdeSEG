"""Read-only Seq5 probe for SAM2 pre-suppression masks and memory attention.

The probe replays native SAM2 and must be run only after the frozen cohort run
has reached a terminal state. Ground-truth masks choose a fixed set of query
locations for post-hoc attention summaries; they are never passed to SAM2.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.native_value_consolidation import NativeValueConsolidation
from scripts.memory_token_io import digest
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    PROTOCOL,
    SAM2_REFERENCE,
    load_shape,
    native_reference,
    run_condition,
    verify_native_setup,
)
from scripts.run_regional_refinement import write


SEQUENCE = "seq5"
# Sample both sides of the object-presence collapse/recovery. All are
# ground-truth-positive frames, so the observer can compare raw decoder masks
# with the masks SAM2 returns after its native object-presence decision.
TARGETS = (92, 93, 157, 158, 182, 184, 241, 242)
ACTIVE_RUN = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/tome_region_identification_v1/"
    "post_value_source_review_20261112/dependency_value_intervention_v1/full_attempt01"
)
CURRENT_FULL_RUN = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/pointer_conditioned_tome_polypgen_v1/"
    "full_attempt01"
)
DEFAULT_OUTPUT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/tome_region_identification_v1/"
    "post_value_source_review_20261112/dependency_value_intervention_v1/"
    "seq5_raw_mask_attention_probe01"
)


def _active_frozen_run():
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect active processes: {result.stderr.strip()}")
    active_roots = (str(ACTIVE_RUN), str(CURRENT_FULL_RUN))
    active_commands = [
        line.strip() for line in result.stdout.splitlines()
        if any(root in line for root in active_roots)
        and "seq5_presence_memory_diagnostic.py" not in line
    ]
    if active_commands:
        print("Active frozen run process(es):", *active_commands, sep="\n  ")
    return bool(active_commands)


def _mask_array(path):
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy((mask > 0).astype(np.uint8))


def _grid_mask(frame, size=(64, 64)):
    """Return post-hoc GT occupancy; never used by SAM2 or the memory reader."""
    mask = _mask_array(frame.mask_path).float()[None, None]
    return F.interpolate(mask, size=size, mode="nearest")[0, 0].bool().cpu()


def _summarize_spatial_attention(source_probs, source_gt):
    """Summarize attention over one square spatial source without changing it."""
    if source_probs.ndim != 4 or not torch.isfinite(source_probs).all():
        raise RuntimeError("Expected finite [batch, heads, queries, spatial_keys] probabilities")
    side = int(np.sqrt(source_probs.shape[-1]))
    if side * side != source_probs.shape[-1] or tuple(source_gt.shape) != (side, side):
        raise RuntimeError("Spatial attention and post-hoc GT grids are misaligned")
    token_mass = source_probs.mean(dim=(0, 1, 2)).detach().float().cpu()
    source_attention_mass = float(source_probs.sum(-1).mean().detach().cpu())
    if not torch.isclose(token_mass.sum(), torch.tensor(source_attention_mass), atol=1e-6, rtol=1e-5):
        raise RuntimeError("Per-token attention mass does not sum to its source mass")
    object_cells = source_gt.flatten().to(source_probs.device)
    object_area_fraction = float(object_cells.float().mean().cpu())
    object_attention_mass = float(
        source_probs[..., object_cells].sum(-1).mean().detach().cpu()
    ) if object_cells.any() else 0.0
    object_attention_fraction = (
        object_attention_mass / source_attention_mass
        if source_attention_mass > 0 else 0.0
    )
    object_attention_enrichment = (
        object_attention_fraction / object_area_fraction
        if object_area_fraction > 0 and source_attention_mass > 0 else None
    )
    top_mass, top_indices = torch.topk(
        token_mass, k=min(8, int(token_mass.numel())), largest=True, sorted=True
    )
    top_tokens = []
    for mass, flat_index in zip(top_mass.tolist(), top_indices.tolist(), strict=True):
        y, x = divmod(int(flat_index), side)
        top_tokens.append({
            "flat_index": int(flat_index),
            "y": y,
            "x": x,
            "mean_attention_mass": float(mass),
            "posthoc_gt_object": bool(source_gt[y, x]),
        })
    normalized = token_mass / token_mass.sum().clamp_min(torch.finfo(token_mass.dtype).tiny)
    entropy = float(-(
        normalized * normalized.clamp_min(torch.finfo(normalized.dtype).tiny).log()
    ).sum())
    return {
        "attention_mass": source_attention_mass,
        "posthoc_gt_object_area_fraction": object_area_fraction,
        "posthoc_gt_object_attention_mass": object_attention_mass,
        "posthoc_gt_object_fraction_of_source_attention": object_attention_fraction,
        "posthoc_gt_object_attention_enrichment_vs_area": object_attention_enrichment,
        "spatial_attention_entropy_nats": entropy,
        "top_attention_tokens": top_tokens,
    }


def _evenly_spaced(indices, count=32):
    if not indices.numel():
        return indices
    positions = torch.linspace(0, indices.numel() - 1, min(count, indices.numel()))
    return indices[positions.round().long()].unique(sorted=True)


def _query_groups(frame):
    grid = _grid_mask(frame).flatten()
    return {
        "gt_object": _evenly_spaced(torch.where(grid)[0]),
        "gt_background": _evenly_spaced(torch.where(~grid)[0]),
    }


class Seq5DiagnosticReader(NativeValueConsolidation):
    """Native no-op reader with read-only decoder and SDPA observers."""

    def __init__(self, model, query_groups, source_gt_masks):
        super().__init__(model, mode="noop")
        self.query_groups = query_groups
        # This metadata labels saved attention summaries only. It is not passed
        # to the model and cannot affect prediction or memory construction.
        self.source_gt_masks = source_gt_masks
        self.raw_decoder = {}
        self.post_presence = {}
        self.attention = {}
        self.decoder_handle = None
        self.original_track_step = None

    def _capture_decoder(self, module, args, output):
        frame = self.frame
        if frame not in TARGETS:
            return
        if frame in self.raw_decoder:
            raise RuntimeError(f"Duplicate mask-decoder capture at frame {frame}")
        masks, iou, tokens, object_score = output
        if not isinstance(masks, torch.Tensor) or masks.ndim != 4:
            raise RuntimeError(f"Unexpected raw decoder mask shape at frame {frame}")
        if not isinstance(object_score, torch.Tensor):
            raise RuntimeError("Expected the native decoder object-presence output")
        if masks.shape[0] != 1 or masks.shape[1] < 1:
            raise RuntimeError(f"Unexpected decoder batch/mask count at frame {frame}")
        if masks.shape[1] > 1:
            selected = int(torch.argmax(iou[0]).item())
            raw = masks[0, selected].detach().float().cpu()
        else:
            selected = 0
            raw = masks[0, 0].detach().float().cpu()
        self.raw_decoder[frame] = {
            "raw_lowres_logits": raw,
            "all_lowres_logits": masks.detach().float().cpu(),
            "iou_predictions": iou.detach().float().cpu(),
            "selected_mask_index": selected,
            "object_score": float(object_score.detach().float().cpu().reshape(-1)[0]),
        }

    def _track_step_observed(self, *args, **kwargs):
        output = self.original_track_step(*args, **kwargs)
        # SAM2Base.track_step returns its current-frame output dictionary directly.
        # The predictor later wraps that dictionary with resized masks, but this
        # hook runs inside track_step and must not assume that outer tuple shape.
        if not isinstance(output, dict):
            raise RuntimeError(
                f"Expected SAM2 track_step to return a dict, got {type(output).__name__}"
            )
        current_out = output
        frame = int(kwargs["frame_idx"] if "frame_idx" in kwargs else args[0])
        if frame in TARGETS:
            if frame in self.post_presence:
                raise RuntimeError(f"Duplicate post-presence capture at frame {frame}")
            self.post_presence[frame] = {
                "lowres_logits": current_out["pred_masks"].detach().float().cpu(),
                "object_score": float(
                    current_out["object_score_logits"].detach().float().cpu().reshape(-1)[0]
                ),
            }
        return output

    def _source_attention(self, query, key, value, sdpa_output, layer, pointers):
        groups = self.query_groups[self.frame]
        target = self.attention.setdefault(self.frame, {})
        if layer in target:
            raise RuntimeError(f"Duplicate attention capture at frame {self.frame}, layer {layer}")
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise RuntimeError("Unexpected projected SAM2 attention tensor layout")
        scale = 1.0 / (query.shape[-1] ** 0.5)
        rows = {}
        for group_name, indices in groups.items():
            if not indices.numel():
                raise RuntimeError(f"No {group_name} query positions at frame {self.frame}")
            q = query.index_select(-2, indices.to(query.device)).float()
            logits = torch.matmul(q, key.float().transpose(-2, -1)) * scale
            probs = torch.softmax(logits, dim=-1)
            actual = sdpa_output.index_select(-2, indices.to(sdpa_output.device)).float()
            source_rows = []
            for span in self.spans:
                start, stop = int(span["start"]), int(span["stop"])
                source_probs = probs[..., start:stop]
                source_values = value[..., start:stop, :].float()
                context = torch.matmul(source_probs, source_values)
                source_alignment = F.cosine_similarity(context, actual, dim=-1).mean()
                source_gt = self.source_gt_masks.get(int(span["frame"]))
                side = int(np.sqrt(stop - start))
                if side * side != stop - start or source_gt is None or tuple(source_gt.shape) != (side, side):
                    raise RuntimeError(
                        f"Missing aligned post-hoc GT grid for source frame {span['frame']}"
                    )
                spatial_summary = _summarize_spatial_attention(source_probs, source_gt)
                source_rows.append({
                    "source_frame": int(span["frame"]),
                    "t_pos": int(span["t_pos"]),
                    "keys": stop - start,
                    **spatial_summary,
                    "weighted_value_l2": float(context.norm(dim=-1).mean().cpu()),
                    "cosine_to_total_attention_output": float(source_alignment.cpu()),
                })
            pointer_context_l2 = 0.0
            pointer_alignment = 0.0
            if pointers:
                pointer_probs = probs[..., -pointers:]
                pointer_values = value[..., -pointers:, :].float()
                pointer_context = torch.matmul(pointer_probs, pointer_values)
                pointer_context_l2 = float(pointer_context.norm(dim=-1).mean().cpu())
                pointer_alignment = float(
                    F.cosine_similarity(pointer_context, actual, dim=-1).mean().cpu()
                )
            pointer_mass = (
                float(probs[..., -pointers:].sum(-1).mean().cpu()) if pointers else 0.0
            )
            rows[group_name] = {
                "query_count": int(indices.numel()),
                "memory_sources": source_rows,
                "pointer_attention_mass": pointer_mass,
                "pointer_weighted_value_l2": pointer_context_l2,
                "pointer_cosine_to_total_attention_output": pointer_alignment,
                "attended_output_l2": float(actual.norm(dim=-1).mean().cpu()),
            }
        target[layer] = rows

    def _sdpa(self, query, key, value, *args, **kwargs):
        output = super()._sdpa(query, key, value, *args, **kwargs)
        if self.frame in TARGETS and self.active is not None:
            layer, pointers = self.active
            self._source_attention(query, key, value, output, layer, pointers)
        return output

    def __enter__(self):
        super().__enter__()
        self.original_track_step = self.model.track_step
        self.model.track_step = self._track_step_observed
        self.decoder_handle = self.model.sam_mask_decoder.register_forward_hook(
            self._capture_decoder
        )
        return self

    def __exit__(self, *exc):
        if self.decoder_handle is not None:
            self.decoder_handle.remove()
            self.decoder_handle = None
        if self.original_track_step is not None:
            self.model.track_step = self.original_track_step
            self.original_track_step = None
        return super().__exit__(*exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument(
        "--after-full-run-terminal", action="store_true",
        help="Required acknowledgement that the frozen full-cohort process is terminal.",
    )
    args = parser.parse_args()
    if not args.after_full_run_terminal:
        raise RuntimeError("Wait for the frozen full-cohort run to finish before this replay")
    if _active_frozen_run():
        raise RuntimeError("The frozen full-cohort run is still active; diagnostic not started")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    manifest = ROOT / "data/polypgen_sequence.jsonl"
    checkpoint_sha = digest(args.checkpoint)
    protocol = json.loads(PROTOCOL.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    if checkpoint_sha != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the frozen protocol")
    native_setup = verify_native_setup(args.checkpoint, manifest)
    if not NATIVE_RESULTS.is_file() or not NATIVE_SETUP.is_file():
        raise RuntimeError("Archived native result/receipt is unavailable")
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    frames = videos[SEQUENCE]
    native_results = json.loads(NATIVE_RESULTS.read_text())
    prompt, native_rows, receipt = native_reference(
        NATIVE_RESULTS.parent, native_results, native_protocol, SEQUENCE, frames
    )
    if not set(TARGETS).issubset(native_rows):
        raise RuntimeError("One or more requested target frames is absent from native reference")
    query_groups = {frame: _query_groups(frames[frame]) for frame in TARGETS}
    source_gt_masks = {index: _grid_mask(frame) for index, frame in enumerate(frames)}
    captures = []

    def make_reader(model):
        reader = Seq5DiagnosticReader(model, query_groups, source_gt_masks)
        captures.append(reader)
        return reader

    setup = {
        "status": "running",
        "sequence": SEQUENCE,
        "targets": list(TARGETS),
        "prompt": prompt,
        "native_exact_required": True,
        "device": "cpu",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": digest(manifest),
        "native_setup_sha256": digest(NATIVE_SETUP),
        "native_results_sha256": digest(NATIVE_RESULTS),
        "protocol_sha256": digest(PROTOCOL),
        "sam2_reference": str(SAM2_REFERENCE),
        "sam2_revision": native_setup.get("sam2_revision"),
        "native_receipt": receipt,
        "attention_queries": "deterministically sampled from GT object/background grid positions for post-hoc measurement only; GT is never passed to SAM2",
        "measurements": ["raw mask-decoder logits before object-presence suppression", "attention mass and weighted-value norm by native memory source", "native exact parity at every frame"],
    }
    write(args.output_dir / "setup.json", setup)

    rows, runtime = run_condition(
        condition="native",
        sequence=SEQUENCE,
        frames=frames,
        prompt=prompt,
        native_rows=native_rows,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        max_noncond_maps=6,
        condition_name="native_noop_probe",
        memory_reader_factory=make_reader,
        native_exact_required=True,
    )
    if len(captures) != 1:
        raise RuntimeError("Expected exactly one instrumentation reader")
    reader = captures[0]
    if (
        set(reader.raw_decoder) != set(TARGETS)
        or set(reader.post_presence) != set(TARGETS)
        or set(reader.attention) != set(TARGETS)
    ):
        raise RuntimeError("Not all target raw-mask/attention captures were recorded")
    if any(set(reader.attention[frame]) != {0, 1, 2, 3} for frame in TARGETS):
        raise RuntimeError("Expected all four attention layers at all target frames")

    native_by_frame = {row["frame"]: row for row in rows}
    report = {"native_exact_all_frames": True, "runtime": runtime, "targets": {}}
    zero_presence = [
        int(row["frame"]) for row in rows
        if row.get("present") and float(row.get("object_score", 0.0)) <= 0.0
    ]
    zero_presence_intervals = []
    for frame in zero_presence:
        if not zero_presence_intervals or frame > zero_presence_intervals[-1][-1] + 1:
            zero_presence_intervals.append([frame])
        else:
            zero_presence_intervals[-1].append(frame)
    report["gt_present_object_score_nonpositive"] = {
        "frame_count": len(zero_presence),
        "intervals_inclusive": [
            {"start": interval[0], "end": interval[-1], "length": len(interval)}
            for interval in zero_presence_intervals
        ],
        "interpretation": (
            "Diagnostic count only: SAM2's native presence decision suppresses these masks; "
            "this does not identify the causal memory token or establish a replacement method."
        ),
    }
    raw_tensors = {}
    for frame in TARGETS:
        raw = reader.raw_decoder[frame]
        post_presence = reader.post_presence[frame]
        stored = raw["raw_lowres_logits"]
        score = raw["object_score"]
        if post_presence["object_score"] != score:
            raise RuntimeError(f"Decoder and track-step object scores differ at frame {frame}")
        if score > 0:
            if not torch.equal(stored, post_presence["lowres_logits"][0, 0]):
                raise RuntimeError(f"Present-frame raw mask changed before memory storage at {frame}")
        elif not torch.all(post_presence["lowres_logits"] == -1024.0):
            raise RuntimeError(f"Absent-frame native mask was not suppressed at {frame}")
        target = _mask_array(frames[frame].mask_path).float()[None, None]
        gt_lowres = F.interpolate(target, size=stored.shape[-2:], mode="nearest")[0, 0].bool()
        raw_binary = stored > 0
        intersection = int((raw_binary & gt_lowres).sum())
        denominator = int(raw_binary.sum() + gt_lowres.sum())
        raw_dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
        # run_condition has reset the inference state; native row and decoder hook are
        # sufficient for the pre-/post-suppression parity checks below.
        native = native_by_frame[frame]
        report["targets"][str(frame)] = {
            "native_dice_after_presence_logic": native["scores"]["dice"],
            "native_object_score": native["object_score"],
            "raw_object_score": score,
            "raw_decoder_mask_shape": list(stored.shape),
            "raw_decoder_mask_sha256": hashlib.sha256(
                stored.contiguous().numpy().tobytes()
            ).hexdigest(),
            "raw_lowres_dice_diagnostic": raw_dice,
            "raw_positive_pixels": int(raw_binary.sum()),
            "ground_truth_lowres_pixels": int(gt_lowres.sum()),
            "native_presence_suppressed": bool(score <= 0),
            "raw_equals_post_presence_mask": bool(
                torch.equal(stored, post_presence["lowres_logits"][0, 0])
            ),
            "attention_by_layer": reader.attention[frame],
        }
        raw_tensors[str(frame)] = raw

    torch.save(raw_tensors, args.output_dir / "raw_decoder_captures.pt")
    write(args.output_dir / "native_frames.json", rows)
    report["status"] = "complete"
    report["all_native_frames_exact"] = runtime["native_exact_frames_checked"] == len(rows)
    if not report["all_native_frames_exact"]:
        raise RuntimeError("Native replay did not compare every saved target frame")
    write(args.output_dir / "diagnostic.json", report)
    write(args.output_dir / "completion.json", {
        "status": "complete",
        "native_exact_frames": runtime["native_exact_frames_checked"],
        "diagnostic_targets": list(TARGETS),
        "sam2_parameters_unchanged": runtime["parameter_hash_unchanged"],
        "temporal_slots_restored": runtime["temporal_slots_restored"],
        "source_parameters": "frozen; inference-only observer",
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

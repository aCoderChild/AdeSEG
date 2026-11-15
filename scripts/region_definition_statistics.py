"""Descriptive audit statistics, not a replacement region-selection algorithm."""

import numpy as np

from modeling.region_tokens import _author


def contour_coverage(binary, labels):
    """One-pixel author Boundary IoU contour on native256; NN token footprints.

    Keep author border convention, but also report without the outermost raster
    row/column. Empty denominators are undefined, never a perfect score.
    """
    binary = np.asarray(binary, dtype=np.uint8)
    labels = np.asarray(labels)
    if binary.shape != (256, 256) or labels.shape != (64, 64):
        raise ValueError("Expected native256 mask and memory64 labels")
    contour = _author()["boundary"](binary, 1 / np.hypot(*binary.shape)).astype(bool)
    covered = np.repeat(np.repeat(labels == 2, 4, axis=0), 4, axis=1)
    interior = np.ones_like(contour)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    result = {}
    for name, eligible in (
        ("including_border", contour),
        ("excluding_border", contour & interior),
    ):
        total = int(eligible.sum())
        missed = int((eligible & ~covered).sum())
        footprint = covered if name == "including_border" else covered & interior
        footprint_pixels = int(footprint.sum())
        contour_covered = total - missed
        result[name] = {
            "contour_pixels": total,
            "missed_pixels": missed,
            "missed_fraction": missed / total if total else None,
            "boundary_footprint_pixels": footprint_pixels,
            "contour_covered_pixels": contour_covered,
            "boundary_footprint_precision": (
                contour_covered / footprint_pixels if footprint_pixels else None
            ),
        }
    result["image_border_contour_pixels"] = int((contour & ~interior).sum())
    return result


def size_bin(area, cutpoints):
    """Frozen positive-frame quartiles; ties go to the lower bin."""
    return (
        "absent"
        if area == 0
        else f"Q{int(np.searchsorted(cutpoints, area, side='left')) + 1}"
    )


def summarize_frames(records):
    """Pooled descriptive fractions; retain numerators, denominators and N/A."""
    result = {}
    for name in records[0]["schemes"] if records else []:
        selected = [r["schemes"][name] for r in records]
        counts = np.array([r["counts"] for r in selected])
        item = {
            "frames": len(selected),
            "mean_token_counts_bg_fg_boundary": counts.mean(axis=0).tolist(),
            "empty_interior_frames": int((counts[:, 1] == 0).sum()),
            "empty_boundary_frames": int((counts[:, 2] == 0).sum()),
            "pooled_gt_confusion": np.sum(
                [r["confusion"] for r in selected], axis=0
            ).tolist(),
        }
        for reference in ("self_contour", "gt_contour"):
            item[reference] = {}
            for mode in ("including_border", "excluding_border"):
                rows = [r[reference][mode] for r in selected]
                total = sum(r["contour_pixels"] for r in rows)
                missed = sum(r["missed_pixels"] for r in rows)
                covered = total - missed
                footprint_values = []
                for record, contour_record in zip(selected, rows):
                    footprint = contour_record.get("boundary_footprint_pixels")
                    if footprint is None and mode == "including_border":
                        # Existing saved audit rows predate this statistic. Each
                        # 64x64 label covers a disjoint 4x4 block in native256.
                        footprint = int(record["counts"][2]) * 16
                    if footprint is not None:
                        footprint_values.append(int(footprint))
                footprint_total = sum(footprint_values)
                fractions = [
                    r["missed_fraction"]
                    for r in rows
                    if r["missed_fraction"] is not None
                ]
                item[reference][mode] = {
                    "contour_pixels": total,
                    "missed_pixels": missed,
                    "pooled_missed_fraction": missed / total if total else None,
                    "mean_frame_missed_fraction": float(np.mean(fractions))
                    if fractions
                    else None,
                    "defined_frames": len(fractions),
                    "boundary_footprint_pixels": (
                        footprint_total if len(footprint_values) == len(rows) else None
                    ),
                    "contour_covered_pixels": covered,
                    "pooled_boundary_footprint_precision": (
                        covered / footprint_total
                        if footprint_total and len(footprint_values) == len(rows)
                        else None
                    ),
                    "precision_defined_frames": len(footprint_values),
                }
        result[name] = item
    return result

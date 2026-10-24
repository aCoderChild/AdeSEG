"""Label-defined false alarms and sampled-frame re-detection; no tuned threshold."""

import numpy as np


def propagation_summary(rows):
    """A re-detection is any predicted/GT overlap, not an arbitrary Dice cutoff.

    Returns stop at the next absent frame or sequence end. Undetected returns
    are explicitly censored, never silently omitted from the event denominator.
    """
    if not rows or any(b["frame"] != a["frame"] + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("Expected nonempty consecutive sampled target indices")
    absent = [r for r in rows if not r["present"]]
    present = [r for r in rows if r["present"]]
    events = []
    for index in range(1, len(rows)):
        if rows[index - 1]["present"] or not rows[index]["present"]:
            continue
        end = index
        while end < len(rows) and rows[end]["present"]:
            end += 1
        detected = next(
            (j for j in range(index, end) if rows[j]["scores"]["iou"] > 0), None
        )
        events.append({
            "return_frame": rows[index]["frame"],
            "detected": detected is not None,
            "delay_sampled_frames": None if detected is None else detected - index,
            "followup_sampled_frames": end - index,
            "censored": detected is None,
        })
    return {
        "targets": len(rows),
        "present_targets": len(present),
        "gt_empty_targets": len(absent),
        "gt_empty_false_alarm_count": sum(bool(r["absent_fp"]) for r in absent),
        "gt_empty_false_alarm_rate": (
            float(np.mean([bool(r["absent_fp"]) for r in absent])) if absent else None
        ),
        "present_dice": (
            float(np.mean([r["scores"]["dice"] for r in present])) if present else None
        ),
        "returns": events,
        "return_count": len(events),
        "undetected_returns": sum(e["censored"] for e in events),
        "units": "sampled manifest frames, not seconds or continuous source frames",
    }


def paired_method_summary(methods):
    """Fixed two contrasts × three endpoints, reusing sequence statistics."""
    from evaluation.statistics import (
        holm_adjust, paired_wilcoxon, sequence_bootstrap_interval,
    )

    endpoints = ("gt_empty_false_alarm_rate", "present_dice", "return_failure_fraction")
    values = {}
    for method, sequences in methods.items():
        values[method] = {key: {} for key in endpoints}
        for sequence, row in sequences.items():
            for key in endpoints[:2]:
                if row[key] is not None:
                    values[method][key][sequence] = row[key]
            if row["return_count"]:
                values[method][endpoints[2]][sequence] = (
                    row["undetected_returns"] / row["return_count"]
                )
    comparisons = []
    for newer, native in (("samurai", "sam2"), ("samurai_fg_gate", "samurai")):
        for key in endpoints:
            left, right = values[newer][key], values[native][key]
            names = sorted(set(left) & set(right))
            delta = {name: left[name] - right[name] for name in names}
            comparisons.append({
                "contrast": f"{newer} minus {native}", "endpoint": key,
                "n_sequences": len(names),
                "mean_difference": float(np.mean(list(delta.values()))) if delta else None,
                "ci95": sequence_bootstrap_interval(delta, seed=20261024) if delta else None,
                "p": paired_wilcoxon(left, right)["p"],
            })
    adjusted = holm_adjust({str(i): row["p"] for i, row in enumerate(comparisons)})
    for i, row in enumerate(comparisons):
        row["holm6_p"] = adjusted[str(i)]
    return {
        "comparisons": comparisons,
        "sequence_endpoints": values,
        "limits": "Exploratory; return delay is descriptive and censored returns remain explicit",
    }

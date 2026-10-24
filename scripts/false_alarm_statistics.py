"""Label-defined false alarms and sampled-frame re-detection; no tuned threshold."""

import numpy as np


def prediction_nonempty(row):
    """Recover mask occupancy from saved masks' exact binary metric convention.

    On a positive GT frame with zero overlap, precision=1 means no predicted
    pixels (0/0 convention); precision=0 means a nonempty disjoint prediction.
    This is NOT recovery of the unsaved native object score.
    """
    if "prediction_nonempty" in row:
        if not isinstance(row["prediction_nonempty"], bool):
            raise ValueError("Expected boolean mask occupancy")
        return row["prediction_nonempty"]
    if not row["present"]:
        if not isinstance(row["absent_fp"], bool):
            raise ValueError("Expected saved absent-frame mask occupancy")
        return row["absent_fp"]
    if row["scores"]["dice"] > 0:
        return True
    precision = row["scores"]["precision"]
    if precision not in (0, 1):
        raise ValueError("Unexpected zero-overlap precision convention")
    return precision == 0


def native_opportunities(rows):
    """Baseline empty-mask opportunities, not an upper bound on rollout effects."""
    if not rows or any(b["frame"] != a["frame"] + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("Expected complete consecutive sampled target records")
    counts = dict(targets=len(rows), gt_empty=0, gt_empty_after_empty_prediction=0,
                  after_correct_empty=0, after_missed_present=0,
                  false_alarms_after_empty_prediction=0,
                  false_alarms_after_nonempty_prediction=0,
                  present_after_empty_prediction=0,
                  consecutive_gt_empty_false_alarms=0)
    previous_empty, previous_present = False, True  # The single foreground GT prompt.
    for row in rows:
        nonempty = prediction_nonempty(row)
        if not row["present"]:
            counts["gt_empty"] += 1
            if previous_empty:
                counts["gt_empty_after_empty_prediction"] += 1
                counts["after_correct_empty" if not previous_present else "after_missed_present"] += 1
                counts["false_alarms_after_empty_prediction"] += nonempty
            elif nonempty:
                counts["false_alarms_after_nonempty_prediction"] += 1
                counts["consecutive_gt_empty_false_alarms"] += not previous_present
        elif previous_empty:
            counts["present_after_empty_prediction"] += 1
        previous_empty, previous_present = not nonempty, row["present"]
    return counts


def relapse_risk_summary(rows, fixed_source_indices=None):
    """Score-defined own risk is descriptive; fixed baseline risk supports pairing."""
    if fixed_source_indices is None:
        if not all("previous_object_score" in r for r in rows):
            return {"relapse_risk_targets": None, "relapse_false_alarms": None,
                    "relapse_false_alarm_rate": None}
        fixed_source_indices = {
            r.get("source_frame_index", r["frame"]) for r in rows
            if not r["present"] and r["previous_object_score"] <= 0
        }
    selected = [r for r in rows if r.get("source_frame_index", r["frame"]) in fixed_source_indices]
    if any(r["present"] for r in selected) or len(selected) != len(fixed_source_indices):
        raise ValueError("Fixed baseline relapse-risk set does not match target coverage/GT")
    false_alarms = sum(bool(r["absent_fp"]) for r in selected)
    return {"relapse_risk_targets": len(selected), "relapse_false_alarms": false_alarms,
            "relapse_false_alarm_rate": false_alarms / len(selected) if selected else None}


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
        **relapse_risk_summary(rows),
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

    endpoints = ("relapse_false_alarm_rate", "present_dice", "return_failure_fraction")
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

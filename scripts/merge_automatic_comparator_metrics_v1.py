"""Combine independently scored automatic-prompt cohorts without opening labels.

This report is exploratory. In particular, project Lean is a modified runtime,
SAM2Long is offline, and missing model runs are not interpreted as zero Dice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METHODS = ("native", "lean", "samurai", "sam2long", "refiner")
AUTHOR_LEAN = "lean_author_mps"
CMSA = "cmsa_net"
PRIORITY = {"seq11", "seq20", "seq23"}
NO_INITIALIZATION = {"seq1", "seq7"}
ADDITIONAL = {f"seq{i}" for i in range(1, 24)} - PRIORITY - NO_INITIALIZATION
ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    before = sha(path)
    contents = path.read_bytes()
    require(before == hashlib.sha256(contents).hexdigest() == sha(path), f"Input changed during read: {path}")
    value = json.loads(contents)
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def verify_scored(score_path: Path, audit_path: Path, hash_key: str,
                  scorer: Path, auditor: Path, audit_source_key: str) -> dict:
    score, audit = read(score_path), read(audit_path)
    require(score.get("passed") is True and audit.get("passed") is True,
            f"Score or independent audit did not pass: {score_path}")
    require(audit.get(hash_key) == sha(score_path), f"Audit is not bound to {score_path}")
    require(score.get("scorer_sha256") == sha(scorer) and audit.get(audit_source_key) == sha(auditor),
            "Score or audit implementation differs from the reviewed source")
    require(audit.get("error_count", 0) == 0 and not audit.get("errors"),
            f"Audit reports discrepancies: {audit_path}")
    if "report_sha256" in score:
        require(score["report_sha256"] == sha(score_path.parent / "report.md"),
                f"Report differs from scoring receipt: {score_path}")
    if "report_sha256" in audit:
        require(audit["report_sha256"] == sha(score_path.parent / "report.md"),
                f"Report differs from independent audit: {score_path}")
    return score


def check_same_frames(left: list[dict], right: list[dict]) -> None:
    require(len(left) == len(right), "Frame coverage differs")
    for a, b in zip(left, right, strict=True):
        for key in ("frame", "source_frame_index", "post_prompt", "initialization_frame"):
            require(a[key] == b[key], f"Frame identity differs: {key}")
        left_presence = {v["present"] for v in a.values() if isinstance(v, dict) and "present" in v}
        right_presence = {v["present"] for v in b.values() if isinstance(v, dict) and "present" in v}
        require(len(left_presence) == len(right_presence) == 1 and left_presence == right_presence,
                "Label presence differs across recorded rows")


def summarize(rows: list[dict], arm: str) -> dict:
    selected = [r for r in rows if r["post_prompt"]]
    present, empty = [], []
    for row in selected:
        value = row[arm]
        require(isinstance(value["present"], bool), "Presence must be boolean")
        if value["present"]:
            dice = value["dice"]
            require(isinstance(dice, (int, float)) and math.isfinite(dice) and 0 <= dice <= 1,
                    "Invalid present-frame Dice")
            present.append(float(dice))
        else:
            require(isinstance(value["false_alarm"], bool), "False alarm must be boolean")
            empty.append(value["false_alarm"])
    return {"present_frames": len(present),
            "mean_present_dice": sum(present) / len(present) if present else None,
            "empty_frames": len(empty), "false_alarms": sum(empty)}


def check_summary(actual: dict, claimed: dict) -> None:
    for key in ("present_frames", "empty_frames", "false_alarms"):
        require(actual[key] == claimed[key], f"Summary mismatch: {key}")
    a, b = actual["mean_present_dice"], claimed["mean_present_dice"]
    require((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9),
            "Summary mismatch: mean Dice")


def merge(additional: dict, prior: dict, factorial: dict, no_init: dict) -> dict:
    require(set(additional["rows"]) == ADDITIONAL and set(prior["rows"]) == PRIORITY
            and set(factorial["rows"]) == PRIORITY and set(no_init["summaries"]) == NO_INITIALIZATION,
            "Sequence coverage differs from the declared 18 + 3 + 2 cohort")
    table = {}
    for seq in ADDITIONAL:
        rows = additional["rows"][seq]
        table[seq] = {}
        for destination, source in (("native", "native"), ("samurai", "samurai"),
                                    ("sam2long", "sam2long"), ("lean", "A"), ("refiner", "D")):
            if source not in additional["summaries"][seq]:
                require(source in ("A", "D"), f"Missing required comparator {seq} {source}")
                table[seq][destination] = None
                continue
            result = summarize(rows, source)
            check_summary(result, additional["summaries"][seq][source])
            table[seq][destination] = result
        for row in rows:
            presence = {v["present"] for v in row.values() if isinstance(v, dict) and "present" in v}
            require(len(presence) == 1, f"Method label presence differs: {seq}")
    for seq in PRIORITY:
        prior_rows, fact_rows = prior["rows"][seq], factorial["rows"][seq]
        check_same_frames(prior_rows, fact_rows)
        table[seq] = {}
        for name in ("native", "lean", "samurai", "sam2long"):
            result = summarize(prior_rows, name)
            check_summary(result, prior["summaries"][seq][name])
            table[seq][name] = result
        check_summary(summarize(fact_rows, "A"), factorial["summaries"][seq]["A"]["post_prompt"])
        require(summarize(fact_rows, "A") == table[seq]["lean"], f"Prior Lean mismatch: {seq}")
        result = summarize(fact_rows, "D")
        check_summary(result, factorial["summaries"][seq]["D"]["post_prompt"])
        table[seq]["refiner"] = result
    for seq in NO_INITIALIZATION:
        item = no_init["summaries"][seq]
        require(item["initialization"] == "no_detection" and item["post_prompt_frame_count"] == 0
                and item["post_prompt_present_dice"] is None, f"No-initialization record differs: {seq}")
        table[seq] = {name: None for name in METHODS}
    return table


def aggregate(table: dict, methods: tuple[str, ...]) -> dict:
    eligible = [s for s, values in table.items() if all(values[m] is not None
                and values[m]["present_frames"] > 0 for m in methods)]
    metrics = {}
    for m in methods:
        present = sum(table[s][m]["present_frames"] for s in eligible)
        empty = sum(table[s][m]["empty_frames"] for s in eligible)
        metrics[m] = {"patient_macro_dice": (sum(table[s][m]["mean_present_dice"] for s in eligible) / len(eligible)
                                              if eligible else None),
                      "frame_weighted_dice": (sum(table[s][m]["mean_present_dice"] * table[s][m]["present_frames"]
                                                   for s in eligible) / present if present else None),
                      "present_frames": present, "empty_frames": empty,
                      "false_alarms": sum(table[s][m]["false_alarms"] for s in eligible)}
    return {"sequences": sorted(eligible, key=lambda s: int(s[3:])),
            "cohort_rule": "All requested methods available and each has at least one labeled-present post-prompt frame; empty-alarm totals use this same cohort.",
            "metrics": metrics}


def aggregate_empty(table: dict, methods: tuple[str, ...]) -> dict:
    eligible = [s for s, values in table.items() if all(values[m] is not None
                and values[m]["empty_frames"] > 0 for m in methods)]
    metrics = {}
    for method in methods:
        false_alarms = sum(table[s][method]["false_alarms"] for s in eligible)
        empty_frames = sum(table[s][method]["empty_frames"] for s in eligible)
        metrics[method] = {"false_alarms": false_alarms, "empty_frames": empty_frames,
                           "patient_macro_false_alarm_rate":
                           (sum(table[s][method]["false_alarms"] / table[s][method]["empty_frames"]
                                for s in eligible) / len(eligible) if eligible else None)}
    return {"sequences": sorted(eligible, key=lambda s: int(s[3:])),
            "cohort_rule": "All requested methods available and each has labeled-empty post-prompt frames; present frames are not required.",
            "metrics": metrics}


def paired_comparison(table: dict, reference: str, candidate: str) -> dict:
    cohort = aggregate(table, (reference, candidate))["sequences"]
    deltas = {seq: table[seq][candidate]["mean_present_dice"] - table[seq][reference]["mean_present_dice"]
              for seq in cohort}
    return {"reference": reference, "candidate": candidate, "sequences": cohort,
            "patient_macro_dice_difference": sum(deltas.values()) / len(deltas) if deltas else None,
            "improved": [seq for seq, delta in deltas.items() if delta > 1e-12],
            "worsened": [seq for seq, delta in deltas.items() if delta < -1e-12],
            "unchanged": [seq for seq, delta in deltas.items() if abs(delta) <= 1e-12],
            "per_sequence_difference": deltas}


def add_author_lean(table: dict, additional: dict, prior: dict, author: dict) -> None:
    expected = ADDITIONAL | PRIORITY
    require(set(author["rows"]) == expected and set(author["summaries"]) == expected,
            "Author Lean sequence coverage differs")
    for seq in expected:
        rows = author["rows"][seq]
        reference = additional["rows"][seq] if seq in ADDITIONAL else prior["rows"][seq]
        check_same_frames(reference, rows)
        result = summarize(rows, AUTHOR_LEAN)
        check_summary(result, author["summaries"][seq])
        table[seq][AUTHOR_LEAN] = result
    for seq in NO_INITIALIZATION:
        table[seq][AUTHOR_LEAN] = None


def check_additional_input_match(author: dict, additional: dict) -> None:
    for key in ("images", "prompt_frame", "prompt"):
        require(author[key] == additional[key], f"Author Lean input mismatch: {key}")


def check_priority_input_match(author: dict, completion: dict) -> None:
    require(author["images"] == [row["sha256"] for row in completion["input_images"]],
            "Priority author Lean image bytes differ")
    require(author["prompt_frame"] == completion["prompt_frame"]
            and author["prompt"] == completion["prompt_provenance"],
            "Priority author Lean prompt or viewport differs")


def verify_author_input_bindings(a, additional: dict, factorial: dict, author: dict) -> dict:
    add_path = a.additional_score.parent.parent / "verification/verification.json"
    author_path = a.author_lean_score.parent.parent / "verification/verification.json"
    extra = {str(p): sha(p) for p in (add_path, author_path)}
    require(extra[str(add_path)] == additional["verification_sha256"]
            and extra[str(author_path)] == author["verification_sha256"], "Verification receipt hash differs")
    add_proof, author_proof = read(add_path), read(author_path)
    require(add_proof.get("passed") is True and author_proof.get("passed") is True
            and add_proof.get("label_guard", {}).get("open_attempts") == 0
            and author_proof.get("label_guard", {}).get("open_attempts") == 0
            and add_proof.get("scorer_sha256") == additional["scorer_sha256"]
            and author_proof.get("scorer_sha256") == author["scorer_sha256"],
            "Comparator verification receipt identity differs")
    for seq in ADDITIONAL:
        check_additional_input_match(author_proof["records"][seq], add_proof["inputs"][seq])
    run_map = a.priority_factorial_score.parent.parent / "run_map.json"
    extra[str(run_map)] = sha(run_map)
    require(extra[str(run_map)] == factorial["run_map_sha256"], "Priority run map differs")
    mapping = read(run_map)
    for seq in PRIORITY:
        completion = Path(mapping[seq]["A"]).resolve(strict=True) / "completion.json"
        require(completion.is_relative_to((Path.home() / 'Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment').resolve()),
                "Priority completion outside Drive experiment storage")
        extra[str(completion)] = sha(completion)
        require(extra[str(completion)] == factorial["run_receipt_sha256"][seq]["A"],
                "Priority completion differs from audited factorial")
        check_priority_input_match(author_proof["records"][seq], read(completion))
    require(all(sha(Path(path)) == digest for path, digest in extra.items()), "Input binding changed during check")
    return extra


def add_cmsa(table: dict, additional: dict, prior: dict, factorial: dict,
             historical: dict, cropped: dict) -> None:
    expected = ADDITIONAL | (PRIORITY - {"seq23"})
    require(set(historical["rows"]) == expected and set(historical["summaries"]) == expected,
            "Historical CMSA sequence coverage differs")
    for seq in expected:
        rows = historical["rows"][seq]
        reference = additional["rows"][seq] if seq in ADDITIONAL else prior["rows"][seq]
        check_same_frames(reference, rows)
        result = summarize(rows, "cmsa")
        check_summary(result, historical["summaries"][seq])
        table[seq][CMSA] = result
    crop_rows = [{**{k: row[k] for k in ("frame", "source_frame_index", "post_prompt", "initialization_frame")},
                  "cmsa": row["full_frame"]} for row in cropped["rows"]]
    check_same_frames(prior["rows"]["seq23"], crop_rows)
    result = summarize(crop_rows, "cmsa")
    check_summary(result, cropped["summaries"]["full_frame"])
    require(cropped["prior_project_lean_postprompt"] == factorial["summaries"]["seq23"]["A"]["post_prompt"],
            "Cropped CMSA prior Lean reference differs")
    table["seq23"][CMSA] = result
    for seq in NO_INITIALIZATION:
        table[seq][CMSA] = None


def verify_cmsa_bindings(historical: dict, historical_audit: dict, historical_proof_path: Path,
                         cropped: dict, cropped_audit: dict, cropped_proof_path: Path,
                         additional_score_sha: str, additional_verification_sha: str,
                         factorial_score_sha: str, factorial_audit_sha: str,
                         reconciliation_path: Path | None = None) -> dict:
    bindings = {}
    for name, score, audit, path in (("historical", historical, historical_audit, historical_proof_path),
                                     ("cropped", cropped, cropped_audit, cropped_proof_path)):
        digest = sha(path)
        require(score.get("verification_sha256") == audit.get("verification_sha256") == digest,
                f"{name} CMSA verification receipt differs")
        proof = read(path)
        require(proof.get("passed") is True and proof.get("label_guard", {}).get("open_attempts") == 0
                and proof.get("scorer_sha256") == score.get("scorer_sha256")
                and proof.get("prior_sha256") == factorial_score_sha
                and proof.get("prior_audit_sha256") == factorial_audit_sha,
                f"{name} CMSA verification source differs")
        if name == "historical":
            require(proof.get("batch_hashes", {}).get("score") == additional_score_sha
                    and proof.get("batch_hashes", {}).get("verification") == additional_verification_sha,
                    "Historical CMSA uses a different comparator batch")
            if reconciliation_path is not None:
                require(Path(proof["reconciliation_path"]).resolve(strict=True)
                        == reconciliation_path.resolve(strict=True)
                        and proof.get("reconciliation_sha256") == sha(reconciliation_path),
                        "Historical CMSA uses a different corrected audit reconciliation")
        require(sha(path) == digest, f"{name} CMSA verification changed")
        bindings[str(path)] = digest
    return bindings


def cmsa_historical_sources(reconciled: bool) -> tuple[str, str]:
    version = "v2" if reconciled else "v1"
    return (f"score_archived_cmsa_automatic_cohort_{version}.py",
            f"audit_archived_cmsa_automatic_cohort_{version}.py")


def render_cohort_summary(title: str, cohort: dict) -> list[str]:
    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines = [f"## {title}", "", f"Sequences: {', '.join(cohort['sequences']) or 'none'}.",
             "", "| Method | Patient-macro Dice | Frame-weighted Dice | Present frames | Empty false alarms / frames |",
             "|---|---:|---:|---:|---:|"]
    for name, metric in cohort["metrics"].items():
        lines.append(f"| {name} | {number(metric['patient_macro_dice'])} | "
                     f"{number(metric['frame_weighted_dice'])} | {metric['present_frames']} | "
                     f"{metric['false_alarms']}/{metric['empty_frames']} |")
    return lines + ["", cohort["cohort_rule"], ""]


def verify_additional_reconciliation(score_path: Path, audit_path: Path, reconciliation_path: Path) -> dict:
    batch = score_path.parent.parent.resolve(strict=True)
    receipt = read(reconciliation_path)
    source = ROOT / "scripts"
    require(receipt.get("passed") is True and receipt.get("label_guard", {}).get("open_attempts") == 0
            and Path(receipt["batch"]).resolve(strict=True) == batch
            and Path(receipt["corrected_audit_path"]).resolve(strict=True) == audit_path.resolve(strict=True),
            "Corrected audit reconciliation identity differs")
    bound = {"batch_run_manifest_sha256": batch / "run_manifest.json",
             "batch_run_map_sha256": batch / "run_map.json", "batch_phase_log_sha256": batch / "phases.jsonl",
             "batch_verification_sha256": batch / "verification/verification.json",
             "batch_scores_sha256": score_path, "failed_audit_log_sha256": batch / "audit.log",
             "failed_audit_source_sha256": source / "audit_additional_author_comparators_v2.py",
             "corrected_audit_sha256": audit_path,
             "corrected_auditor_sha256": source / "audit_additional_author_comparators_v3.py",
             "reconciler_sha256": source / "reconcile_additional_author_audit_v3.py"}
    snapshots = {str(path): sha(path) for path in bound.values()}
    require(all(receipt.get(key) == snapshots[str(path)] for key, path in bound.items()),
            "Corrected audit reconciliation inputs differ")
    audit = read(audit_path)
    require(audit.get("verification_sha256") == snapshots[str(batch / "verification/verification.json")]
            and audit.get("run_map_sha256") == snapshots[str(batch / "run_map.json")],
            "Corrected audit refers to different frozen inputs")
    require(all(sha(Path(path)) == digest for path, digest in snapshots.items()),
            "Corrected audit reconciliation changed while reading")
    snapshots[str(reconciliation_path)] = sha(reconciliation_path)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("additional-score", "additional-audit", "priority-four-score", "priority-factorial-score",
                "priority-factorial-audit", "no-init-score", "no-init-audit", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--author-lean-score", type=Path)
    parser.add_argument("--author-lean-audit", type=Path)
    parser.add_argument("--additional-reconciliation", type=Path,
                        help="Required when using the corrected V3 independent audit.")
    for name in ("cmsa-historical-score", "cmsa-historical-audit", "cmsa-historical-verification",
                 "cmsa-seq23-score", "cmsa-seq23-audit", "cmsa-seq23-verification"):
        parser.add_argument("--" + name, type=Path)
    a = parser.parse_args()
    require((a.author_lean_score is None) == (a.author_lean_audit is None),
            "Author Lean score and independent audit must be supplied together")
    cmsa_names = ("cmsa_historical_score", "cmsa_historical_audit", "cmsa_historical_verification",
                  "cmsa_seq23_score", "cmsa_seq23_audit", "cmsa_seq23_verification")
    cmsa_present = [getattr(a, name) is not None for name in cmsa_names]
    require(all(cmsa_present) or not any(cmsa_present), "Both CMSA scores, audits and verification receipts are required")
    sources = ROOT / "scripts"
    input_names = ("additional_score", "additional_audit", "priority_four_score", "priority_factorial_score",
                   "priority_factorial_audit", "no_init_score", "no_init_audit")
    source_names = ("score_additional_author_comparators_v2.py", "audit_additional_author_comparators_v2.py",
                    "report_sam2long_automatic_priority.py", "score_boundary_video_factorial_v1.py",
                    "audit_boundary_video_factorial_v1.py", "score_unarchived_no_detection_v1.py",
                    "audit_unarchived_no_detection_scores_v1.py")
    if a.additional_reconciliation is not None:
        input_names += ("additional_reconciliation",)
        source_names += ("audit_additional_author_comparators_v3.py", "reconcile_additional_author_audit_v3.py")
    if a.author_lean_score is not None:
        input_names += ("author_lean_score", "author_lean_audit")
        source_names += ("score_author_lean_automatic_v1.py", "audit_author_lean_automatic_v1.py")
    if all(cmsa_present):
        input_names += cmsa_names
        source_names += (*cmsa_historical_sources(a.additional_reconciliation is not None),
                         "score_cmsa_seq23_viewport_v1.py", "audit_cmsa_seq23_viewport_scores_v1.py")
    reports = [getattr(a, n).parent / "report.md" for n in
               ("additional_score", "priority_four_score", "priority_factorial_score", "no_init_score")]
    if a.author_lean_score is not None:
        reports.append(a.author_lean_score.parent / "report.md")
    if all(cmsa_present):
        reports.extend((a.cmsa_historical_score.parent / "report.md", a.cmsa_seq23_score.parent / "report.md"))
    snapshots = {str(p): sha(p) for p in [*(getattr(a, n) for n in input_names), *reports,
                                          *(sources / n for n in source_names), Path(__file__)]}
    auditor = (sources / "audit_additional_author_comparators_v3.py" if a.additional_reconciliation is not None
               else sources / "audit_additional_author_comparators_v2.py")
    additional = verify_scored(a.additional_score, a.additional_audit, "score_sha256",
                               sources / source_names[0], auditor, "auditor_sha256")
    reconciliation_bindings = (verify_additional_reconciliation(a.additional_score, a.additional_audit,
                                                                  a.additional_reconciliation)
                               if a.additional_reconciliation is not None else None)
    if reconciliation_bindings is not None:
        snapshots.update(reconciliation_bindings)
    factorial = verify_scored(a.priority_factorial_score, a.priority_factorial_audit, "score_sha256",
                              sources / source_names[3], sources / source_names[4], "audit_source_sha256")
    no_init = verify_scored(a.no_init_score, a.no_init_audit, "scores_sha256",
                            sources / source_names[5], sources / source_names[6], "audit_source_sha256")
    prior = read(a.priority_four_score)
    require(prior.get("passed") is True and factorial.get("prior_four_system_report_sha256") == sha(a.priority_four_score),
            "Priority comparator is not bound to audited factorial scores")
    require(prior.get("report_sha256") == sha(a.priority_four_score.parent / "report.md"), "Prior report changed")
    require(prior.get("reporter_sha256") == sha(sources / source_names[2]), "Prior reporter source changed")
    table = merge(additional, prior, factorial, no_init)
    methods = METHODS
    if a.author_lean_score is not None:
        author = verify_scored(a.author_lean_score, a.author_lean_audit, "score_sha256",
                               sources / "score_author_lean_automatic_v1.py",
                               sources / "audit_author_lean_automatic_v1.py", "auditor_sha256")
        author_input_bindings = verify_author_input_bindings(a, additional, factorial, author)
        snapshots.update(author_input_bindings)
        add_author_lean(table, additional, prior, author)
        methods += (AUTHOR_LEAN,)
    if all(cmsa_present):
        cmsa_scorer, cmsa_auditor = cmsa_historical_sources(a.additional_reconciliation is not None)
        historical = verify_scored(a.cmsa_historical_score, a.cmsa_historical_audit, "score_sha256",
                                   sources / cmsa_scorer, sources / cmsa_auditor, "auditor_sha256")
        cropped = verify_scored(a.cmsa_seq23_score, a.cmsa_seq23_audit, "score_sha256",
                                sources / "score_cmsa_seq23_viewport_v1.py",
                                sources / "audit_cmsa_seq23_viewport_scores_v1.py", "auditor_sha256")
        cmsa_input_bindings = verify_cmsa_bindings(
            historical, read(a.cmsa_historical_audit), a.cmsa_historical_verification,
            cropped, read(a.cmsa_seq23_audit), a.cmsa_seq23_verification,
            snapshots[str(a.additional_score)], additional["verification_sha256"],
            snapshots[str(a.priority_factorial_score)], snapshots[str(a.priority_factorial_audit)],
            a.additional_reconciliation)
        snapshots.update(cmsa_input_bindings)
        add_cmsa(table, additional, prior, factorial, historical, cropped)
        methods += (CMSA,)
    result = {"passed": True, "scope": "Previously inspected development videos; exploratory, not a blind test.",
              "table": table, "matched_all_methods": aggregate(table, methods),
              "available_per_method": {m: aggregate(table, (m,)) for m in methods},
              "paired_vs_project_lean": {m: paired_comparison(table, "lean", m) for m in methods if m != "lean"},
              "matched_all_methods_empty": aggregate_empty(table, methods),
              "available_empty_per_method": {m: aggregate_empty(table, (m,)) for m in methods},
              "no_initialization": {s: {k: no_init["summaries"][s][k] for k in
                  ("detector_calls", "all_frame_present", "all_frame_empty",
                   "labeled_empty_false_alarms", "labeled_empty_denominator")}
                  for s in sorted(NO_INITIALIZATION)},
              "inputs_sha256": {name: snapshots[str(getattr(a, name))] for name in input_names},
              "source_sha256": {name: snapshots[str(sources / name)] for name in source_names},
              "merger_sha256": snapshots[str(Path(__file__))],
              "author_input_binding_sha256": author_input_bindings if a.author_lean_score is not None else None,
              "cmsa_verification_binding_sha256": cmsa_input_bindings if all(cmsa_present) else None,
              "additional_reconciliation_sha256": reconciliation_bindings,
              "limitations": ["Project Lean is modified, not original author Lean.",
                 "SAM2Long is offline; direct speed/causality equivalence is not implied.",
                 "Patient-macro summaries treat sequences as distinct patients as reported by the user; patient identifiers were not independently verified.",
                 "Seq23 has a retrospectively selected inset-excluded crop.",
                 "Seq1 and seq7 had no initialization; no post-prompt Dice is defined.",
                 ("Missing refiner entries are not zero scores; CMSA is not included here." if not all(cmsa_present) else
                  "CMSA is unprompted and not online-equivalent; 20 runs are historical CPU predictions without recorded inference-time input image hashes, while seq23 is a new cropped MPS run."),
                 ("The original independent audit failed from metadata indexing; a corrected V3 audit verifies the preserved scores and predictions." if reconciliation_bindings is not None else
                  "The additional comparator uses its original independent audit."),
                 ("Original Lean is not included in this report." if a.author_lean_score is None else
                  "Author Lean uses scoped MPS compatibility fixes and retains its own released fallback behavior.")]}
    out = a.output.resolve()
    permitted = (Path.home() / 'Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment').resolve()
    require(out.is_relative_to(permitted) and not out.exists(), "Output must be a new directory in Drive experiment storage")
    require(all(sha(Path(path)) == expected for path, expected in snapshots.items()), "Input changed during merge")
    out.mkdir(parents=True)
    headers = ["Sequence", "Native", "Project Lean", "SAMURAI", "SAM2Long (offline)", "Refiner D"]
    if a.author_lean_score is not None:
        headers.append("Author Lean (MPS fixes)")
    if all(cmsa_present):
        headers.append("CMSA (mixed provenance)")
    lines = ["# Automatic-prompt exploratory comparator summary", "",
             "| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
    for seq in sorted(table, key=lambda s: int(s[3:])):
        cells = []
        for m in methods:
            value = table[seq][m]
            if value is None:
                cells.append("N/A")
            else:
                dice = "N/A" if value["mean_present_dice"] is None else f"{value['mean_present_dice']:.4f}"
                cells.append(f"{dice} ({value['false_alarms']}/{value['empty_frames']})")
        lines.append(f"| {seq} | " + " | ".join(cells) + " |")
    lines += ["", "Cells show present-frame mean Dice (labeled-empty false alarms / empty frames).",
              "Dice-cohort false-alarm totals exclude sequences without present post-prompt frames; separate empty-frame cohorts, including empty-only initialized sequences, are in results.json.",
              "Seq1 and seq7 had no object initialization in the project automatic-prompt run:", "",
              "| Sequence | Detector calls | Labeled-present frames | Empty false alarms / frames |",
              "|---|---:|---:|---:|"]
    for seq, values in result["no_initialization"].items():
        lines.append(f"| {seq} | {values['detector_calls']} | {values['all_frame_present']} | "
                     f"{values['labeled_empty_false_alarms']}/{values['labeled_empty_denominator']} |")
    lines += [""]
    lines += render_cohort_summary("Matched cohort with all included methods", result["matched_all_methods"])
    lines += ["## Available cohort for each method", "",
              "These cohorts can differ; their means are not a matched comparison.", ""]
    for method, cohort in result["available_per_method"].items():
        lines += render_cohort_summary(method, cohort)
    lines += ["## Paired changes relative to project Lean", "",
              "Each row uses only sequences with both methods available and at least one labeled-present post-prompt frame.", "",
              "| Method | Patients | Macro Dice change | Improved | Worsened | Unchanged |", "|---|---:|---:|---:|---:|---:|"]
    for method, paired in result["paired_vs_project_lean"].items():
        difference = paired["patient_macro_dice_difference"]
        difference_text = "N/A" if difference is None else f"{difference:+.4f}"
        lines.append(f"| {method} | {len(paired['sequences'])} | {difference_text} | "
                     f"{len(paired['improved'])} | {len(paired['worsened'])} | {len(paired['unchanged'])} |")
    lines += ["", "Per-sequence changes and membership for every pair are recorded in results.json.", ""]
    lines.extend("- " + x for x in result["limitations"])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    result["report_sha256"] = sha(out / "report.md")
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

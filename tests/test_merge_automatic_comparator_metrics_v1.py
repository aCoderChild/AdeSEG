import copy
import json

import pytest

from scripts import merge_automatic_comparator_metrics_v1 as merger
from scripts.merge_automatic_comparator_metrics_v1 import (
    ADDITIONAL, AUTHOR_LEAN, CMSA, NO_INITIALIZATION, PRIORITY, add_author_lean, add_cmsa, aggregate, aggregate_empty,
    check_additional_input_match, check_priority_input_match, check_same_frames, check_summary, merge, sha, summarize,
    verify_cmsa_bindings,
    verify_additional_reconciliation,
    verify_scored,
)


def row(index, post, present, dice, alarm=False):
    return {"frame": index, "source_frame_index": index, "post_prompt": post,
            "initialization_frame": index == 0,
            "native": {"present": present, "dice": dice, "false_alarm": alarm}}


def test_summarize_excludes_initialization_and_reports_empty_alarms():
    rows = [row(0, False, True, 1.0), row(1, True, True, 0.25),
            row(2, True, False, 1.0, True)]
    result = summarize(rows, "native")
    assert result == {"present_frames": 1, "mean_present_dice": 0.25,
                      "empty_frames": 1, "false_alarms": 1}
    check_summary(result, result)
    with pytest.raises(ValueError, match="Summary mismatch"):
        check_summary(result, result | {"false_alarms": 0})


def test_common_cohort_excludes_missing_and_undefined():
    def value(dice):
        return {"present_frames": int(dice is not None), "mean_present_dice": dice,
                "empty_frames": 1, "false_alarms": 0}
    table = {"seq2": {"native": value(0.8), "lean": value(0.6)},
             "seq10": {"native": value(0.9), "lean": None},
             "seq3": {"native": value(None), "lean": value(None)}}
    result = aggregate(table, ("native", "lean"))
    assert result["sequences"] == ["seq2"]
    assert result["metrics"]["native"]["patient_macro_dice"] == 0.8
    assert result["metrics"]["lean"]["frame_weighted_dice"] == 0.6


def test_report_distinguishes_matched_patient_and_frame_means():
    cohort = {"sequences": ["seq2", "seq3"], "cohort_rule": "matched frames",
              "metrics": {"native": {"patient_macro_dice": 0.75, "frame_weighted_dice": 0.6,
                                      "present_frames": 3, "false_alarms": 1, "empty_frames": 2}}}
    rendered = "\n".join(merger.render_cohort_summary("Matched", cohort))
    assert "seq2, seq3" in rendered
    assert "0.7500 | 0.6000 | 3 | 1/2" in rendered
    assert "matched frames" in rendered


def test_paired_comparison_counts_gains_and_losses_on_matched_sequences():
    def value(dice):
        return {"present_frames": 1, "mean_present_dice": dice, "empty_frames": 0, "false_alarms": 0}
    table = {"seq2": {"lean": value(.4), "refiner": value(.6)},
             "seq3": {"lean": value(.8), "refiner": value(.7)},
             "seq4": {"lean": value(.5), "refiner": value(.5)},
             "seq5": {"lean": value(.9), "refiner": None}}
    paired = merger.paired_comparison(table, "lean", "refiner")
    assert paired["sequences"] == ["seq2", "seq3", "seq4"]
    assert paired["improved"] == ["seq2"] and paired["worsened"] == ["seq3"]
    assert paired["unchanged"] == ["seq4"]
    assert paired["patient_macro_dice_difference"] == pytest.approx(1 / 30)


def test_priority_matching_rejects_frame_or_presence_mismatch():
    rows = [row(0, False, True, 1.0), row(1, True, False, 1.0)]
    check_same_frames(rows, copy.deepcopy(rows))
    changed = copy.deepcopy(rows)
    changed[1]["native"]["present"] = True
    with pytest.raises(ValueError, match="presence"):
        check_same_frames(rows, changed)
    changed = copy.deepcopy(rows)
    changed[1]["source_frame_index"] = 9
    with pytest.raises(ValueError, match="identity"):
        check_same_frames(rows, changed)


def test_invalid_dice_is_not_silently_aggregated():
    with pytest.raises(ValueError, match="Invalid"):
        summarize([row(1, True, True, float("nan"))], "native")


def test_merge_requires_complete_cohort_and_preserves_missing_refiner():
    def with_arms(arms):
        rows = [row(0, False, False, 1.0), row(1, True, True, 0.5), row(2, True, False, 1.0, True)]
        for frame in rows:
            example = frame.pop("native")
            for arm in arms:
                frame[arm] = copy.deepcopy(example)
        return rows

    additional_rows = {s: with_arms(("native", "samurai", "sam2long")) for s in ADDITIONAL}
    additional = {"rows": additional_rows,
                  "summaries": {s: {arm: summarize(rows, arm) for arm in ("native", "samurai", "sam2long")}
                                for s, rows in additional_rows.items()}}
    priority_rows = {s: with_arms(("native", "lean", "samurai", "sam2long")) for s in PRIORITY}
    prior = {"rows": priority_rows,
             "summaries": {s: {arm: summarize(rows, arm) for arm in ("native", "lean", "samurai", "sam2long")}
                           for s, rows in priority_rows.items()}}
    factorial_rows = {s: with_arms(("A", "D")) for s in PRIORITY}
    factorial = {"rows": factorial_rows,
                 "summaries": {s: {arm: {"post_prompt": summarize(rows, arm)} for arm in ("A", "D")}
                               for s, rows in factorial_rows.items()}}
    no_init = {"summaries": {s: {"initialization": "no_detection", "post_prompt_frame_count": 0,
                                 "post_prompt_present_dice": None} for s in NO_INITIALIZATION}}
    result = merge(additional, prior, factorial, no_init)
    assert len(result) == 23
    assert result["seq2"]["lean"] is None
    assert result["seq1"]["native"] is None
    assert result["seq11"]["refiner"]["mean_present_dice"] == 0.5
    assert aggregate(result, ("native", "lean", "refiner"))["sequences"] == ["seq11", "seq20", "seq23"]
    del additional["rows"]["seq2"]
    with pytest.raises(ValueError, match="coverage"):
        merge(additional, prior, factorial, no_init)


def test_uneven_frame_denominators_and_source_binding(tmp_path):
    table = {"seq2": {"native": {"present_frames": 1, "mean_present_dice": 1.0,
                                  "empty_frames": 2, "false_alarms": 1}},
             "seq3": {"native": {"present_frames": 3, "mean_present_dice": 0.0,
                                  "empty_frames": 1, "false_alarms": 1}}}
    metrics = aggregate(table, ("native",))["metrics"]["native"]
    assert metrics == {"patient_macro_dice": 0.5, "frame_weighted_dice": 0.25,
                       "present_frames": 4, "empty_frames": 3, "false_alarms": 2}
    scorer, auditor = tmp_path / "scorer.py", tmp_path / "auditor.py"
    scorer.write_text("scorer")
    auditor.write_text("auditor")
    score, audit = tmp_path / "score.json", tmp_path / "audit.json"
    score.write_text(json.dumps({"passed": True, "scorer_sha256": sha(scorer)}))
    audit.write_text(json.dumps({"passed": True, "score_sha256": sha(score), "auditor_sha256": sha(auditor)}))
    verify_scored(score, audit, "score_sha256", scorer, auditor, "auditor_sha256")
    auditor.write_text("tampered")
    with pytest.raises(ValueError, match="implementation"):
        verify_scored(score, audit, "score_sha256", scorer, auditor, "auditor_sha256")
    auditor.write_text("auditor")
    score.write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError):
        verify_scored(score, audit, "score_sha256", scorer, auditor, "auditor_sha256")


def test_empty_only_sequence_is_counted_in_separate_empty_cohort():
    table = {"seq2": {"native": {"present_frames": 1, "mean_present_dice": 0.5,
                                  "empty_frames": 2, "false_alarms": 1}},
             "seq3": {"native": {"present_frames": 0, "mean_present_dice": None,
                                  "empty_frames": 4, "false_alarms": 2}}}
    assert aggregate(table, ("native",))["sequences"] == ["seq2"]
    result = aggregate_empty(table, ("native",))
    assert result["sequences"] == ["seq2", "seq3"]
    assert result["metrics"]["native"] == {"false_alarms": 3, "empty_frames": 6,
                                             "patient_macro_false_alarm_rate": 0.5}


def test_author_lean_requires_matching_full_cohort_and_frame_identity():
    seqs = ADDITIONAL | PRIORITY
    reference = [row(0, False, False, 1.0), row(1, True, True, 0.4)]
    additional = {"rows": {s: copy.deepcopy(reference) for s in ADDITIONAL}}
    prior = {"rows": {s: copy.deepcopy(reference) for s in PRIORITY}}
    author_rows = {}
    for seq in seqs:
        rows = copy.deepcopy(reference)
        for frame in rows:
            frame[AUTHOR_LEAN] = frame.pop("native")
        author_rows[seq] = rows
    author = {"rows": author_rows,
              "summaries": {s: summarize(rows, AUTHOR_LEAN) for s, rows in author_rows.items()}}
    table = {s: {} for s in seqs | NO_INITIALIZATION}
    add_author_lean(table, additional, prior, author)
    assert table["seq23"][AUTHOR_LEAN]["mean_present_dice"] == 0.4
    assert table["seq1"][AUTHOR_LEAN] is None
    author["rows"]["seq23"][1]["source_frame_index"] = 7
    with pytest.raises(ValueError, match="identity"):
        add_author_lean(table, additional, prior, author)


def test_author_matching_requires_image_prompt_and_viewport_bytes():
    prompt = {"prompt_mask_sha256": "mask", "roi_xyxy": [446, 0, 1280, 720]}
    author = {"images": ["a", "b"], "prompt_frame": 1, "prompt": prompt}
    check_additional_input_match(author, copy.deepcopy(author))
    completion = {"input_images": [{"sha256": "a"}, {"sha256": "b"}],
                  "prompt_frame": 1, "prompt_provenance": copy.deepcopy(prompt)}
    check_priority_input_match(author, completion)
    for key in ("images", "prompt_frame", "prompt"):
        other = copy.deepcopy(author)
        other[key] = None
        with pytest.raises(ValueError, match="mismatch"):
            check_additional_input_match(author, other)
    completion["prompt_provenance"]["roi_xyxy"][0] = 0
    with pytest.raises(ValueError, match="viewport"):
        check_priority_input_match(author, completion)


def test_cmsa_cropped_and_historical_are_matched_without_using_viewport_presence():
    reference = [row(0, False, False, 1.0), row(1, True, True, 0.4)]
    additional = {"rows": {s: copy.deepcopy(reference) for s in ADDITIONAL}}
    prior = {"rows": {s: copy.deepcopy(reference) for s in PRIORITY}}
    historical_rows = {}
    for seq in ADDITIONAL | (PRIORITY - {"seq23"}):
        frames = copy.deepcopy(reference)
        for frame in frames:
            frame["cmsa"] = frame.pop("native")
        historical_rows[seq] = frames
    historical = {"rows": historical_rows,
                  "summaries": {s: summarize(rows, "cmsa") for s, rows in historical_rows.items()}}
    cropped_rows = []
    for frame in reference:
        entry = {key: frame[key] for key in ("frame", "source_frame_index", "post_prompt", "initialization_frame")}
        entry["full_frame"] = copy.deepcopy(frame["native"])
        entry["viewport"] = {"present": False, "dice": 1.0, "false_alarm": False}
        cropped_rows.append(entry)
    cropped = {"rows": cropped_rows, "summaries": {"full_frame": summarize(reference, "native")},
               "prior_project_lean_postprompt": {"marker": 1}}
    factorial = {"summaries": {"seq23": {"A": {"post_prompt": {"marker": 1}}}}}
    table = {s: {} for s in ADDITIONAL | PRIORITY | NO_INITIALIZATION}
    add_cmsa(table, additional, prior, factorial, historical, cropped)
    assert table["seq23"][CMSA]["mean_present_dice"] == 0.4
    assert table["seq1"][CMSA] is None
    cropped["rows"][1]["full_frame"]["present"] = False
    with pytest.raises(ValueError, match="presence"):
        add_cmsa(table, additional, prior, factorial, historical, cropped)


def test_cmsa_proofs_bind_exact_baseline_and_batch(tmp_path):
    score_sha = "a" * 64
    additional_proof_sha = "b" * 64
    factorial_sha, factorial_audit_sha = "c" * 64, "d" * 64
    hist_path, crop_path = tmp_path / "hist.json", tmp_path / "crop.json"
    common = {"passed": True, "label_guard": {"open_attempts": 0}, "scorer_sha256": "s",
              "prior_sha256": factorial_sha, "prior_audit_sha256": factorial_audit_sha}
    historical_proof = common | {"batch_hashes": {"score": score_sha, "verification": additional_proof_sha}}
    hist_path.write_text(json.dumps(historical_proof))
    crop_path.write_text(json.dumps(common))
    historical = {"verification_sha256": sha(hist_path), "scorer_sha256": "s"}
    cropped = {"verification_sha256": sha(crop_path), "scorer_sha256": "s"}
    historical_audit = {"verification_sha256": sha(hist_path)}
    cropped_audit = {"verification_sha256": sha(crop_path)}
    verify_cmsa_bindings(historical, historical_audit, hist_path, cropped, cropped_audit, crop_path,
                         score_sha, additional_proof_sha, factorial_sha, factorial_audit_sha)
    with pytest.raises(ValueError, match="different comparator"):
        verify_cmsa_bindings(historical, historical_audit, hist_path, cropped, cropped_audit, crop_path,
                             "e" * 64, additional_proof_sha, factorial_sha, factorial_audit_sha)
    with pytest.raises(ValueError, match="verification source"):
        verify_cmsa_bindings(historical, historical_audit, hist_path, cropped, cropped_audit, crop_path,
                             score_sha, additional_proof_sha, "f" * 64, factorial_audit_sha)


def test_reconciled_cmsa_uses_versioned_sources_and_exact_receipt(tmp_path):
    assert merger.cmsa_historical_sources(False) == (
        "score_archived_cmsa_automatic_cohort_v1.py", "audit_archived_cmsa_automatic_cohort_v1.py")
    assert merger.cmsa_historical_sources(True) == (
        "score_archived_cmsa_automatic_cohort_v2.py", "audit_archived_cmsa_automatic_cohort_v2.py")
    receipt = tmp_path / "reconciliation.json"
    receipt.write_text("receipt")
    hist_path, crop_path = tmp_path / "hist.json", tmp_path / "crop.json"
    common = {"passed": True, "label_guard": {"open_attempts": 0}, "scorer_sha256": "s",
              "prior_sha256": "prior", "prior_audit_sha256": "audit"}
    hist = common | {"batch_hashes": {"score": "score", "verification": "verify"},
                     "reconciliation_path": str(receipt), "reconciliation_sha256": sha(receipt)}
    hist_path.write_text(json.dumps(hist))
    crop_path.write_text(json.dumps(common))
    historical = {"verification_sha256": sha(hist_path), "scorer_sha256": "s"}
    cropped = {"verification_sha256": sha(crop_path), "scorer_sha256": "s"}
    args = (historical, {"verification_sha256": sha(hist_path)}, hist_path,
            cropped, {"verification_sha256": sha(crop_path)}, crop_path,
            "score", "verify", "prior", "audit", receipt)
    verify_cmsa_bindings(*args)
    receipt.write_text("changed receipt")
    with pytest.raises(ValueError, match="different corrected audit"):
        verify_cmsa_bindings(*args)


def test_corrected_audit_binds_original_failure_and_frozen_batch(tmp_path, monkeypatch):
    repo, batch, corrected = tmp_path / "repo", tmp_path / "batch", tmp_path / "corrected"
    (repo / "scripts").mkdir(parents=True)
    (batch / "verification").mkdir(parents=True)
    (batch / "scores").mkdir()
    corrected.mkdir()
    for name in ("audit_additional_author_comparators_v2.py", "audit_additional_author_comparators_v3.py",
                 "reconcile_additional_author_audit_v3.py"):
        (repo / "scripts" / name).write_text(name)
    paths = {"batch_run_manifest_sha256": batch / "run_manifest.json",
             "batch_run_map_sha256": batch / "run_map.json", "batch_phase_log_sha256": batch / "phases.jsonl",
             "batch_verification_sha256": batch / "verification/verification.json",
             "batch_scores_sha256": batch / "scores/results.json", "failed_audit_log_sha256": batch / "audit.log",
             "failed_audit_source_sha256": repo / "scripts/audit_additional_author_comparators_v2.py",
             "corrected_auditor_sha256": repo / "scripts/audit_additional_author_comparators_v3.py",
             "reconciler_sha256": repo / "scripts/reconcile_additional_author_audit_v3.py"}
    for path in paths.values():
        if not path.exists():
            path.write_text(path.name)
    audit = corrected / "audit.json"
    audit.write_text(json.dumps({"run_map_sha256": sha(paths["batch_run_map_sha256"]),
                                 "verification_sha256": sha(paths["batch_verification_sha256"])}))
    paths["corrected_audit_sha256"] = audit
    rec = {"passed": True, "label_guard": {"open_attempts": 0}, "batch": str(batch),
           "corrected_audit_path": str(audit), **{key: sha(path) for key, path in paths.items()}}
    rec_path = corrected / "reconciliation.json"
    rec_path.write_text(json.dumps(rec))
    monkeypatch.setattr(merger, "ROOT", repo)
    verify_additional_reconciliation(paths["batch_scores_sha256"], audit, rec_path)
    paths["failed_audit_log_sha256"].write_text("altered")
    with pytest.raises(ValueError, match="inputs differ"):
        verify_additional_reconciliation(paths["batch_scores_sha256"], audit, rec_path)

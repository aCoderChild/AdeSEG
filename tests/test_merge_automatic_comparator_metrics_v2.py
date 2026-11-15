from scripts import merge_automatic_comparator_metrics_v2 as merger


def test_reconciled_cmsa_selects_corrected_v3_sources():
    assert merger.cmsa_historical_sources(True) == (
        "score_archived_cmsa_automatic_cohort_v3.py", "audit_archived_cmsa_automatic_cohort_v3.py")
    assert merger.cmsa_historical_sources(False) == (
        "score_archived_cmsa_automatic_cohort_v1.py", "audit_archived_cmsa_automatic_cohort_v1.py")

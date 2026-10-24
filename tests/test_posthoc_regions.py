import numpy as np
import pytest
import zipfile

from scripts.analyze_posthoc_regions import absent_counts, equivalence, rates, attention_summary, case_rows


def test_tost_is_not_failure_to_reject_zero():
    assert equivalence(np.linspace(-0.001, 0.001, 21))["equivalent"]
    assert not equivalence(np.linspace(-0.3, 0.3, 21))["equivalent"]
    assert not equivalence([0.03] * 21)["equivalent"]
    assert not equivalence([0.02] * 21)["equivalent"]
    assert not equivalence([0.0])["equivalent"]
    with pytest.raises(ValueError):
        equivalence([0.0, float("nan")])


def test_false_alarm_denominators_distinguish_new_from_residual():
    def row(before, after):
        return {"present": False, "baseline": {"false_positive_on_absent": before},
                "excluded": {"false_positive_on_absent": after}}
    rows = [row(True, True), row(True, False), row(False, True), row(False, False)]
    counts = absent_counts(rows)
    assert counts == dict(absent=4, baseline_fp=2, excluded_fp=2, baseline_clear=2, induced_fp=1, cleared_fp=1)
    assert rates(counts)["residual_fp_rate"] == 0.5
    assert rates(counts)["induced_fp_rate"] == 0.5
    assert rates(absent_counts([]))["induced_fp_rate"] is None
    assert absent_counts(rows + [{"present": True, "excluded": None}]) == counts


def test_attention_summary_equal_sequence_not_read_weighted():
    def seq(e, share):
        return {"rows": [{"stratum": "all", "definition": "mixed_cells", "layer": "0",
                          "query_region": "0", "memory_region": "2", "enrichment": e,
                          "spatial_attention_share": share}]}
    result = attention_summary({"sequences": [seq(1, .2), seq(9, .8)]})[0]
    assert result["n_sequences"] == 2
    assert result["exp_median_log_enrichment"] == pytest.approx(3)
    assert result["mean_spatial_attention_share"] == pytest.approx(.5)


def test_explicit_artifact_override_does_not_read_original(tmp_path):
    archive = tmp_path / "seq6.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("seq6/cases.jsonl", '{"frame": 4}\n')
    sources = {"seq6": {"path": str(archive), "member": "seq6/cases.jsonl"}}
    assert list(case_rows(tmp_path / "nonexistent", "seq6", sources)) == [{"frame": 4}]

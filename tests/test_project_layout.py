"""Keep the public research tree small and prevent legacy recipe regressions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_package_and_methods():
    for name in ("backbones", "adapters", "losses", "memory", "inference", "evaluation"):
        assert (ROOT / "adeseg" / name / "__init__.py").is_file()
    for name in ("medsam2_high_resolution", "lean_feature_aggregation",
                 "reference_baselines", "exploratory_ablations"):
        assert (ROOT / "experiments" / name / "README.md").is_file()


def test_no_inactive_experimental_runners_in_active_main():
    assert not (ROOT / "protocols").exists()
    assert not (ROOT / "modeling/seq23_viewport.py").exists()
    assert not (ROOT / "scripts/run_lean_auto_multiscale.py").exists()
    assert (ROOT / "experiments/reference_baselines/checkpoint.json").is_file()
    for name in ("train_med", "train_lean", "compare_models", "evaluate_cmsa",
                 "evaluate_lean", "validate_results", "validate_cmsa"):
        assert (ROOT / "scripts" / f"{name}.py").is_file()


def test_historical_compatibility_sources_still_present():
    for name in ("multiscale_channel_value_refiner_v1.py",
                 "structured_feature_aggregation.py",
                 "medsam2_latest.py"):
        assert (ROOT / "modeling" / name).is_file()
    assert (ROOT / "modeling/medical_highres/adapter.py").is_file()


def test_three_module_scientific_structure_and_experiment_docs():
    for name in ("fgpi", "rmfa", "ttmr"):
        assert (ROOT / "adeseg" / "modules" / f"{name}.py").is_file()
        assert (ROOT / "docs" / "methodology" / f"{name.upper()}.md").is_file()
    assert (ROOT / "adeseg" / "modules" / "__init__.py").is_file()
    assert (ROOT / "docs" / "methodology" / "OVERVIEW.md").is_file()
    assert (ROOT / "docs" / "experiments" / "PLAN.md").is_file()


def test_public_main_omits_separate_thesis_and_superseded_documents():
    assert not (ROOT / "docs" / "thesis").exists()
    for relative in (
        "docs/ARCHITECTURE.md",
        "docs/CONTRIBUTIONS.md",
        "docs/MODEL_ARCHITECTURE.md",
        "docs/MEDICAL_HIGHRES_ARCHITECTURE.md",
        "docs/REPORT_FRAMEWORK_2026-11.md",
        "docs/REPORT_FRAMEWORK_2026-11.tex",
        "experiments/reference_baselines/modified_lean_742.md",
    ):
        assert not (ROOT / relative).exists()

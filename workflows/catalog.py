"""Entry points for the current study and its comparison methods.

Commands dispatch existing scripts in separate processes. They do not change
experiment settings or bypass source, data, label or resource checks.
"""

STAGES = {
    "current": ("Three-fold channel-value refiner study", {
        "preflight": "preflight_cache_release_crossfit_training_3fold_v1.py",
        "registry": "build_cache_release_crossfit_registry_3fold_v1.py",
        "study": "run_bounded_crossfit_shared_study_3fold_v1.py",
    }),
    "captures": ("Verify and assemble the warm feature captures", {
        "handoff": "verify_cache_release_crossfit_handoff_v1.py",
        "capture": "run_cache_release_full_study_v2.py",
    }),
    "baselines": ("Author-method comparisons", {
        "cmsa-polypgen": "run_cmsa_polypgen.py",
        "lean-polypgen": "run_lean_polypgen_full.py",
        "failure-analysis": "analyze_author_failure_sequences.py",
    }),
}

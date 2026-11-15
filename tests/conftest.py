"""Scientific test groups, without moving hash-frozen experiment test files."""

from workflows.catalog import STAGES


GROUPS = {
    "data": {"data_pipeline", "instance_masks"},
    "regions": {"memory_token_io", "memory_token_regions", "region_definitions",
                "region_definition_statistics", "trimap_amendment", "full_memory_region_audit"},
    "attention": {"cutie_memory", "cutie_region_queries", "cfbi_compact_snapshot", "cfbi_region_pilot", "cfbi_training", "cfbi_presence_loss", "cfbi_region_gate", "region_attention", "region_readout", "region_mass_attention", "decoder_training_mask", "token_attention", "native_enrichment_calibration",
                  "author_region_measurement"},
    "exclusions": {"region_exclusion", "contiguous_exclusion", "bank_integrity", "v2_target_replay", "v2_trajectory_pulse"},
    "baselines": {"author_baseline", "author_baseline_batch", "efficient_smr_adapter"},
    "history": {"region_sparse_statistics", "sparse_memory", "dense_long_memory", "sam2_clip_sampler",
                "matched_region_memory", "contribution_region_memory", "author_sparse_memory"},
    "followup": {"regional_refinement", "current_frame_refinement", "refinement_diagnostics", "current_frame_cv", "readout_stability", "posthoc_regions", "relapse_opportunities", "stale_foreground",
                 "false_alarm_workflow", "enrichment_readout", "region_bias"},
    "infrastructure": {"tracking", "evaluation", "project_structure"},
}

GROUPS["attention"].update({
    "complementary_memory",
    "davis_region_constrained_tome", "soft_mask_aware_tome", "davis_author_comparison",
    "soft_mask_aware_tome_full", "davis_soft_mask_aware_tome", "region_constrained_tome",
    "cfbi_region_affinity", "cfbi_region_qk_training", "cutie_region_projection",
    "cutie_region_prototypes", "matanyone_temporal_uncertainty",
    "prediction_conditioned_audit", "query_conditioned_region_readout", "regional_low_rank", "regional_low_rank_cv", "multi_object_memory", "native_history_training", "native_history_cache",
    "region_contrast", "region_key_value_gate", "region_value_adapter",
    "region_value_gate", "simvos_region_prototypes", "soft_region_roles",
    "soft_role_affinity", "temporal_region_fusion", "temporal_region_fusion_change",
    "position_consistent_tome",
})


def pytest_configure(config):
    for name in GROUPS:
        config.addinivalue_line("markers", f"{name}: {STAGES[name][0] if name in STAGES else 'Shared infrastructure and project structure'}")


def pytest_collection_modifyitems(items):
    import pytest

    for item in items:
        stem = item.path.stem.removeprefix("test_")
        for group, names in GROUPS.items():
            if stem in names:
                item.add_marker(getattr(pytest.mark, group))

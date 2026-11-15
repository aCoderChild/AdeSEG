"""Register new diagnostic tests alongside the frozen historical group registry."""
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure():
    from conftest import GROUPS

    GROUPS['attention'].update({
        'multiscale_memory_utility',
        'native_decoder_diagnostic', 'native_stage_summary', 'decoder_core_training',
        'decoder_adaptation', 'native_stage_capture', 'ordinary_decoder_cv',
        'regional_memory_prompts',
        'memory_prompt_cv',
        'complementary_memory_diagnostic', 'complementary_memory_replay',
        'complementary_memory_training', 'foreground_context_contrast',
        'memsam_memory_writer', 'regional_token_selection',
    })
    GROUPS['baselines'].update({'vos_benchmarks', 'cmsa_polypgen', 'tcmn_staging', 'lean_polypgen_full', 'audit_lean_polypgen_results', 'analyze_author_failure_sequences'})
    GROUPS['regions'].add('instance_token_contribution')
    GROUPS['history'].update({
        'iterative_memory_pilot', 'iterative_memory_reader', 'token_contribution',
        'token_utility_probe', 'within_frame_token_utility',
        'within_frame_utility_records', 'within_frame_utility_runner',
    })
    GROUPS['followup'].add('summarize_iterative_memory_pilot')
    GROUPS['followup'].add('prompt_feature_refinement')

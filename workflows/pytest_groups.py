"""Register new diagnostic tests alongside the frozen historical group registry."""
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure():
    from conftest import GROUPS

    GROUPS['attention'].update({
        'native_decoder_diagnostic', 'native_stage_summary', 'decoder_core_training',
        'decoder_adaptation', 'native_stage_capture', 'ordinary_decoder_cv',
    })

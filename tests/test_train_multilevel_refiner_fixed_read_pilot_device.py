import sys

import pytest

from scripts import train_multilevel_refiner_fixed_read_pilot_device as runner


def arguments(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['pilot', '--data-preflight', '/tmp/not-read',
        '--output', str(runner.DRIVE_OUTPUT / 'not-created'), '--arm', 'multilevel',
        '--device', 'mps'])


def test_mps_unavailable_fails_without_cpu_fallback(monkeypatch):
    arguments(monkeypatch)
    monkeypatch.setattr(runner.torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='MPS is unavailable'):
        runner.main()


def test_mps_implicit_fallback_is_rejected(monkeypatch):
    arguments(monkeypatch)
    monkeypatch.setattr(runner.torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(RuntimeError, match='fallback must be disabled'):
        runner.main()


def test_mps_training_requires_matching_parity_receipt(monkeypatch):
    arguments(monkeypatch)
    monkeypatch.setattr(runner.torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    monkeypatch.setattr(runner, 'check_manifest', lambda _: ({}, 'manifest'))
    with pytest.raises(ValueError, match='requires a completed CPU/MPS parity receipt'):
        runner.main()

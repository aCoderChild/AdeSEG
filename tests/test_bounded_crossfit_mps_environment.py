import torch

from scripts import bounded_crossfit_shared_worker_3fold_v1 as runtime


def test_mps_environment_enables_determinism_before_identity_check(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    observed = []
    monkeypatch.setattr(runtime, 'environment_identity',
                        lambda: observed.append(torch.are_deterministic_algorithms_enabled()) or {'ok': True})
    try:
        assert runtime.prepare_mps_training_environment({'ok': True}) is None
        assert observed == [True]
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)


def test_mps_environment_rejects_incorrect_recorded_identity(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    monkeypatch.setattr(runtime, 'environment_identity', lambda: {'actual': True})
    try:
        try:
            runtime.prepare_mps_training_environment({'expected': True})
        except ValueError as error:
            assert 'differs from capture' in str(error)
        else:
            raise AssertionError('A different runtime identity must be rejected')
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)

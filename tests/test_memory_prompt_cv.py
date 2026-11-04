import json
import pytest

from scripts.run_regional_memory_prompts_cv import ROOT, main, schedule


def test_frozen_cv_schedule_excludes_validation_and_covers_all_training_sequences():
    protocol = json.loads((ROOT / "protocols/regional_memory_prompts_cv_v1.json").read_text())
    seen = set()
    for fold, validation in enumerate(protocol["folds"], 1):
        updates = schedule(protocol, fold)
        assert len(updates) == 560
        assert not seen.intersection(validation)
        seen.update(validation)
        for epoch in range(40):
            rows = [(n, seed) for e, n, seed in updates if e == epoch]
            assert len(rows) == 14 and len({n for n, _ in rows}) == 14
            assert not set(n for n, _ in rows).intersection(validation)
        assert len({seed for _, _, seed in updates}) == 560
        assert updates == schedule(protocol, fold)
    assert len(seen) == 21 and "seq1" not in seen and "seq7" not in seen


def test_efficacy_cannot_start_without_native_history_training_gate(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_regional_memory_prompts_cv.py", "--checkpoint", "missing",
        "--native-run", "missing", "--propagation-gate", "missing", "--output-dir", "missing"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


def test_incomplete_training_schedule_is_rejected():
    protocol = json.loads((ROOT / "protocols/regional_memory_prompts_cv_v1.json").read_text())
    protocol["epochs"] = 1
    with pytest.raises(RuntimeError, match="schedule"):
        schedule(protocol, 1)

"""Track both compute and loss sources for restart compatibility."""
from pathlib import Path
from training.training_provenance import training_source_digest, FILES, ROOT

def test_training_source_digests_cover_all_declared_files():
    for track, paths in FILES.items():
        assert paths
        assert len(paths) == len(set(paths))
        assert all((ROOT / p).is_file() for p in paths)
        assert len(training_source_digest(track)) == 64

def test_med_and_lean_training_source_fingerprints_differ():
    assert training_source_digest("medical") != training_source_digest("lean")

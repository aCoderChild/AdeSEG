from pathlib import Path

import pytest

from scripts.build_davis_manifest import resolve_davis_root
from scripts.build_polypgen_kaggle_manifest import resolve_sequence_root


def test_resolve_davis_root_below_unknown_kaggle_slug(tmp_path: Path):
    davis = tmp_path / "mounted-with-a-renamed-slug" / "DAVIS"
    (davis / "JPEGImages" / "480p").mkdir(parents=True)
    (davis / "Annotations" / "480p").mkdir(parents=True)

    assert resolve_davis_root(tmp_path) == davis


def test_resolve_polypgen_root_below_unknown_kaggle_slug(tmp_path: Path):
    sequences = tmp_path / "mounted-with-a-renamed-slug" / "positive_cropped"
    (sequences / "seq1").mkdir(parents=True)

    assert resolve_sequence_root(tmp_path) == sequences


def test_dataset_root_resolution_rejects_ambiguity(tmp_path: Path):
    for slug in ("first", "second"):
        davis = tmp_path / slug / "DAVIS"
        (davis / "JPEGImages" / "480p").mkdir(parents=True)
        (davis / "Annotations" / "480p").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="exactly one DAVIS root"):
        resolve_davis_root(tmp_path)

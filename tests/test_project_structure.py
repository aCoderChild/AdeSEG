"""Current project navigation stays valid without executing a study."""

from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from workflows.__main__ import ROOT, main
from workflows.catalog import STAGES


def test_catalog_entries_are_unique_existing_scripts():
    files = [script for _, commands in STAGES.values() for script in commands.values()]
    assert len(files) == len(set(files))
    assert all((ROOT / "scripts" / name).is_file() for name in files)
    assert STAGES["current"][1]["study"] == "run_bounded_crossfit_shared_study_3fold_v1.py"


def test_listing_is_read_only_and_advertises_current_commands(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Listing workflows must not start a process")
    monkeypatch.setattr(subprocess, "run", unexpected)
    for arguments in ([], ["--help"], ["--list"], ["current"]):
        assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "current study --help" in output
    assert "xmem-affinity" not in output


def test_dispatch_preserves_arguments_exit_code_and_isolation(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(subprocess, "run", run)
    args = ["--checkpoint", "/path with spaces/model.pt"]
    assert main(["baselines", "cmsa-polypgen", *args]) == 7
    assert calls == [([sys.executable, str(ROOT / "scripts/run_cmsa_polypgen.py"), *args], {"cwd": ROOT})]


@pytest.mark.parametrize("args", [["unknown"], ["current", "unknown"], ["../scripts", "author"]])
def test_unknown_or_arbitrary_path_is_not_executable(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_current_navigation_documents_have_no_broken_relative_links():
    pages = [ROOT / "README.md", ROOT / "docs/METHODS.md", ROOT / "modeling/README.md",
             ROOT / "scripts/README.md", ROOT / "protocols/README.md", ROOT / "tests/README.md"]
    for page in pages:
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("/"):
                continue
            assert (page.parent / target).exists(), (page, target)

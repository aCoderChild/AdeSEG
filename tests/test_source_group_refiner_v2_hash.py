"""Regression for accidentally returning a capture-receipt hash as manifest hash."""
import os
from pathlib import Path

import pytest

from scripts.memory_token_io import digest
from scripts.source_group_refiner_shared_v2 import load_prepared


@pytest.mark.skipif(not os.environ.get('SOURCE_GROUP_TEST_MANIFEST'),
                    reason='Set SOURCE_GROUP_TEST_MANIFEST to a completed audited manifest directory')
def test_real_prepared_manifest_returns_its_own_hash():
    folder = Path(os.environ['SOURCE_GROUP_TEST_MANIFEST'])
    manifest, manifest_sha = load_prepared(folder)
    assert len(manifest['samples']) == 214
    assert manifest_sha == digest(folder/'manifest.json')
    assert all(manifest_sha != evidence['capture_receipt_sha256']
               for evidence in manifest['sequences'].values())

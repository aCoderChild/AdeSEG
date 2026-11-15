import json

import pytest
import torch

from scripts import train_multilevel_refiner_seq4_empty_v2 as run
from scripts.memory_token_io import digest


def test_mps_parity_receipt_is_bound_to_runtime_and_manifest(tmp_path):
    manifest_sha = 'a' * 64
    receipt = {'passed': True, 'manifest_sha256': manifest_sha,
               'runner_sha256': digest(run.Path(run.__file__)),
               'verifier_sha256': digest(run.ROOT / 'scripts/verify_refiner_seq4_empty_mps_v2.py'),
               'model_source_sha256': digest(run.ROOT / 'modeling/multilevel_contrast_refiner.py'),
               'author_gate_sha256': run.author_sha256(),
               'platform': run.platform.platform(), 'machine': run.platform.machine(),
               'torch_version': str(torch.__version__)}
    path = tmp_path / 'parity.json'
    path.write_text(json.dumps(receipt))
    run.verify_parity(path, manifest_sha)
    for key in ('manifest_sha256', 'model_source_sha256', 'author_gate_sha256', 'platform', 'machine'):
        changed = dict(receipt, **{key: 'different'})
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='parity receipt'):
            run.verify_parity(path, manifest_sha)

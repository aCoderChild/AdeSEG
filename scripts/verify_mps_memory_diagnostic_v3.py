"""Reconstruct a completed V2 short-loader audit before any later capture."""
from pathlib import Path

from scripts.audit_mps_memory_diagnostic_v3 import audit
from scripts.crossfit_observer_capture_shared_v1 import OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.warm_crossfit_video_contract_v1 import read_bound, require


def verify_gate(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve(strict=True)) and path.name == 'audit.json' and path.parent.name == 'audit',
            'Invalid MPS memory diagnostic path')
    root = path.parent.parent; sha = phase_artifact(root, 'audit', path); saved = read_bound(path, sha)
    require(saved.get('passed') is True and saved.get('label_guard', {}).get('open_attempts') == 0,
            'MPS memory diagnostic did not pass')
    rebuilt = audit(root)
    require({k:v for k,v in saved.items() if k != 'label_guard'} == rebuilt and digest(path) == sha and
            rebuilt.get('passed') is True, 'MPS memory diagnostic reconstruction differs')
    return saved, sha

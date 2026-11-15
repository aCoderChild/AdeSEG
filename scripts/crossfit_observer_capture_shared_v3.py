"""Versioned observer-only provenance; retain and validate the V1 short gate."""
from scripts.crossfit_observer_capture_shared_v1 import *  # frozen V1 constants and checks
from scripts.crossfit_observer_capture_shared_v1 import source_hashes as v1_source_hashes
from scripts.memory_token_io import digest


def source_hashes():
    result=v1_source_hashes()
    for path in ('scripts/crossfit_observer_capture_shared_v3.py',
                 'scripts/run_crossfit_observer_capture_v3.py',
                 'scripts/audit_crossfit_observer_full_sequence_v3.py',
                 'scripts/run_crossfit_observer_capture_study_v3.py',
                 'scripts/resume_source_group_capture_v1.py',
                 'protocols/EXPLORATORY_CROSSFIT_OBSERVER_CAPTURE_V3.md'):
        result[path]=digest(ROOT/path)
    return result

"""Compatibility import for FGPI prompt construction.

Canonical implementation: adeseg.modules.fgpi.prompts.
Kept so historical scripts and third-party imports continue to work.
"""
from adeseg.modules.fgpi.prompts import (
    _validate_box, feature_guided_prompt, apply_fgpi_prompt,
)

__all__ = ["feature_guided_prompt", "apply_fgpi_prompt"]

#!/usr/bin/env python3
"""Run the multi-object adenoid video-segmentation pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adenoid.pipeline import parse_adenoid_args, run_adenoid_pipeline


if __name__ == "__main__":
    run_adenoid_pipeline(parse_adenoid_args())

"""Evaluate the GT-mask-initialized Lean-SAM2 baseline; NOT the 0.742 automatic-prompt run.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from scripts.run_lean_polypgen_full import main

if __name__ == "__main__":
    main()

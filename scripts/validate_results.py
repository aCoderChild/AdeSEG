"""Independently rescore saved Lean/native masks and compare frame-level outputs.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from scripts.audit_lean_polypgen_results import main

if __name__ == "__main__":
    main()

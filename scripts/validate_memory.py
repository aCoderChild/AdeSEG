"""Independently verify automatic memory-anchor predictions and saved prompt receipts.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from scripts.audit_lean_auto_multiscale import main

if __name__ == "__main__":
    main()

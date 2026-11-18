"""Independently rescore CMSA predictions under its saved comparison protocol.

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from scripts.audit_cmsa_polypgen import main

if __name__ == "__main__":
    main()

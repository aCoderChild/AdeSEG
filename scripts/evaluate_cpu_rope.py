"""Evaluate the CPU RoPE frequency-repeat correction (separate Lean ablation).

Friendly CLI alias. The original source and hash-bound runner remain untouched.
"""
from scripts.run_lean_rope_cpu_fix import main

if __name__ == "__main__":
    main()

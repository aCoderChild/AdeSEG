#!/usr/bin/env bash
set -euo pipefail

KERNEL_SLUG="${1:-bunnoob2005/utility-aware-memory-retrieval-adeseg}"
OUTPUT_ROOT="${2:-/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026}"

mkdir -p "$OUTPUT_ROOT"
kaggle kernels output "$KERNEL_SLUG" --path "$OUTPUT_ROOT" --force
printf 'Kaggle outputs saved directly to %s\n' "$OUTPUT_ROOT"

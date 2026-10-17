#!/usr/bin/env bash
set -euo pipefail

KERNEL_SLUG="${1:-bunnoob2005/adeseg-latent-utility-must-phase-a}"
OUTPUT_ROOT="${2:-/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026}"

mkdir -p "$OUTPUT_ROOT"
kaggle kernels output "$KERNEL_SLUG" --path "$OUTPUT_ROOT" --force
printf 'Kaggle outputs saved directly to %s\n' "$OUTPUT_ROOT"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ -x "$REPO_ROOT/.venv/bin/wandb" ]]; then
        WANDB_CLI="$REPO_ROOT/.venv/bin/wandb"
    else
        WANDB_CLI="$(command -v wandb || true)"
    fi
    if [[ -n "$WANDB_CLI" ]]; then
        while IFS= read -r -d '' RUN_DIR; do
            WANDB_BASE_URL=https://forge.coreweave.com/api/wandb \
                "$WANDB_CLI" sync --project ade-seg --entity adenoid-hypertrophy "$RUN_DIR"
        done < <(find "$OUTPUT_ROOT/latent_utility_must_phase_a" -type d -name 'offline-run-*' -print0 2>/dev/null)
    else
        printf 'W&B CLI unavailable; offline runs remain in Drive for later sync\n' >&2
    fi
fi

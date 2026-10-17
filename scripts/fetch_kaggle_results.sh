#!/usr/bin/env bash
set -euo pipefail

KERNEL_SLUG="${1:-bunnoob2005/adeseg-latent-utility-must-phase-a}"
OUTPUT_ROOT="${2:-/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026}"
VERSION_ID="${3:-}"
EXPECTED_TRACKING="${4:-any}"

STATUS="$(kaggle kernels status "$KERNEL_SLUG")"
if [[ "$STATUS" == *RUNNING* || "$STATUS" == *QUEUED* || "$STATUS" == *PENDING* ]]; then
    printf 'Kaggle run has not finished: %s\n' "$STATUS" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
kaggle kernels output "$KERNEL_SLUG" --path "$OUTPUT_ROOT" --force
printf 'Kaggle outputs saved directly to %s\n' "$OUTPUT_ROOT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION=("$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/validate_kaggle_results.py" "$OUTPUT_ROOT/latent_utility_must_phase_a")
if [[ "$EXPECTED_TRACKING" == online ]]; then
    VALIDATION+=(--require-online)
fi
"${VALIDATION[@]}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
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
    if [[ -n "$VERSION_ID" ]] && rg -q '"all_online": false' "$OUTPUT_ROOT/latent_utility_must_phase_a/delivery_receipt.json"; then
        "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/backfill_kaggle_wandb.py" \
            "$OUTPUT_ROOT/latent_utility_must_phase_a" --version-id "$VERSION_ID"
    fi
fi

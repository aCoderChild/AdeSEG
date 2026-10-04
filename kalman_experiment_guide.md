# Kalman memory: experiment guide

How to run the Kalman-memory experiments on PolypGen, which tables to read, and
what to do when something goes wrong. Method and protocol details are in
`EXPERIMENTS.md`.

## 1. Setup

All results go to Google Drive; nothing is written under the repository.

```bash
cd /Users/maianhpham/Documents/AdeSEG
OUT="/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/Kalman"

# Test one run on all 23 positive sequences: $1 = run name, the rest goes to infer.py.
run_test() {
  caffeinate -i python3 scripts/infer.py --device mps --prompt_source gt_box -o "$OUT/$1/masks" "${@:2}" &&
  caffeinate -i python3 evaluation/temporal.py --output_mask_dir "$OUT/$1/masks"
}

# Train one run: $1 = run name, the rest goes to train_kalman.py.
run_train() {
  setopt local_options pipefail
  caffeinate -i python3 scripts/train_kalman.py --device mps --output_dir "$OUT/$1" "${@:2}" 2>&1 | tee "$OUT/$1.log"
}

# Train, then test with the final checkpoint.
run_kalman() {
  run_train "$@" &&
  run_test "$1" --memory_backend kalman --kalman_checkpoint "$OUT/$1/kalman_memory.pt"
}
```

`caffeinate -i` keeps the Mac awake while a step runs (sleep would pause
training). Keep the lid open or the Mac on power. Always quote `"$OUT"` (the
path contains a space). Use `--device mps` on this
Mac; `train_kalman.py` defaults to `cuda`.

Each run directory ends up as:

| Path | Written by | Content |
|---|---|---|
| `$OUT/<run>/setup.json`, `history.jsonl`, `kalman_memory*.pt` | `train_kalman.py` | settings, per-clip training log, checkpoints |
| `$OUT/<run>/masks/` | `infer.py` | predicted masks, `diagnostics/seqN.csv` (object score, Kalman gain/variance per frame), `efficiency.json`, `prompt_records.json`, `run_manifest.json` |
| `$OUT/<run>/evaluation/` | `temporal.py` | `metrics_per_frame.csv`, `metrics_per_sequence.csv`, `presence_stratified.csv`, `drift_by_*.csv`, overlays |
| `$OUT/summary/` | `summarize_kalman_runs.py` | the comparison tables (section 4) |

Approximate cost on this Mac (MPS): training 1,000 steps x 4 clips of 16 frames
takes about 2 h 10 min (1.9 s per clip); testing takes about 2 minutes of
inference plus 3 minutes of evaluation. The full run list (B1–C3 plus two seeds)
is about 17 hours of training. Overlays are about 1.7 GB per run, so ten runs use about 17 GB of
Drive space; Drive for desktop also caches them on the local disk (58 GB free
at the time of writing) until they are uploaded.

## 2. Checks before any training

Run once, and again after any code change. Stop if either fails.

```bash
python3 scripts/check_split_overlap.py          # expects 12 excluded single frames
python3 scripts/verify_kalman_parity.py --perturb --sequence seq20
python3 scripts/verify_kalman_parity.py --perturb --sequence seq5
```

| Check | Pass condition |
|---|---|
| Split overlap | 12 excluded single frames; 0 negative frames duplicate a test frame |
| Parity, `kalman` | mask IoU 1.0, object-score error < 0.05, gain error < 0.01 |
| Parity, `native_teacher` | mask IoU 1.0, object-score error < 0.05 |

## 3. Runs

Seed 0 throughout, `hard` clips, 1,000 steps x 4 clips, `--absence_weight 0.1`
unless stated. Names matter: the summary script uses `native_gtbox` and
`kalman_untrained` as reference runs.

| ID | Run name | Commands | Question it answers |
|---|---|---|---|
| B1 | `native_gtbox` | `run_test native_gtbox --memory_backend native` | Reference: native MedSAM2 |
| B2 | `kalman_untrained` | `run_kalman kalman_untrained --steps 0` | Does the architecture help without training? |
| A1 | `kalman_aw0_s0` | `run_kalman kalman_aw0_s0 --absence_weight 0` | Absence supervision off |
| A2 | `kalman_aw0.1_s0` | `run_kalman kalman_aw0.1_s0` | Main configuration |
| A3 | `kalman_aw1_s0` | `run_kalman kalman_aw1_s0 --absence_weight 1.0` | Strong absence supervision |
| C1 | `kalman_nodistill_s0` | `run_kalman kalman_nodistill_s0 --distill_weight 0` | Does distillation from the native bank help? |
| C2 | `kalman_easy_s0` | `run_kalman kalman_easy_s0 --clip_difficulty easy` | Do the harder clips help? |
| C3 | `kalman_noabsent_s0` | `run_kalman kalman_noabsent_s0 --absence_probability 0` | Does training with absent stretches help? |
| S1, S2 | `kalman_<best>_s1`, `_s2` | best A-row with `--seed 1`, `--seed 2` | Is the effect larger than seed variation? |

Order: B1, B2, then A1–A3, then C1–C3, then seeds for the best A-row.
Optional: `run_test native_yolo --memory_backend native --prompt_source yolo`.
The YOLO detector was most likely trained on PolypGen, so treat YOLO-prompt
results as secondary.

To run a training job in the background and keep the terminal free:

```bash
nohup zsh -c "$(typeset -f run_train run_test run_kalman); OUT='$OUT'; run_kalman kalman_aw0.1_s0" > /dev/null 2>&1 &
```

Not implemented yet, but expected by reviewers: native MedSAM2 with a trimmed
bank (1, 2, 4 frames), a fixed-gain moving average (EMA-SAM style), and
SAM2Long / DAM4SAM. Report them as missing rather than leaving them out silently.

## 4. Tables

After each batch of runs:

```bash
python3 scripts/summarize_kalman_runs.py --root "$OUT"
```

This writes five CSV files to `$OUT/summary/`. All metrics use propagated
frames only (after the prompt frame).

### `main_results.csv`: one row per run

| Column | Meaning | Better |
|---|---|---|
| `propagated_dice` | mean Dice over all propagated frames (empty prediction on an empty frame scores 1) | higher |
| `present_dice` | mean Dice on frames with a polyp | higher |
| `absent_dice` | mean Dice on empty frames (1 = correctly empty, 0 = any false positive) | higher |
| `absent_fp_rate` | share of empty frames with any predicted pixel | lower |
| `present_detection_rate` | share of polyp frames with any predicted pixel | higher |
| `presence_auroc` | how well the raw object score separates polyp frames from empty frames, independent of the threshold | higher |
| `reappear_first_after_ge1`, `_ge7` | Dice on the first polyp frame after an absence of at least 1 / 7 frames | higher |
| `reappear_1to4_after_ge7` | Dice 1–4 frames after such a reappearance | higher |
| `reappear_ge10_after_ge1` | Dice 10 or more frames after a reappearance (long-term tracking) | higher |
| `retained_memory_mib`, `fps` | retained spatial memory per video and speed | lower / higher |

`retained_memory_mib` for native counts every stored frame memory, although
MedSAM2 attends to only 7; compare memory size against a trimmed bank, not
this number.

### `paired_tests.csv`: each run against `native_gtbox` and `kalman_untrained`

Per-sequence paired comparison on `propagated_dice`, `present_dice`,
`absent_fp_rate`, and `present_detection_rate`: mean difference, number of
sequences better / worse / tied, and the Wilcoxon signed-rank p-value. 21
sequences contain a polyp; 17 contain empty frames.

### `per_sequence.csv`: run x sequence

The same four metrics per sequence, with `c6` marking seq16–seq23 (all from
center C6). Use it for the per-sequence figure and to explain outliers.

### `failure_cases.csv`

Sequences where a run loses more than 0.05 propagated Dice against
`native_gtbox`, or raises the empty-frame false-positive rate by more than 0.2.
Change the thresholds with `--dice_drop` and `--fp_rise`.

### `training_summary.csv`

Mean of each training statistic over the first and last 20% of clips (at most
100 clips each), with two flags:

- `flag_gain_absent_above_0.8`: the learned gain on empty frames is close to the
  gain on polyp frames, so absent frames overwrite the state;
- `flag_gain_gap_below_0.1`: present and absent gains are nearly equal.

## 5. Analysis

Answer each question with the named comparison. Treat a difference as
supported only if `paired_tests.csv` shows p < 0.05 and it holds across seeds.

| Question | Compare | Supported if | Earlier v1 result for context |
|---|---|---|---|
| Does the architecture help without training? | B2 vs B1 | `present_dice` and `reappear_ge10_after_ge1` higher, p < 0.05 | +0.019 present Dice, 14/21 sequences, p = 0.07 |
| Does training add to the architecture? | A2 vs B2 | `present_dice` not lower, `absent_fp_rate` lower | NPO run: present Dice -0.032, p = 0.02 (worse) |
| Does absence supervision reduce false positives, and at what cost? | A1, A2, A3 | `absent_fp_rate` falls with the weight while `present_detection_rate` holds | NPO run: FP 0.65 to 0.60 (p = 0.18), detection 0.875 to 0.858 (p = 0.07) |
| Is it a real improvement or a threshold shift? | `presence_auroc` across A1–A3 | AUROC rises with the weight | not measured |
| Distillation | C1 vs A2 | A2 better on `present_dice` | not measured |
| Harder clips | C2 vs A2 | A2 better on real test videos | `easy` clips scored 0.89 Dice vs 0.54 on real videos |
| Absent stretches in training | C3 vs A2 | A2 better on `absent_fp_rate` and reappearance rows | not measured |
| Robustness | seeds | seed standard deviation smaller than the effect | not measured |

Report in the write-up:

1. **Main table**: B1, B2, best A-row (mean ± std over three seeds) with all
   `main_results.csv` columns except efficiency.
2. **Absence ablation**: A1–A3 with `absent_fp_rate`,
   `present_detection_rate`, `present_dice`, `presence_auroc`.
3. **Component ablation**: A2, C1, C2, C3.
4. **Paired tests**: the rows of `paired_tests.csv` behind every claim.
5. **Failure cases**: `failure_cases.csv` with an overlay figure per sequence
   (section 6).
6. **Efficiency**: retained memory and FPS, with the caveat above.

## 6. Failure cases and what to do

| Symptom | Likely cause | Check | Action |
|---|---|---|---|
| Parity check fails | training and inference code diverged | which field fails (mask, score, gain) | do not train; fix the code first |
| `FloatingPointError: Kalman state became non-finite` | MPS read of an offloaded tensor, or a numerical overflow | rerun the same sequence with `--device cpu` | if CPU is fine, report it as an MPS issue; if not, inspect the state in `diagnostics/seqN.csv` |
| Logged loss jumps to 50–150 | a polyp frame with object score <= 0 gets mask logits of -1024 (MedSAM2 behavior), so the focal term is large | `gradient_norm` in `history.jsonl` stays below about 1 | expected; no action |
| `flag_gain_absent_above_0.8` is True | training learned to write absent frames into the state, undoing the presence gating (seen in the v1 run) | `gain_absent_start` vs `_end` in `training_summary.csv` | report it; a follow-up is to fix `absence_scale` during training |
| A run collapses on one sequence (seq6, seq21 in v1) | memory drift or a wrong presence decision after reappearance | `failure_cases.csv`; `presence` and `gain_mean` columns in `masks/diagnostics/seqN.csv`; overlays in `evaluation/overlays/seqN/` | describe it as a failure case with its overlay frames |
| `absent_fp_rate` identical across all runs on many sequences | the frozen decoder's object score decides absence; the memory cannot change it | `per_sequence.csv`: same value for every run | report as a limitation of a memory-only change |
| `absent_fp_rate` falls but `present_detection_rate` falls too | the model became more conservative overall | `presence_auroc` | if AUROC does not rise, it is a threshold shift, not better separation |
| Differences are not significant | 21 sequences and one seed | `paired_tests.csv`, seed spread | add seeds; report the effect as not significant |
| Drive errors (`Resource deadlock avoided`, missing or partial files) | Drive for desktop still syncing, or not running | Drive app status; file sizes in `$OUT/<run>` | keep the Drive app running; rerun only the failed step (`run_test` overwrites masks and evaluation) |
| Disk or Drive quota full | overlays (about 1.7 GB per run) | `df -h`, Drive storage page | free space before the next run; overlays are required for failure figures |

## 7. Record keeping

- Do not select checkpoints or hyperparameters using test results; the final
  checkpoint (`kalman_memory.pt`) is the one to test.
- Keep the configuration in `setup.json` and the commit hash with each run
  (`git rev-parse HEAD > "$OUT/<run>/commit.txt"`).
- Copy the final tables into `EXPERIMENTS.md` with the run names.

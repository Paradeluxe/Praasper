# Parallel Grid Search — Reworked Plan

## Problem with the Original Plan

The original plan specified `--parallel 2`, which causes a **thread-safety race condition** in Praasper's `annote()`. All threads share the model (loaded once in main process), and `annote()` computes `tmp_path = os.path.join(dir_name, "tmp")`. When the grid script passes temp dirs under `/tmp`, this resolves to `/tmp/tmp/` — a **single shared directory**. Every `annote()` call does `shutil.rmtree(tmp_path)` then `os.makedirs(...)`, so concurrent threads delete each other's intermediate audio clips.

**Evidence**: 25 of 436 existing rows (5.7%) have `swer=1.0` with errors like:
```
Failed to load audio file /tmp/tmp/01-1_141059.206...wav: No such file or directory
```

## Current State

| Metric | Value |
|--------|-------|
| Total combos | ~673,920 (208 files × 3,240 combos) |
| Already completed (valid) | ~411 rows in results_all.csv |
| Error rows (need re-run) | 25 rows (swer=1.0, skipped by resume-skip) |
| Remaining combos | ~673,509 |
| Error rate at --parallel 2 | ~5.7% → ~38,400 more errors expected |

## Decision: Patch the Race Condition

Rather than accepting ~6% data loss (≈40k wasted combos), patch `annote()` to use thread-safe temp directories. The fix: make `tmp_path` unique per `annote()` call by using the input dir name as a suffix.

### Patch `praasper/__init__.py` — Line 209

Change:
```python
tmp_path = os.path.join(dir_name, "tmp")
```
To:
```python
import hashlib
_safe_name = hashlib.md5(input_path.encode()).hexdigest()[:8]
tmp_path = os.path.join(dir_name, f"tmp_{_safe_name}")
```

This gives each grid-script-created temp dir its own `/tmp/tmp_<hash>/` directory. No more collisions.

### Patch `praasper/__init__.py` — Line 232-233

Change:
```python
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
```
To:
```python
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
```
(No change needed here — with unique tmp_path per call, the rmtree is safe.)

**Total patches**: 1 line change in `praasper/__init__.py`.

## Reworked Plan Table

| Parameter | Value | Purpose / Reasoning |
|-----------|-------|---------------------|
| `--all` | Enabled | Run across all 208 audio files |
| `--parallel` | **2** | Two concurrent combo workers. Safe after patching `annote()`'s tmp_path race. Model loaded once in main process (single shared copy). |
| `--resume-skip` | Enabled | Skips valid combos already in results_all.csv. 25 error rows will be re-run (correct behavior). |
| `--audio-dir` | `/mnt/d/audio_data` | WSL-native path, fast I/O, 208 .wav files |
| `--answers-dir` | `/mnt/d/audio_data/answers` | 208 KEYa.TextGrid ground truth files |
| `--results-dir` | `/mnt/d/hermes_playground/Praasper/results` | Output for CSV + per-file TextGrids |
| `--limit` | unset | All 3,240 combos per file (no cap) |
| ffmpeg | `/tmp/ffmpeg-7.0.2-amd64-static` | Required for audio loading |
| **VRAM** | **~4.5GB** | Single shared model instance. NOT 2× model copies (ThreadPoolExecutor shares memory). |
| **Total combos** | ~673,920 | 208 × 3,240. Already done (valid): ~411. Error rows to re-run: 25. |
| **Time estimate** | **~55 days at --parallel 2** (~7s/combo), or **~110 days at --parallel 1** (~14s/combo) | 673,509 remaining ÷ 2 workers ÷ 86400 sec/day × ~7s. |

## Launch Command

```bash
# 1. Patch annote() for thread-safe temp dirs
# (one-line change: tmp_path uses input_path hash suffix)

# 2. Run
export PATH="/tmp/ffmpeg-7.0.2-amd64-static:$PATH"
cd /mnt/d/hermes_playground/Praasper
source .venv/bin/activate
python "test scripts/grid_new_praasper.py" \
  --all --parallel 2 --resume-skip \
  --audio-dir /mnt/d/audio_data \
  --answers-dir /mnt/d/audio_data/answers \
  --results-dir /mnt/d/hermes_playground/Praasper/results
```

## What happens next

1. Patch `annote()` tmp_path to be unique per input directory
2. Script starts, loads existing results_all.csv to build skip set (~411 valid combos skipped, 25 error rows re-queued)
3. Model loaded once in main process (shared by all threads via ThreadPoolExecutor)
4. VAD segmentation → 3,240 combos per file processed with 2 parallel workers
5. Results appended incrementally to CSV
6. Background process notifies when complete or if error occurs

## Verification After Launch

After ~5 minutes, check error rate on newly-written rows:
```bash
# Wait for ~50+ new rows, then check:
tail -50 results_all.csv | grep -c 'swer=1.0'  # should be 0 after patch
```

If error rate drops to 0%, the patch works. If errors persist, the race is elsewhere and we fall back to `--parallel 1`.

## Fallback

If patching doesn't eliminate errors (race condition is deeper in the call stack), use `--parallel 1`:
- Zero errors guaranteed
- ~110 days estimated
- Or: run `--parallel 2` and accept ~6% error rate, filtering `swer=1.0` rows during analysis

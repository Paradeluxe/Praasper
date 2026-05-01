# VAD Grid Search: Extended Amp + Eps Sweet Range

> **For Hermes:** Run directly — no subagents needed.

**Goal:** Test if amp=1.5/2.0 improves hit on quiet speech, and fine-tune eps in the sweet range (0.025–0.045).

**Architecture:** Modify the grid arrays in `grid_new_praasper.py`, run on the existing 8 files with `--resume-skip`, analyze results.

**Tech Stack:** Praasper VAD grid search, 8 audio files (01-1 through 02-4), --parallel 1

---

## Context

From the 8-file search (3,936 combos):
- **Locked:** c0=0, c1=5400, nv=5000
- **Previously ranged:** amp=[1.01, 1.1, 1.2], eps=[0.01–0.05 at 0.005 steps]
- **Finding:** amp is the primary hit↔eff lever, eps is the fine-tuning dial
- **Question:** Does amp > 1.2 capture significantly more quiet speech? What's the eps sweet spot at high amp?

## Proposed Grid

| Param | Values | Count | Rationale |
|-------|--------|-------|-----------|
| **amp** | 1.01, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0 | 7 | Extended range. 1.3 tests moderate push, 1.5 tests aggressive, 2.0 tests extreme |
| **c0** | 0 | 1 | Locked (sub-0.5pp effect) |
| **c1** | 5400 | 1 | Locked (best eff/sWER balance) |
| **nv** | 5000 | 1 | Locked (dominates top-20) |
| **eps** | 0.025, 0.03, 0.035, 0.04, 0.045 | 5 | Sweet range. Previous results showed 0.03–0.045 optimal |

**Total:** 7 × 5 = **35 combos/file**
**8 files:** 35 × 8 = **280 combos**
**ETA:** ~6 hours at --parallel 1 (~0.78 combos/min)

## Files to Modify

**`test scripts/grid_new_praasper.py`** — lines ~230-245 (grid definition):

```python
# Change from:
AMP_VALS = [1.01, 1.1, 1.2]
CUTOFF0_VALS = [0, 200]
CUTOFF1_VALS = [3600, 5400, 12600]
NUMVALID_VALS = [1000, 5000, 10000]
EPS_VALS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]

# To:
AMP_VALS = [1.01, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]
CUTOFF0_VALS = [0]
CUTOFF1_VALS = [5400]
NUMVALID_VALS = [5000]
EPS_VALS = [0.025, 0.03, 0.035, 0.04, 0.045]
```

## Execution Plan

### Step 1: Backup CSV
```bash
cp results/results_all.csv results/results_all_prev.csv
```

### Step 2: Patch grid arrays
```bash
patch the 5 grid value arrays in grid_new_praasper.py
```

### Step 3: Run grid search
```bash
cd /mnt/d/hermes_playground/Praasper
PATH="/tmp/ffmpeg-7.0.2-amd64-static:$PATH" .venv/bin/python "test scripts/grid_new_praasper.py" \
  01-1 01-2 01-3 01-4 02-1 02-2 02-3 02-4 \
  --audio-dir /mnt/d/audio_data \
  --answers-dir /mnt/d/audio_data/answers \
  --results-dir /mnt/d/hermes_playground/Praasper/results \
  --parallel 1 \
  --resume-skip
```

### Step 4: Analyze
- Run top-10 by hit → eff → sWER
- Plot amp vs hit/eff to find the knee point
- Check if amp=2.0 causes OOM or audio clipping issues

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| amp=2.0 clips audio / breaks ASR | Medium | Monitor for sWER spike, kill if errors > 50% |
| OOM on GPU | Low | Same model, just different amp. amp doesn't affect VRAM |
| Resume-skip confusion | Low | New grid = new combo_idx mapping. Backup CSV protects us |

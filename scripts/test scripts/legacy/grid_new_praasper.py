"""
grid_new_praasper.py
====================
Grid search for Praasper VAD+ASR params using the new Praasper (annote()).

Single-audio workflow:
  1. For each combo (cutoff0, cutoff1, numValid, eps_ratio):
       - Create a temp dir with a symlink to the audio
       - Call model.annote(input_path=temp_dir, params=dict)
       - Praasper writes output/output/<folder>/<fname>.TextGrid
       - Read that TextGrid
       - Score with score_pinyin_wer.py
       - Cleanup temp dir
  2. Rank results by sWER (ascending), save results_ranked.json

Usage:
  # Smoke test — 2 combos:
  python grid_new_praasper.py --audio 01-1 --limit 2

  # Full grid on 01-1:
  python grid_new_praasper.py --audio 01-1

  # Full grid on all audio files:
  python grid_new_praasper.py --all
"""

import os
import sys
import json
import time
import shutil
import csv
import math
import tempfile
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Paths ──────────────────────────────────────────────────────────────────────
# Defaults — override via CLI args: --audio-dir, --answers-dir, --results-dir, etc.
SCRIPT_DIR    = Path(__file__).resolve().parent
PRAASPER_REPO = SCRIPT_DIR.parent           # ../  (repo root)
HERMES_HUNTER = PRAASPER_REPO.parent / "hunter"
AUDIO_DIR     = Path("/mnt/e/ProjLegacy/Test_All_Models_for_Praasper/audio")
ANSWERS_DIR   = Path("/mnt/e/ProjLegacy/Test_All_Models_for_Praasper/answers")
RESULTS_DIR   = HERMES_HUNTER / "results"
MODELSCOPE_CACHE = "/mnt/e/modelscope_cache"

# Patch scoring script path (HERMES_HUNTER may not exist yet — handled at runtime)
sys.path.insert(0, str(HERMES_HUNTER))

# ── Master CSV ─────────────────────────────────────────────────────────────────
CSV_PATH = RESULTS_DIR / "results_all.csv"
CSV_HEADER = [
    "file_key", "rep", "combo_idx",
    "amp", "cutoff0", "cutoff1", "numValid", "eps_ratio",
    "swer", "errors", "gt_syllables", "num_intervals",
    "hit_rate", "eff_rate", "error",
]

# ── Param grid ────────────────────────────────────────────────────────────────
AMP_VALS      = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]  # 8 — extended range
CUTOFF0_VALS  = [0]                                          # 1 — locked
CUTOFF1_VALS  = [5400]                                       # 1 — locked
NUMVALID_VALS = [5000]                                       # 1 — locked
EPS_VALS      = [0.02, 0.04, 0.06]                           # 3 — sweet spots
# Total: 8×1×1×1×3 = 24 combos/file

GRID = {
    "amp":        AMP_VALS,
    "cutoff0":    CUTOFF0_VALS,
    "cutoff1":    CUTOFF1_VALS,
    "numValid":   NUMVALID_VALS,
    "eps_ratio":  EPS_VALS,
}

# ── Scoring via score_pinyin_wer helpers ─────────────────────────────────────
from pypinyin import lazy_pinyin
from Levenshtein import distance as lev_dist
from collections import defaultdict


def build_params_dict(amp, cutoff0, cutoff1, numValid, eps_ratio):
    return {
        "onset": {
            "algorithm": "DBSCAN",
            "amp":       str(amp),
            "cutoff0":   str(cutoff0),
            "cutoff1":   str(cutoff1),
            "numValid":  str(numValid),
            "eps_ratio": str(eps_ratio),
            "min_speech": "0.2",
        },
        "offset": {
            "algorithm": "DBSCAN",
            "amp":       str(amp),
            "cutoff0":   str(cutoff0),
            "cutoff1":   str(cutoff1),
            "numValid":  str(numValid),
            "eps_ratio": str(eps_ratio),
            "min_speech": "0.2",
        },
    }


def combo_key(amp, cutoff0, cutoff1, numValid, eps_ratio):
    return f"a={amp}_c0={cutoff0}_c1={cutoff1}_nv={numValid}_eps={eps_ratio}"


# ── Scoring via score_pinyin_wer helpers ─────────────────────────────────────
def load_gt_intervals(gt_path):
    """Load intervals from a TextGrid."""
    import textgrid
    tg = textgrid.TextGrid.fromFile(str(gt_path))
    result = []
    for tier in tg.tiers:
        for iv in tier.intervals:
            if iv.mark and iv.mark.strip():
                result.append({
                    "start": iv.minTime,
                    "end":   iv.maxTime,
                    "text":  iv.mark.strip(),
                })
    return result


def load_out_intervals(out_path):
    """Load intervals from Praasper's output TextGrid."""
    import textgrid
    tg = textgrid.TextGrid.fromFile(str(out_path))
    result = []
    for tier in tg.tiers:
        for iv in tier.intervals:
            if iv.mark and iv.mark.strip():
                result.append({
                    "start": iv.minTime,
                    "end":   iv.maxTime,
                    "text":  iv.mark.strip(),
                })
    return result



def to_pinyin(text: str) -> list:
    return lazy_pinyin(list(text))


def intervals_overlap(a_start, a_end, b_start, b_end):
    return a_end > b_start and a_start < b_end


def find_connected_components(gt_intervals, out_intervals):
    """BFS on bipartite graph. Returns list of (gt_indices, out_indices, edges)."""
    n_gt  = len(gt_intervals)
    n_out = len(out_intervals)

    gt_adj  = defaultdict(list)
    out_adj = defaultdict(list)

    for gi, (gs, ge, _) in enumerate(gt_intervals):
        for oi, (os_, oe_, _) in enumerate(out_intervals):
            if oe_ > gs and os_ < ge:  # any intersection
                gt_adj[gi].append(oi)
                out_adj[oi].append(gi)

    visited_gt  = set()
    visited_out = set()
    components  = []

    for start_gi in range(n_gt):
        if start_gi in visited_gt:
            continue
        queue = [start_gi]
        component_gt  = []
        component_out = []
        while queue:
            gi = queue.pop(0)
            if gi in visited_gt:
                continue
            visited_gt.add(gi)
            component_gt.append(gi)
            for oi in gt_adj[gi]:
                if oi not in visited_out:
                    visited_out.add(oi)
                    component_out.append(oi)
                    for ngi in out_adj[oi]:
                        if ngi not in visited_gt:
                            queue.append(ngi)
        if component_gt or component_out:
            edges = [(gi, oi) for gi in component_gt for oi in gt_adj[gi] if oi in component_out]
            components.append({
                "gt_indices":  component_gt,
                "out_indices": component_out,
                "edges":       edges,
            })

    return components


def score_swer(gt_intervals, out_intervals):
    """
    Syllable-level WER via connected components.
    gt_intervals:  [{start, end, text}, ...]
    out_intervals: [{start, end, text}, ...]
    Returns (swer, total_errors, total_gt_syllables)
    """
    if not gt_intervals:
        return (0.0, 0, 0) if out_intervals else (0.0, 0, 0)
    if not out_intervals:
        total_syls = sum(len(to_pinyin(g["text"])) for g in gt_intervals)
        return (1.0, total_syls, total_syls)

    # Split GT syllables
    gt_syls = []  # [(start, end, pinyin)]
    for iv in gt_intervals:
        py = to_pinyin(iv["text"])
        dur = iv["end"] - iv["start"]
        step = dur / len(py) if len(py) > 0 else 0
        for k, p in enumerate(py):
            t_start = iv["start"] + k * step
            t_end   = t_start + step
            gt_syls.append((t_start, t_end, p))

    # Split output syllables
    out_syls = []  # [(start, end, pinyin)]
    for iv in out_intervals:
        py = to_pinyin(iv["text"])
        if not py:
            continue
        dur = iv["end"] - iv["start"]
        step = dur / len(py) if len(py) > 0 else 0
        for k, p in enumerate(py):
            t_start = iv["start"] + k * step
            t_end   = t_start + step
            out_syls.append((t_start, t_end, p))

    # Build connected components using interval overlap
    n_gt  = len(gt_syls)
    n_out = len(out_syls)

    gt_adj  = defaultdict(list)
    out_adj = defaultdict(list)

    for gi, (gs, ge, _) in enumerate(gt_syls):
        for oi, (os_, oe_, _) in enumerate(out_syls):
            if oe_ > gs and os_ < ge:
                gt_adj[gi].append(oi)
                out_adj[oi].append(gi)

    visited_gt  = set()
    visited_out = set()
    components  = []

    for start_gi in range(n_gt):
        if start_gi in visited_gt:
            continue
        queue = [start_gi]
        c_gt  = []
        c_out = []
        while queue:
            gi = queue.pop(0)
            if gi in visited_gt:
                continue
            visited_gt.add(gi)
            c_gt.append(gi)
            for oi in gt_adj[gi]:
                if oi not in visited_out:
                    visited_out.add(oi)
                    c_out.append(oi)
                    for ngi in out_adj[oi]:
                        if ngi not in visited_gt:
                            queue.append(ngi)
        if c_gt or c_out:
            edges = [(gi, oi) for gi in c_gt for oi in gt_adj[gi] if oi in c_out]
            components.append({"gt": c_gt, "out": c_out, "edges": edges})

    # Score each component
    total_errors = 0
    total_gt_syls = 0

    for comp in components:
        gt_indices  = comp["gt"]
        out_indices = comp["out"]

        gt_texts  = [gt_syls[i][2]  for i in gt_indices]
        out_texts = [out_syls[i][2] for i in out_indices]

        ed = lev_dist(gt_texts, out_texts)
        total_errors    += ed
        total_gt_syls   += len(gt_texts)

    # Uncovered GT syllables → WER = 1.0 for those
    uncovered = n_gt - total_gt_syls
    if uncovered > 0:
        total_errors  += uncovered
        total_gt_syls += uncovered

    swer = total_errors / total_gt_syls if total_gt_syls > 0 else 0.0
    return (swer, total_errors, total_gt_syls)


# ── Hit / Eff Rate ────────────────────────────────────────────────────────────

def compute_hit_eff(gt_intervals, out_intervals):
    """Compute Hit Rate and Effective Rate from raw interval lists.
    gt_intervals / out_intervals: [{start, end, text}, ...]
    hit_rate = overlap / gt_total  (recall — how much ground truth is captured)
    eff_rate = overlap / out_total (precision — how much VAD output is real speech)
    Returns (hit_rate, eff_rate) rounded to 4 decimals."""
    gt_ivs = [(iv["start"], iv["end"]) for iv in gt_intervals]
    out_ivs = [(iv["start"], iv["end"]) for iv in out_intervals]

    overlap = 0.0
    for os_, oe in out_ivs:
        for gs, ge in gt_ivs:
            overlap += max(0, min(oe, ge) - max(os_, gs))

    gt_total = sum(ge - gs for gs, ge in gt_ivs)
    out_total = sum(oe - os_ for os_, oe in out_ivs)

    eff = overlap / out_total if out_total > 0 else 0.0
    hit = overlap / gt_total if gt_total > 0 else 0.0
    return round(hit, 4), round(eff, 4)


# ── CSV helpers ───────────────────────────────────────────────────────────────

import csv as _csv
from threading import Lock as _Lock

_csv_lock = _Lock()

def _csv_ensure_header():
    """Write CSV header if file doesn't exist."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with _csv_lock:
            if not CSV_PATH.exists():  # double-check
                with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = _csv.writer(f)
                    writer.writerow(CSV_HEADER)

def append_csv_row(row: dict):
    """Append one result row to the master CSV. Thread-safe."""
    _csv_ensure_header()
    with _csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow([row.get(k, "") for k in CSV_HEADER])

def load_csv_completed():
    """Return set of (file_key, rep, combo_idx) already in the CSV.
    Only includes rows from the new 5-param grid (has valid amp)."""
    if not CSV_PATH.exists():
        return set()
    completed = set()
    try:
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                # Only count rows with valid amp (new 5-param grid)
                try:
                    float(row.get("amp", ""))
                except (ValueError, TypeError):
                    continue
                if row.get("error", "").strip() == "" and row.get("swer", "1.0") != "1.0":
                    try:
                        ci = int(row.get("combo_idx", "-1"))
                    except ValueError:
                        continue
                    completed.add((row.get("file_key", ""), row.get("rep", ""), ci))
    except Exception:
        pass
    return completed


# ── Main ───────────────────────────────────────────────────────────────────────

def run_combo(model, audio_path, gt_path, output_dir, combo_idx,
              amp, cutoff0, cutoff1, numValid, eps_ratio):
    """
    Run one combo. Symlink audio → temp dir → annote() → score.
    Returns dict with combo params and sWER.
    """
    params = build_params_dict(amp, cutoff0, cutoff1, numValid, eps_ratio)
    key    = combo_key(amp, cutoff0, cutoff1, numValid, eps_ratio)

    # Create temp dir with symlink to audio
    tmpdir = tempfile.mkdtemp(prefix="praasper_grid_")
    audio_link = Path(tmpdir) / audio_path.name
    try:
        audio_link.symlink_to(audio_path.resolve())
    except OSError:
        # Cross-filesystem or already exists — copy instead
        shutil.copy2(audio_path, audio_link)

    try:
        # Call annote — it writes to output/<tmpdir_name>/<fname>.TextGrid
        model.annote(
            input_path=tmpdir,
            params=params,
            seg_dur=10.0,
            min_pause=0.2,
            skip_existing=False,
            verbose=False,
        )

        # Find Praasper's output path
        tmpdir_name = Path(tmpdir).name                    # e.g. praasper_grid_xxxxx
        out_subdir  = Path(tmpdir) / "output" / tmpdir_name
        fname_stem  = audio_path.stem
        out_tg      = out_subdir / f"{fname_stem}.TextGrid"

        if not out_tg.exists():
            # Try fallback: maybe it wrote elsewhere
            # Praasper uses dir_name = os.path.dirname(os.path.dirname(wav_path))
            # Since wav_path = tmpdir/<name>.wav → dirname = tmpdir, dirname of that = tmpdir's parent
            parent_out = Path(tmpdir).parent / "output" / tmpdir_name / f"{fname_stem}.TextGrid"
            if parent_out.exists():
                out_tg = parent_out
            else:
                return {
                    "combo_idx": combo_idx,
                    "amp": amp, "cutoff0": cutoff0, "cutoff1": cutoff1,
                    "numValid": numValid, "eps_ratio": eps_ratio,
                    "key": key,
                    "swer": 1.0, "errors": -1, "gt_syllables": -1,
                    "num_intervals": -1,
                    "hit_rate": 0.0, "eff_rate": 0.0,
                    "error": f"Output TextGrid not found at {out_tg}",
                }

        # Load and score
        gt_ints  = load_gt_intervals(gt_path)
        out_ints = load_out_intervals(out_tg)

        swer, errors, gt_syls = score_swer(gt_ints, out_ints)
        hit_rate, eff_rate = compute_hit_eff(gt_ints, out_ints)

        # Save output TextGrid to results dir
        result_tg = output_dir / f"combo_{combo_idx:05d}.TextGrid"
        shutil.copy2(out_tg, result_tg)

        return {
            "combo_idx":   combo_idx,
            "amp": amp, "cutoff0": cutoff0, "cutoff1": cutoff1,
            "numValid": numValid, "eps_ratio": eps_ratio,
            "key": key,
            "swer":        round(swer, 6),
            "errors":      errors,
            "gt_syllables": gt_syls,
            "num_intervals": len(out_ints),
            "hit_rate":    hit_rate,
            "eff_rate":    eff_rate,
            "error": None,
        }

    except Exception as e:
        return {
            "combo_idx": combo_idx,
            "amp": amp, "cutoff0": cutoff0, "cutoff1": cutoff1,
            "numValid": numValid, "eps_ratio": eps_ratio,
            "key": key,
            "swer": 1.0, "errors": -1, "gt_syllables": -1,
            "num_intervals": -1,
            "hit_rate": 0.0, "eff_rate": 0.0,
            "error": str(e),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_grid(audio_key, audio_path, gt_path, results_dir, limit=0, parallel=1, model=None, rep="", resume=True):
    """Run grid for one audio file. Pass model= to reuse an already-loaded model.
    
    Resume logic: if output dir already has results_ranked.json and combo_index.json,
    skip combos that already have a valid (non-error) result. Use --force to re-run all.
    """
    from praasper import init_model

    print(f"\n{'='*60}", flush=True)
    print(f"Audio: {audio_key} — {audio_path}", flush=True)

    if model is None:
        # Load model fresh
        print("Loading Praasper model...", flush=True)
        t_load = time.time()
        model = init_model(ASR="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda:0")
        print(f"Model loaded in {time.time()-t_load:.1f}s", flush=True)
    else:
        print("(reusing pre-loaded model)", flush=True)

    # Build combo list — stable order so combo_idx is deterministic
    combos = list(product(
        GRID["amp"],
        GRID["cutoff0"],
        GRID["cutoff1"],
        GRID["numValid"],
        GRID["eps_ratio"],
    ))
    if limit > 0:
        combos = combos[:limit]
    n = len(combos)
    print(f"Grid: {n} combos ({len(AMP_VALS)}×{len(CUTOFF0_VALS)}×{len(CUTOFF1_VALS)}×{len(NUMVALID_VALS)}×{len(EPS_VALS)})", flush=True)

    output_subdir = results_dir / f"{audio_key}{rep}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    rep_display = rep if rep else "(none)"

    # ── Resume: load completed combos from CSV ────────────────────────────────
    done_ids = set()
    if resume:
        csv_completed = load_csv_completed()
        for fk, r, ci in csv_completed:
            if fk == audio_key and r == rep_display:
                done_ids.add(ci)
        if done_ids:
            print(f"Resume: found {len(done_ids)}/{n} combos already in CSV", flush=True)

    # ── Build stable combo_index (for printing best-at-end only) ──────────────
    combo_index = {}
    for i, (a, c0, c1, nv, eps) in enumerate(combos):
        combo_index[i] = {
            "amp": a, "cutoff0": c0, "cutoff1": c1,
            "numValid": nv, "eps_ratio": eps,
        }

    t0 = time.time()
    newly_done = 0
    all_results = []  # keep in memory for final ranking
    results_lock = __import__('threading').Lock()

    def process_combo(combo_item):
        i, (a, c0, c1, nv, eps) = combo_item
        r = run_combo(
            model=model,
            audio_path=audio_path,
            gt_path=gt_path,
            output_dir=output_subdir,
            combo_idx=i,
            amp=a, cutoff0=c0, cutoff1=c1, numValid=nv, eps_ratio=eps,
        )
        with results_lock:
            all_results.append(r)
            csv_row = {**r, "file_key": audio_key, "rep": rep_display}
            append_csv_row(csv_row)
        return r

    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pending = [(i, c) for i, c in enumerate(combos) if i not in done_ids]
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(process_combo, item): item for item in pending}
            for future in as_completed(futures):
                future.result()
                newly_done += 1
                elapsed = time.time() - t0
                done_count = len(done_ids) + newly_done
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (n - done_count) / rate if rate > 0 else 0
                print(f"  {done_count}/{n} ({100*done_count/n:.0f}%) — {rate:.1f} combo/s — "
                      f"ETA {eta/60:.1f} min — parallel={parallel}", flush=True)
    else:
        for i, (a, c0, c1, nv, eps) in enumerate(combos):
            if i in done_ids:
                continue

            r = run_combo(
                model=model,
                audio_path=audio_path,
                gt_path=gt_path,
                output_dir=output_subdir,
                combo_idx=i,
                amp=a, cutoff0=c0, cutoff1=c1, numValid=nv, eps_ratio=eps,
            )
            all_results.append(r)
            newly_done += 1

            # Append to master CSV immediately (checkpoint per combo)
            csv_row = {**r, "file_key": audio_key, "rep": rep_display}
            append_csv_row(csv_row)

            if (i + 1) % 5 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                done_count = len(done_ids) + newly_done
                rate    = done_count / elapsed if elapsed > 0 else 0
                eta     = (n - done_count) / rate if rate > 0 else 0
                print(f"  {done_count}/{n} ({100*done_count/n:.0f}%) — {rate:.1f} combo/s — "
                      f"ETA {eta/60:.1f} min — last: {r['key']} sWER={r['swer']:.4f}", flush=True)

    elapsed = time.time() - t0
    if newly_done == 0 and len(all_results) == 0:
        print(f"\nAll {n} combos already in CSV. Skipped.", flush=True)
    else:
        print(f"\nDone: {newly_done} new combos in {elapsed:.0f}s ({newly_done/elapsed:.1f} combo/s)", flush=True)

    # ── Load full results from CSV for ranking ────────────────────────────────
    all_results_csv = []
    try:
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get("file_key") == audio_key and row.get("rep") == rep_display:
                    # Only include rows with valid amp (new 5-param grid)
                    try:
                        float(row.get("amp", ""))
                        all_results_csv.append(row)
                    except (ValueError, TypeError):
                        pass  # skip old 4-param rows
    except Exception:
        pass

    if all_results_csv:
        results_sorted = sorted(all_results_csv, key=lambda x: float(x.get("swer", 1.0)))
        best = results_sorted[0]
        print(f"Best:  a={best.get('amp','')}_c0={best.get('cutoff0','')}_c1={best.get('cutoff1','')}_"\
              f"nv={best.get('numValid','')}_eps={best.get('eps_ratio','')}  "\
              f"sWER={best.get('swer','')}  ({best.get('errors','')}/{best.get('gt_syllables','')} errors)", flush=True)
        print(f"Results → {CSV_PATH}  ({len(results_sorted)} combos for {audio_key}{rep})", flush=True)
        return results_sorted

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Praasper VAD+ASR grid search")
    parser.add_argument("audio_keys", nargs="*",
                        help="Audio keys to process, e.g. 01-1 01-2 01-3. "
                             "Omit or use --all to process all files.")
    parser.add_argument("--all", action="store_true",
                        help="Run all audio files (model loaded once)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit combos per file (0 = full grid)")
    parser.add_argument("--eps", type=str, default="",
                        help="Comma-separated eps values to sweep, e.g. '0.01,0.015,0.02'. Overrides EPS_VALS.")
    parser.add_argument("--amp", type=str, default="",
                        help="Comma-separated amp values to sweep, e.g. '1.01,1.02,1.03'. Overrides AMP_VALS.")
    parser.add_argument("--rep", type=str, default="",
                        help="Suffix appended to result subdir name, e.g. '_r1' → results/01-1_r1/")
    parser.add_argument("--resume-skip", action="store_true",
                        help="Skip combos that already have a valid result (resume mode)")
    parser.add_argument("--audio-dir", type=str, default=str(AUDIO_DIR),
                        help="Path to audio files directory")
    parser.add_argument("--answers-dir", type=str, default=str(ANSWERS_DIR),
                        help="Path to ground truth TextGrid files")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="Path to results output directory")
    parser.add_argument("--modelscope-cache", type=str, default=MODELSCOPE_CACHE,
                        help="Path to ModelScope cache directory")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel combo workers (default: 1)")
    args = parser.parse_args()

    # Apply CLI overrides to path variables
    AUDIO_DIR = Path(args.audio_dir)
    ANSWERS_DIR = Path(args.answers_dir)
    RESULTS_DIR = Path(args.results_dir)
    CSV_PATH = RESULTS_DIR / "results_all.csv"
    MODELSCOPE_CACHE = args.modelscope_cache

    # Override EPS_VALS if --eps given
    if args.eps:
        EPS_VALS[:] = [float(x.strip()) for x in args.eps.split(",")]
        GRID["eps_ratio"] = EPS_VALS
        print(f"EPS_VALS overridden: {EPS_VALS}", flush=True)

    # Override AMP_VALS if --amp given
    if args.amp:
        AMP_VALS[:] = [float(x.strip()) for x in args.amp.split(",")]
        GRID["amp"] = AMP_VALS
        print(f"AMP_VALS overridden: {AMP_VALS}", flush=True)

    os.environ["modelscope_cache"] = MODELSCOPE_CACHE
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    sys.path.insert(0, str(PRAASPER_REPO))

    # Monkey-patch FunASR audio loader: use soundfile instead of ffmpeg
    import numpy as np
    import soundfile as sf
    def _load_audio_soundfile(file, sr=16000):
        data, file_sr = sf.read(file, dtype='float32')
        if len(data.shape) > 1:
            data = data.mean(axis=1)  # mono
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        return data

    from funasr.utils import load_utils
    load_utils._load_audio_ffmpeg = _load_audio_soundfile

    from praasper import init_model

    # Resolve file list
    if args.all:
        files = []
        for wav_path in sorted(AUDIO_DIR.glob("*.wav")):
            key     = wav_path.stem
            gt_path = ANSWERS_DIR / f"{key}a.TextGrid"
            if not gt_path.exists():
                print(f"SKIP {key}: no GT TextGrid", flush=True)
                continue
            files.append((key, wav_path, gt_path))
    elif args.audio_keys:
        files = []
        for key in args.audio_keys:
            wav_path = AUDIO_DIR / f"{key}.wav"
            gt_path  = ANSWERS_DIR / f"{key}a.TextGrid"
            if not wav_path.exists():
                print(f"SKIP {key}: WAV not found at {wav_path}", flush=True)
                continue
            if not gt_path.exists():
                print(f"SKIP {key}: no GT TextGrid at {gt_path}", flush=True)
                continue
            files.append((key, wav_path, gt_path))
    else:
        print("ERROR: pass audio keys (e.g. 01-1 01-2) or use --all", flush=True)
        sys.exit(1)

    if not files:
        print("No valid audio files found.", flush=True)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Batch: {len(files)} file(s) — model loaded once")
    print(f"Grid: {len(AMP_VALS)}×{len(CUTOFF0_VALS)}×{len(CUTOFF1_VALS)}×{len(NUMVALID_VALS)}×{len(EPS_VALS)} "
          f"= {len(AMP_VALS)*len(CUTOFF0_VALS)*len(CUTOFF1_VALS)*len(NUMVALID_VALS)*len(EPS_VALS)} combos/file")
    print(f"{'='*60}\n", flush=True)

    # Load model ONCE
    print("Loading Praasper model...", flush=True)
    t_batch = time.time()
    model = init_model(ASR="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda:0")
    print(f"Model loaded in {time.time()-t_batch:.1f}s\n", flush=True)

    for i, (key, wav_path, gt_path) in enumerate(files):
        t_file = time.time()
        print(f"[{i+1}/{len(files)}] Starting {key}...", flush=True)
        run_grid(key, wav_path, gt_path, RESULTS_DIR,
                 limit=args.limit, parallel=args.parallel, model=model, rep=args.rep,
                 resume=args.resume_skip)
        print(f"[{i+1}/{len(files)}] {key} done in {time.time()-t_file:.0f}s\n", flush=True)

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(files)} files in {time.time()-t_batch:.0f}s")
    print(f"{'='*60}", flush=True)

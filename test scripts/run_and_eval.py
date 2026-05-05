#!/usr/bin/env python3
"""
Run latest Praasper (no grid) + evaluate with pinyin sWER.

Usage:
    # Single file
    python run_and_eval.py --audio 01-1

    # Multiple files
    python run_and_eval.py --audio 01-1 01-2 01-3

    # All files in audio dir
    python run_and_eval.py --all

    # Custom paths
    python run_and_eval.py --all \
        --audio-dir /mnt/e/Corpus/ma/audio \
        --answers-dir /mnt/e/ProjLegacy/Test_All_Models_for_Praasper/answers \
        --output-dir /mnt/e/Corpus/ma/output/audio
"""

import os
import sys
import gc
import argparse
import csv
from pathlib import Path
from collections import defaultdict

# ── Monkey-patch FunASR ffmpeg loader → soundfile ──────────────────────────────
import soundfile as sf
import librosa
import torch

def _load_audio_soundfile(file, sr=16000):
    data, file_sr = sf.read(file, dtype='float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data

from funasr.utils import load_utils
load_utils._load_audio_ffmpeg = _load_audio_soundfile

# ── Import Praasper ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from praasper import init_model

# ── Scoring deps ───────────────────────────────────────────────────────────────
try:
    from pypinyin import lazy_pinyin
except ImportError:
    print("ERROR: pypinyin not installed. Run: pip install pypinyin")
    sys.exit(1)

try:
    from textgrid import TextGrid
except ImportError:
    print("ERROR: textgrid not installed. Run: pip install textgrid")
    sys.exit(1)

try:
    from Levenshtein import distance as lev_dist
except ImportError:
    def lev_dist(a, b):
        m, n = len(a), len(b)
        if m < n:
            return lev_dist(b, a)
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

# ── Paths ──────────────────────────────────────────────────────────────────────
AUDIO_DIR   = Path("/mnt/e/Corpus/ma/audio")
ANSWERS_DIR = Path("/mnt/e/ProjLegacy/Test_All_Models_for_Praasper/answers")
OUTPUT_DIR  = Path("/mnt/e/Corpus/ma/output/audio")

# ── Scoring functions (from grid_new_praasper.py) ──────────────────────────────
def to_pinyin(text: str) -> list:
    return lazy_pinyin(list(text))


def tg_to_intervals(tg_path: str):
    tg = TextGrid.fromFile(tg_path)
    tier = tg[0] if tg else None
    if not tier:
        return []
    return [
        {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark.strip()}
        for iv in tier.intervals
        if iv.mark.strip()
    ]


def compute_hit_eff(gt_ivs, out_ivs):
    overlap = 0.0
    for o in out_ivs:
        for g in gt_ivs:
            s = max(o["start"], g["start"])
            e = min(o["end"], g["end"])
            if e > s:
                overlap += e - s

    gt_total = sum(g["end"] - g["start"] for g in gt_ivs)
    out_total = sum(o["end"] - o["start"] for o in out_ivs)

    hit = overlap / gt_total if gt_total > 0 else 0.0
    eff = overlap / out_total if out_total > 0 else 0.0
    return round(hit, 4), round(eff, 4)


def score_swer(gt_ivs, out_ivs):
    if not gt_ivs:
        return (0.0, 0, 0)
    if not out_ivs:
        total_syls = sum(len(to_pinyin(g["text"])) for g in gt_ivs)
        return (1.0, total_syls, total_syls)

    gt_syls = []
    for iv in gt_ivs:
        py = to_pinyin(iv["text"])
        dur = iv["end"] - iv["start"]
        step = dur / len(py) if len(py) > 0 else 0
        for k, p in enumerate(py):
            t0 = iv["start"] + k * step
            t1 = t0 + step
            gt_syls.append((t0, t1, p))

    out_syls = []
    for iv in out_ivs:
        py = to_pinyin(iv["text"])
        if not py:
            continue
        dur = iv["end"] - iv["start"]
        step = dur / len(py) if len(py) > 0 else 0
        for k, p in enumerate(py):
            t0 = iv["start"] + k * step
            t1 = t0 + step
            out_syls.append((t0, t1, p))

    n_gt = len(gt_syls)
    n_out = len(out_syls)

    gt_adj = defaultdict(list)
    out_adj = defaultdict(list)
    for gi, (gs, ge, _) in enumerate(gt_syls):
        for oi, (os_, oe_, _) in enumerate(out_syls):
            if oe_ > gs and os_ < ge:
                gt_adj[gi].append(oi)
                out_adj[oi].append(gi)

    visited_gt = set()
    visited_out = set()
    components = []

    for start_gi in range(n_gt):
        if start_gi in visited_gt:
            continue
        queue = [start_gi]
        c_gt = []
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
            components.append({"gt": c_gt, "out": c_out})

    total_errors = 0
    total_gt_syls = 0
    for comp in components:
        gt_texts = [gt_syls[i][2] for i in comp["gt"]]
        out_texts = [out_syls[i][2] for i in comp["out"]]
        ed = lev_dist(gt_texts, out_texts)
        total_errors += ed
        total_gt_syls += len(gt_texts)

    uncovered = n_gt - total_gt_syls
    if uncovered > 0:
        total_errors += uncovered
        total_gt_syls += uncovered

    swer = total_errors / total_gt_syls if total_gt_syls > 0 else 0.0
    return swer, total_errors, total_gt_syls


def evaluate_file(gt_path, out_path):
    if not os.path.exists(gt_path) or not os.path.exists(out_path):
        return None
    gt_ivs = tg_to_intervals(gt_path)
    out_ivs = tg_to_intervals(out_path)
    hit, eff = compute_hit_eff(gt_ivs, out_ivs)
    swer, errors, gt_syls = score_swer(gt_ivs, out_ivs)
    return {
        "hit": hit,
        "eff": eff,
        "swer": swer,
        "errors": errors,
        "gt_syls": gt_syls,
        "n_intv": len(out_ivs),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run Praasper + evaluate (no grid)")
    parser.add_argument("--audio", nargs="+", help="File key(s) e.g. 01-1 01-2")
    parser.add_argument("--all", action="store_true", help="Process all .wav in audio dir")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--answers-dir", type=Path, default=ANSWERS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--csv", default="results_run_and_eval.csv")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output TextGrid exists")
    parser.add_argument("--params", default=None, help="Path to VAD params .txt file (default=auto_vad)")
    args = parser.parse_args()

    if args.all:
        files = sorted([f.stem for f in args.audio_dir.glob("*.wav")])
    elif args.audio:
        files = args.audio
    else:
        print("ERROR: provide --audio FILE [FILE ...] or --all")
        sys.exit(1)

    print(f"Files to process: {len(files)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_key", "hit_rate", "eff_rate", "sWER", "errors", "gt_syllables", "num_intervals"])

    model = init_model()

    results = []
    for i, fk in enumerate(files):
        wav_path = args.audio_dir / f"{fk}.wav"
        gt_path  = args.answers_dir / f"{fk}a.TextGrid"
        out_path = args.output_dir / f"{fk}.TextGrid"

        print(f"\n[{i+1}/{len(files)}] {fk}")

        if not wav_path.exists():
            print(f"  SKIP: wav not found {wav_path}")
            continue

        if args.skip_existing and out_path.exists():
            print(f"  SKIP: output exists {out_path}")
        else:
            print(f"  Running Praasper...", flush=True)
            model.annote(input_path=str(wav_path), verbose=False, skip_existing=False, params=args.params)
            print(f"  Done.", flush=True)

        # Evaluate
        res = evaluate_file(str(gt_path), str(out_path))
        if res is None:
            print(f"  EVAL SKIP: gt={gt_path.exists()} out={out_path.exists()}")
            continue

        print(f"  hit={res['hit']:.4f} eff={res['eff']:.4f} sWER={res['swer']:.4f} "
              f"err={res['errors']}/{res['gt_syls']} #intval={res['n_intv']}")

        results.append(res)
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([fk, f"{res['hit']:.6f}", f"{res['eff']:.6f}",
                             f"{res['swer']:.6f}", res["errors"], res["gt_syls"], res["n_intv"]])

        gc.collect()

    if results:
        n = len(results)
        print(f"\n=== SUMMARY ({n} files) ===")
        print(f"  hit  = {sum(r['hit'] for r in results)/n:.4f}")
        print(f"  eff  = {sum(r['eff'] for r in results)/n:.4f}")
        print(f"  sWER = {sum(r['swer'] for r in results)/n:.4f}")
        print(f"  CSV: {args.csv}")

    model.release_resources()


if __name__ == "__main__":
    main()

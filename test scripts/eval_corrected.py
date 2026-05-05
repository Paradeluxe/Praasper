#!/usr/bin/env python3
"""Corrected evaluation for Praasper output vs ground truth.

Fixes over old eval:
  - sWER = syllable-level pinyin WER (not raw char SequenceMatcher)
  - Syllables split by temporal overlap into connected components
  - Levenshtein distance per component
  - Uncovered GT syllables count as errors

Usage:
    python eval_corrected.py --output-dir /path/to/output --answers-dir /path/to/answers
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

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
    # Fallback Levenshtein for list inputs
    def lev_dist(a, b):
        """Levenshtein distance for two lists."""
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


def to_pinyin(text: str) -> list:
    """Convert text to pinyin syllable list."""
    return lazy_pinyin(list(text))


def tg_to_intervals(tg_path: str):
    """Load TextGrid and return list of {start, end, text} for non-empty intervals."""
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
    """Compute hit rate and effective rate from interval lists."""
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
    return hit, eff


def score_swer(gt_ivs, out_ivs):
    """
    Syllable-level pinyin WER with temporal connected components.
    Returns (swer, errors, gt_syllables).
    """
    if not gt_ivs:
        return (0.0, 0, 0)
    if not out_ivs:
        total_syls = sum(len(to_pinyin(g["text"])) for g in gt_ivs)
        return (1.0, total_syls, total_syls)

    # Split GT into syllables with temporal slices
    gt_syls = []
    for iv in gt_ivs:
        py = to_pinyin(iv["text"])
        dur = iv["end"] - iv["start"]
        step = dur / len(py) if len(py) > 0 else 0
        for k, p in enumerate(py):
            t0 = iv["start"] + k * step
            t1 = t0 + step
            gt_syls.append((t0, t1, p))

    # Split output into syllables with temporal slices
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

    # Build adjacency by temporal overlap
    gt_adj = defaultdict(list)
    out_adj = defaultdict(list)
    for gi, (gs, ge, _) in enumerate(gt_syls):
        for oi, (os_, oe_, _) in enumerate(out_syls):
            if oe_ > gs and os_ < ge:
                gt_adj[gi].append(oi)
                out_adj[oi].append(gi)

    # Find connected components via BFS
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

    # Score each component
    total_errors = 0
    total_gt_syls = 0
    for comp in components:
        gt_texts = [gt_syls[i][2] for i in comp["gt"]]
        out_texts = [out_syls[i][2] for i in comp["out"]]
        ed = lev_dist(gt_texts, out_texts)
        total_errors += ed
        total_gt_syls += len(gt_texts)

    # Uncovered GT syllables
    uncovered = n_gt - total_gt_syls
    if uncovered > 0:
        total_errors += uncovered
        total_gt_syls += uncovered

    swer = total_errors / total_gt_syls if total_gt_syls > 0 else 0.0
    return swer, total_errors, total_gt_syls


def evaluate_file(gt_path, out_path):
    """Evaluate one file. Returns dict or None."""
    if not os.path.exists(gt_path):
        return None
    if not os.path.exists(out_path):
        return None

    gt_ivs = tg_to_intervals(gt_path)
    out_ivs = tg_to_intervals(out_path)

    hit, eff = compute_hit_eff(gt_ivs, out_ivs)
    swer, errors, gt_syls = score_swer(gt_ivs, out_ivs)
    n_intv = len(out_ivs)

    return {
        "hit": hit,
        "eff": eff,
        "swer": swer,
        "errors": errors,
        "gt_syls": gt_syls,
        "n_intv": n_intv,
    }


def main():
    parser = argparse.ArgumentParser(description="Corrected Praasper evaluation")
    parser.add_argument("--output-dir", required=True, help="Directory with Praasper output .TextGrid files")
    parser.add_argument("--answers-dir", required=True, help="Directory with ground truth *a.TextGrid files")
    parser.add_argument("--csv", default="results_eval_corrected.csv", help="Output CSV path")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary, skip per-file CSV")
    args = parser.parse_args()

    out_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith(".TextGrid")])
    print(f"Found {len(out_files)} output files")

    # Write CSV header
    if not args.summary_only:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_key", "hit_rate", "eff_rate", "sWER", "errors", "gt_syllables", "num_intervals"])

    results = []
    for i, fname in enumerate(out_files):
        fk = fname.replace(".TextGrid", "")
        gt_name = f"{fk}a.TextGrid"
        gt_path = os.path.join(args.answers_dir, gt_name)
        out_path = os.path.join(args.output_dir, fname)

        print(f"\n[{i+1}/{len(out_files)}] {fk}", flush=True)
        res = evaluate_file(gt_path, out_path)
        if res is None:
            print(f"  SKIP: gt={os.path.exists(gt_path)} out={os.path.exists(out_path)}")
            continue

        print(f"  hit={res['hit']:.4f} eff={res['eff']:.4f} sWER={res['swer']:.4f} "
              f"err={res['errors']}/{res['gt_syls']} #intval={res['n_intv']}")

        results.append(res)
        if not args.summary_only:
            with open(args.csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([fk, f"{res['hit']:.6f}", f"{res['eff']:.6f}",
                                 f"{res['swer']:.6f}", res["errors"], res["gt_syls"], res["n_intv"]])

    if results:
        n = len(results)
        avg_hit = sum(r["hit"] for r in results) / n
        avg_eff = sum(r["eff"] for r in results) / n
        avg_swer = sum(r["swer"] for r in results) / n
        print(f"\n=== SUMMARY ({n} files) ===")
        print(f"  hit = {avg_hit:.4f}")
        print(f"  eff = {avg_eff:.4f}")
        print(f"  sWER = {avg_swer:.4f}")
        if not args.summary_only:
            print(f"\n  CSV saved to {args.csv}")
    else:
        print("\nNo results.")


if __name__ == "__main__":
    main()

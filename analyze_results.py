#!/usr/bin/env python3
"""Analyze Praasper grid search results from one or more CSV files.

Loads all valid results, aggregates by combo, and ranks top-N.
Usage:
    python analyze_results.py                          # default dirs
    python analyze_results.py results/results_all.csv  # specific files
    python analyze_results.py results/results_all.csv results_new/results_all.csv
    python analyze_results.py --top 20 --sort eff      # sort by eff, show top 20
"""
import sys
import os
import csv
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')

DEFAULT_CSVS = [
    'results/results_all.csv',
]


def load_csvs(paths):
    """Load all CSVs, return combined list of valid rows."""
    all_rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  SKIP (not found): {p}")
            continue
        with open(p, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only valid results (no error)
                err = row.get('error', '').strip()
                if not err:
                    row['_source'] = os.path.dirname(p)
                    all_rows.append(row)
    return all_rows


def aggregate(rows):
    """Aggregate by (amp, c0, c1, nv, eps) combo. Returns ranked list."""
    combos = defaultdict(lambda: {'swer': 0, 'hit': 0, 'eff': 0, 'intv': 0, 'count': 0, 'files': set()})
    for row in rows:
        try:
            key = (row['amp'], row['cutoff0'], row['cutoff1'], row['numValid'], row['eps_ratio'])
            c = combos[key]
            c['swer'] += float(row.get('swer', 0))
            c['hit'] += float(row.get('hit_rate', 0))
            c['eff'] += float(row.get('eff_rate', 0))
            c['intv'] += float(row.get('num_intervals', 0))
            c['count'] += 1
            c['files'].add(row['file_key'])
        except (ValueError, KeyError):
            continue

    n = len(combos)
    ranked = []
    for k, v in combos.items():
        cnt = v['count']
        ranked.append({
            'amp': float(k[0]), 'c0': float(k[1]), 'c1': float(k[2]), 'nv': float(k[3]), 'eps': float(k[4]),
            'hit': v['hit'] / cnt,
            'eff': v['eff'] / cnt,
            'swer': v['swer'] / cnt,
            'avg_intv': v['intv'] / cnt,
            'count': cnt,
        })
    return ranked, n


def sort_ranked(ranked, sort_key='hit'):
    """Sort by priority. Default: hit desc, eff desc, sWER asc, intv asc."""
    if sort_key == 'hit':
        return sorted(ranked, key=lambda x: (-x['hit'], -x['eff'], x['swer'], x['avg_intv']))
    elif sort_key == 'eff':
        return sorted(ranked, key=lambda x: (-x['eff'], -x['hit'], x['swer'], x['avg_intv']))
    elif sort_key == 'swer':
        return sorted(ranked, key=lambda x: (x['swer'], -x['hit'], -x['eff'], x['avg_intv']))
    elif sort_key == 'intv':
        return sorted(ranked, key=lambda x: (x['avg_intv'], -x['hit'], -x['eff'], x['swer']))
    return ranked


def print_table(ranked, top_n=10, sort_key='hit'):
    ranked = sort_ranked(ranked, sort_key)
    print(f"\n{'='*110}")
    print(f"  TOP {min(top_n, len(ranked))} COMBOS  (sort: {sort_key}, {len(ranked)} total combos)")
    print(f"{'='*110}")
    print(f"  {'Rank':>4} | {'amp':>5} | {'c0':>5} | {'c1':>6} | {'nv':>6} | {'eps':>5} | {'hit':>7} | {'eff':>7} | {'sWER':>6} | {'avg intv':>8} | {'files':>5}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*5}")
    for i, r in enumerate(ranked[:top_n], 1):
        print(f"  {i:>4} | {r['amp']:>5.2f} | {r['c0']:>5.0f} | {r['c1']:>6.0f} | {r['nv']:>6.0f} | {r['eps']:>5.2f} | {r['hit']:>7.4f} | {r['eff']:>7.4f} | {r['swer']:>6.4f} | {r['avg_intv']:>8.0f} | {r['count']:>5}")
    print(f"{'='*110}\n")


def print_per_file(rows, sort_key='hit'):
    """Print per-file summary."""
    files = sorted(set(r['file_key'] for r in rows))
    print(f"\n{'─'*70}")
    print(f"  PER-FILE SUMMARY  ({len(files)} files)")
    print(f"{'─'*70}")
    for fk in files:
        frows = [r for r in rows if r['file_key'] == fk]
        n = len(frows)
        if n == 0:
            continue
        avg_hit = sum(float(r['hit_rate']) for r in frows) / n
        avg_eff = sum(float(r['eff_rate']) for r in frows) / n
        avg_swer = sum(float(r['swer']) for r in frows) / n
        errs = sum(1 for r in frows if r.get('error', '').strip())
        src = frows[0].get('_source', '?')
        print(f"  {fk:>6} | hit={avg_hit:.4f} eff={avg_eff:.4f} sWER={avg_swer:.4f} | {n} combos | {errs} errs | {src}")
    print(f"{'─'*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze Praasper grid results')
    parser.add_argument('csvs', nargs='*', help='CSV file paths (default: auto-discover)')
    parser.add_argument('--top', type=int, default=10, help='Top N combos to show')
    parser.add_argument('--sort', choices=['hit', 'eff', 'swer', 'intv'], default='hit', help='Sort priority')
    parser.add_argument('--all', action='store_true', help='Show all combos, not just top N')
    parser.add_argument('--per-file', action='store_true', help='Show per-file summary')
    args = parser.parse_args()

    # Auto-discover CSVs if none given
    if not args.csvs:
        csvs = []
        for d in ['results', 'results_new', 'results_test_0102']:
            p = os.path.join('/mnt/d/hermes_playground/Praasper', d, 'results_all.csv')
            if os.path.exists(p):
                csvs.append(p)
        if not csvs:
            print("No CSVs found. Run a grid search first.")
            sys.exit(1)
    else:
        csvs = args.csvs

    print(f"\nLoading {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"  {c}")

    rows = load_csvs(csvs)
    print(f"Total valid rows: {len(rows)}")

    if not rows:
        print("No valid results found.")
        sys.exit(1)

    ranked, n_combos = aggregate(rows)

    top_n = len(ranked) if args.all else args.top
    print_table(ranked, top_n=top_n, sort_key=args.sort)

    if args.per_file:
        print_per_file(rows, sort_key=args.sort)


if __name__ == '__main__':
    main()

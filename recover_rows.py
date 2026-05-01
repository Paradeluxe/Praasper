#!/usr/bin/env python3
"""Extract matching rows from old backup CSV into current CSV for --resume-skip."""
import csv
from itertools import product

# Current grid values
AMP = [1.01, 1.1, 1.2]
CUTOFF0 = [0, 200]
CUTOFF1 = [3600, 5400, 12600]
NUMVALID = [1000, 5000, 10000]
EPS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]

valid_combos = set()
for a, c0, c1, nv, e in product(AMP, CUTOFF0, CUTOFF1, NUMVALID, EPS):
    valid_combos.add((float(a), int(c0), int(c1), int(nv), float(e)))

backup = "results/results_all_backup_20260425_132716.csv"
current = "results/results_all.csv"
header = None

with open(current, "r") as f:
    header = f.readline().strip()

matches = []
with open(backup, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            key = (float(row["amp"]), int(row["cutoff0"]), int(row["cutoff1"]), 
                   int(row["numValid"]), float(row["eps_ratio"]))
        except (ValueError, KeyError):
            continue
        if key in valid_combos and row.get("error", "").strip() == "" and row.get("swer", "") != "":
            matches.append(row)

with open(current, "w", newline="") as f:
    f.write(header + "\n")
    writer = csv.DictWriter(f, fieldnames=header.split(","))
    for row in matches:
        writer.writerow(row)

print(f"Saved {len(matches)} matching rows from backup into current CSV")

#!/usr/bin/env python3
"""Cross-check eval_corrected.py vs grid_new_praasper.py scoring on fresh 01-1."""
import sys
sys.path.insert(0, '/mnt/e/praasper/test scripts')

from textgrid import TextGrid
from grid_new_praasper import to_pinyin, score_swer, compute_hit_eff

GT_PATH  = '/mnt/e/ProjLegacy/Test_All_Models_for_Praasper/answers/01-1a.TextGrid'
OUT_PATH = '/mnt/e/Corpus/ma/output/audio/01-1.TextGrid'

def tg_to_ivs(path):
    tg = TextGrid.fromFile(path)
    tier = tg[0]
    return [
        {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark.strip()}
        for iv in tier.intervals
        if iv.mark.strip()
    ]

gt_ivs  = tg_to_ivs(GT_PATH)
out_ivs = tg_to_ivs(OUT_PATH)

hit, eff = compute_hit_eff(gt_ivs, out_ivs)
swer, errors, gt_syls = score_swer(gt_ivs, out_ivs)

print(f"grid_new_praasper scoring on 01-1:")
print(f"  hit={hit:.4f} eff={eff:.4f} sWER={swer:.4f} err={errors}/{gt_syls} #intval={len(out_ivs)}")

import os
import sys
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')
from textgrid import TextGrid

def levenshtein(s1, s2):
    if len(s1) < len(s2): return levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
            prev_row = curr_row
    return prev_row[-1]

def calculate_metrics(gt_path, out_path):
    try:
        gt = TextGrid.fromFile(gt_path)
        out = TextGrid.fromFile(out_path)
    except Exception as e:
        return None

    gt_tier = gt[0] if gt else None
    out_tier = out[0] if out else None
    
    if not gt_tier or not out_tier: return None

    gt_intervals = gt_tier.intervals
    out_intervals = out_tier.intervals

    gt_time = sum(i.maxTime - i.minTime for i in gt_intervals if i.mark.strip())
    out_time = sum(i.maxTime - i.minTime for i in out_intervals if i.mark.strip())
    
    overlap_time = 0
    for g in gt_intervals:
        if not g.mark.strip(): continue
        for o in out_intervals:
            if not o.mark.strip(): continue
            start = max(g.minTime, o.minTime)
            end = min(g.maxTime, o.maxTime)
            if end > start:
                overlap_time += (end - start)
    
    gt_text = "".join(i.mark.strip() for i in gt_intervals)
    out_text = "".join(i.mark.strip() for i in out_intervals)
    
    import difflib
    ratio = difflib.SequenceMatcher(None, gt_text, out_text).ratio()
    sWER = 1 - ratio
    
    hit_rate = overlap_time / gt_time if gt_time > 0 else 0
    eff_rate = overlap_time / out_time if out_time > 0 else 0
    
    return {'hit': hit_rate, 'eff': eff_rate, 'swer': sWER}

output_dir = '/mnt/d/output/audio_data'
answers_dir = '/mnt/d/audio_data/answers'

files = sorted([f for f in os.listdir(output_dir) if f.endswith('.TextGrid')])
results = []

for f in files:
    key = f.replace('.TextGrid', '')
    ans_file = f"{key}a.TextGrid"
    ans_path = os.path.join(answers_dir, ans_file)
    out_path = os.path.join(output_dir, f)
    
    if os.path.exists(ans_path):
        res = calculate_metrics(ans_path, out_path)
        if res: results.append(res)

if results:
    n = len(results)
    avg_hit = sum(r['hit'] for r in results) / n
    avg_eff = sum(r['eff'] for r in results) / n
    avg_swer = sum(r['swer'] for r in results) / n
    
    print(f"Results for {n} files:")
    print(f"Hit Rate: {avg_hit:.4f}")
    print(f"Eff Rate: {avg_eff:.4f}")
    print(f"sWER:     {avg_swer:.4f}")
else:
    print("No results.")

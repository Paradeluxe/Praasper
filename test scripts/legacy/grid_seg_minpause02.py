#!/usr/bin/env python3
"""Grid search seg_dur with min_pause FIXED at 0.2 on subjects 01-02 (8 files)."""
import os, sys, csv, gc

# ── Cache dirs on D: drive ──
IS_WINDOWS = os.name == 'nt'
if IS_WINDOWS:
    os.environ['HF_HOME'] = 'D:\\.cache\\huggingface'
    os.environ['MODELSCOPE_CACHE'] = 'D:\\.cache\\modelscope'
BASE = 'D:/hermes_playground/Praasper' if IS_WINDOWS else '/mnt/d/hermes_playground/Praasper'
AUDIO = 'D:/audio_data' if IS_WINDOWS else '/mnt/d/audio_data'
OUTPUT = 'D:/output/audio_data' if IS_WINDOWS else '/mnt/d/output/audio_data'

sys.path.insert(0, BASE)
from textgrid import TextGrid
import difflib

# ── Paths ──
INPUT_DIR = AUDIO
OUTPUT_DIR = OUTPUT
ANSWERS_DIR = os.path.join(AUDIO, 'answers')
RESULTS_CSV = os.path.join(BASE, 'results', 'grid_seg_minpause02.csv')

# ── Grid: only seg_dur varies, min_pause=0.2 fixed ──
SEG_DUR_VALUES = [5, 10, 15, 20, 30]
MIN_PAUSE = 0.2

# ── Target files (subjects 01-02) ──
TARGET_FILES = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith('.wav') and (f.startswith('01-') or f.startswith('02-'))
])
print(f"Target files ({len(TARGET_FILES)}): {TARGET_FILES}")

# ── Eval logic ──
def calculate_metrics(gt_path, out_path):
    try:
        gt = TextGrid.fromFile(gt_path)
        out = TextGrid.fromFile(out_path)
    except Exception as e:
        return None
    gt_tier = gt[0] if gt else None
    out_tier = out[0] if out else None
    if not gt_tier or not out_tier:
        return None
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
    ratio = difflib.SequenceMatcher(None, gt_text, out_text).ratio()
    sWER = 1 - ratio
    hit_rate = overlap_time / gt_time if gt_time > 0 else 0
    eff_rate = overlap_time / out_time if out_time > 0 else 0
    n_interv = sum(1 for i in out_intervals if i.mark.strip())  # #intval
    return {'hit': hit_rate, 'eff': eff_rate, 'swer': sWER, 'n_interv': n_interv}


def clear_output():
    """Remove all TextGrids from output dir."""
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.TextGrid'):
                os.remove(os.path.join(OUTPUT_DIR, f))


# ── Main: Step 1 - Run auto_vad once per file to find best VAD params ──
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

print("\n" + "="*60)
print("STEP 1: Finding best VAD params via auto_vad (once per file)")
print("="*60)

from praasper import init_model
model = init_model()

file_best_params = {}  # fname -> best VAD params dict

for wav_fname in TARGET_FILES:
    wav_path = os.path.join(INPUT_DIR, wav_fname)
    print(f"\n  Finding best params for {wav_fname}...")
    try:
        model.auto_vad(wav_path=wav_path, file_info=f"  {wav_fname}")
        file_best_params[wav_fname] = model.params.copy()
        print(f"    OK Best params: {model.params['onset']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        continue

print(f"\nauto_vad complete. Found params for {len(file_best_params)}/{len(TARGET_FILES)} files.")

# ── Step 2: Grid search seg_dur × min_pause=0.2 ──
print("\n" + "="*60)
print(f"STEP 2: Grid search seg_dur with min_pause={MIN_PAUSE} (5 combos)")
print("="*60)

all_results = []

for seg_dur in SEG_DUR_VALUES:
    clear_output()
    per_file_metrics = []
    
    print(f"\n--- Combo: seg_dur={seg_dur}s, min_pause={MIN_PAUSE} ---")
    
    for wav_fname in TARGET_FILES:
        if wav_fname not in file_best_params:
            continue
        
        wav_path = os.path.join(INPUT_DIR, wav_fname)
        try:
            model.annote(
                input_path=wav_path,
                seg_dur=seg_dur,
                min_pause=MIN_PAUSE,
                params=file_best_params[wav_fname],
                verbose=False
            )
            
            key = wav_fname.replace('.wav', '')
            ans_file = f"{key}a.TextGrid"
            ans_path = os.path.join(ANSWERS_DIR, ans_file)
            out_path = os.path.join(OUTPUT_DIR, wav_fname.replace('.wav', '.TextGrid'))
            
            if os.path.exists(out_path) and os.path.exists(ans_path):
                res = calculate_metrics(ans_path, out_path)
                if res:
                    per_file_metrics.append(res)
                    print(f"  {wav_fname}: hit={res['hit']:.4f} eff={res['eff']:.4f} sWER={res['swer']:.4f} #intval={res['n_interv']}")
                
                os.remove(out_path)
            
        except Exception as e:
            print(f"  ERROR on {wav_fname}: {e}")
            continue
    
    if per_file_metrics:
        n = len(per_file_metrics)
        metrics = {
            'hit': sum(r['hit'] for r in per_file_metrics) / n,
            'eff': sum(r['eff'] for r in per_file_metrics) / n,
            'swer': sum(r['swer'] for r in per_file_metrics) / n,
            'n_files': n
        }
        row = {
            'seg_dur': seg_dur,
            'min_pause': MIN_PAUSE,
            'hit_rate': round(metrics['hit'], 4),
            'eff_rate': round(metrics['eff'], 4),
            'sWER': round(metrics['swer'], 4),
            'n_files': metrics['n_files']
        }
        all_results.append(row)
        print(f"  AVG: hit={metrics['hit']:.4f} eff={metrics['eff']:.4f} sWER={metrics['swer']:.4f}")

# ── Cleanup ──
model.release_resources()
gc.collect()

# ── Save results ──
if all_results:
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seg_dur', 'min_pause', 'hit_rate', 'eff_rate', 'sWER', 'n_files'])
        writer.writeheader()
        writer.writerows(all_results)
    
    ranked = sorted(all_results, key=lambda x: (-x['hit_rate'], -x['eff_rate'], x['sWER']))
    
    print(f"\n{'='*60}")
    print(f"RESULTS (ranked by hit -> eff -> sWER)")
    print(f"{'='*60}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:2}. seg_dur={r['seg_dur']:>2}s, min_pause={r['min_pause']:.1f}  |  hit={r['hit_rate']:.4f}  eff={r['eff_rate']:.4f}  sWER={r['sWER']:.4f}")
    
    print(f"\nResults saved to {RESULTS_CSV}")
else:
    print("No results to save.")

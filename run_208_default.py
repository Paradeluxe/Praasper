#!/usr/bin/env python3
"""Run Praasper default pipeline (auto_vad grid search) on all 208 audio files.
Scores with pinyin-level sWER against ground truth TextGrids."""
import os, sys, gc, glob
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')
sys.path.insert(0, '/home/maria/.hermes/hermes-agent/venv/lib/python3.12/site-packages/funasr/models/fun_asr_nano')

# Monkey-patch FunASR audio loader
import soundfile as sf
import librosa
def _load_audio_soundfile(file, sr=16000):
    data, file_sr = sf.read(file, dtype='float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data
from funasr.utils import load_utils
load_utils._load_audio_ffmpeg = _load_audio_soundfile

import textgrid as tg_mod
from pypinyin import pinyin, Style
import Levenshtein
from collections import defaultdict
from praasper import init_model

AUDIO_DIR = '/mnt/d/audio_data'
ANSWERS_DIR = '/mnt/d/audio_data/answers'
OUTPUT_DIR = '/mnt/d/output/audio_data'
RESULTS_CSV = '/mnt/d/hermes_playground/Praasper/results_208_default.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Collect all wav files ────────────────────────────────────────────────
wavs = sorted(glob.glob(f'{AUDIO_DIR}/*.wav'))
file_keys = [os.path.splitext(os.path.basename(w))[0] for w in wavs]
print(f'Found {len(file_keys)} wav files')

# ── Pinyin-level sWER (from grid_new_praasper.py) ────────────────────────
def to_pinyin_syls(text):
    text = text.strip().replace(' ', '')
    if not text:
        return []
    py = pinyin(text, style=Style.NORMAL)
    return [p[0] for p in py]

def connected_components(gt_syls, out_syls):
    n, m = len(gt_syls), len(out_syls)
    parent = list(range(n + m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    used_out = set()
    for i, gs in enumerate(gt_syls):
        best_j, best_sim = -1, 0.0
        for j, os_ in enumerate(out_syls):
            if j in used_out:
                continue
            sim = 1 - Levenshtein.distance(gs, os_) / max(len(gs), len(os_), 1)
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_sim >= 0.6 and best_j >= 0:
            union(i, n + best_j)
            used_out.add(best_j)
    groups = defaultdict(lambda: ([], []))
    for i in range(n):
        groups[find(i)][0].append(i)
    for j in range(m):
        groups[find(n + j)][1].append(j)
    return [v for v in groups.values() if v[0]]

def score_pinyin_wer(gt_text, out_text):
    gt_s = to_pinyin_syls(gt_text)
    out_s = to_pinyin_syls(out_text)
    if not gt_s:
        return 0.0 if not out_s else 1.0
    comps = connected_components(gt_s, out_s)
    total_errors, total_len = 0, 0
    for g_idx, o_idx in comps:
        g_part = ''.join(gt_s[i] for i in g_idx)
        o_part = ''.join(out_s[j] for j in o_idx)
        total_errors += Levenshtein.distance(g_part, o_part)
        total_len += max(len(g_part), len(o_part), 1)
    return total_errors / total_len if total_len > 0 else 0.0

# ── Write CSV header ─────────────────────────────────────────────────────
with open(RESULTS_CSV, 'w', encoding='utf-8') as f:
    f.write('file_key,hit_rate,eff_rate,sWER,num_intervals\n')

model = init_model()

total_hit, total_eff, total_swer, count = 0, 0, 0, 0

for i, fk in enumerate(file_keys):
    wav = f'{AUDIO_DIR}/{fk}.wav'
    gt_path = f'{ANSWERS_DIR}/{fk}a.TextGrid'
    out_path = f'{OUTPUT_DIR}/{fk}.TextGrid'

    if not os.path.exists(gt_path):
        print(f'[{i+1}/{len(file_keys)}] SKIP {fk}: no GT TextGrid')
        continue

    print(f'[{i+1}/{len(file_keys)}] {fk} ...', end=' ', flush=True)

    try:
        model.annote(input_path=wav, verbose=False, skip_existing=False, params=None)
    except Exception as e:
        print(f'ERROR: {e}')
        continue

    # Find output TextGrid
    src = out_path
    if not os.path.exists(src):
        for fn in os.listdir(OUTPUT_DIR):
            if fn.startswith(fk) and fn.endswith('.TextGrid'):
                src = f'{OUTPUT_DIR}/{fn}'
                break

    if not os.path.exists(src):
        print('ERROR: no output TextGrid')
        continue

    try:
        gt = tg_mod.TextGrid.fromFile(gt_path)
        out = tg_mod.TextGrid.fromFile(src)
        gt_tier = gt[0]
        out_tier = out[0]

        gt_ivs = [iv for iv in gt_tier.intervals if iv.mark.strip()]
        out_ivs = [iv for iv in out_tier.intervals if iv.mark.strip()]

        gt_time = sum(iv.maxTime - iv.minTime for iv in gt_ivs)
        out_time = sum(iv.maxTime - iv.minTime for iv in out_ivs)

        overlap = 0
        for g in gt_ivs:
            for o in out_ivs:
                s, e = max(g.minTime, o.minTime), min(g.maxTime, o.maxTime)
                if e > s:
                    overlap += e - s

        hit = overlap / gt_time if gt_time > 0 else 0
        eff = overlap / out_time if out_time > 0 else 0

        gt_text = ''.join(iv.mark.strip() for iv in gt_ivs)
        out_text = ''.join(iv.mark.strip() for iv in out_ivs)
        swer = score_pinyin_wer(gt_text, out_text)

        n_intv = len(out_ivs)
        print(f'hit={hit:.4f} eff={eff:.4f} sWER={swer:.4f} #iv={n_intv}')

        with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
            f.write(f'{fk},{hit:.6f},{eff:.6f},{swer:.6f},{n_intv}\n')

        total_hit += hit
        total_eff += eff
        total_swer += swer
        count += 1

        # Clean up output to save disk
        os.remove(src)

    except Exception as e:
        print(f'SCORING ERROR: {e}')

    gc.collect()

# ── Summary ──────────────────────────────────────────────────────────────
print(f'\n=== SUMMARY ===')
print(f'Files scored: {count}/{len(file_keys)}')
if count > 0:
    print(f'Avg hit_rate: {total_hit/count:.4f}')
    print(f'Avg eff_rate: {total_eff/count:.4f}')
    print(f'Avg sWER:     {total_swer/count:.4f}')
print(f'Results: {RESULTS_CSV}')

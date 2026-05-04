#!/usr/bin/env python3
"""Run Praasper default SOP on subjects 01-02 audio files."""
import os, sys, gc, shutil, textgrid
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')

# Monkey-patch FunASR audio loader to use soundfile instead of ffmpeg
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

from praasper import init_model

AUDIO_DIR = '/mnt/d/audio_data'
ANSWERS_DIR = '/mnt/d/audio_data/answers'
OUTPUT_DIR = '/mnt/d/output/audio_data'
RESULTS_CSV = '/mnt/d/hermes_playground/Praasper/results_default_0102.csv'

FILES = ['01-1','01-2','01-3','01-4','02-1','02-2','02-3','02-4']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Write CSV header
with open(RESULTS_CSV, 'w', encoding='utf-8') as f:
    f.write('file_key,hit_rate,eff_rate,sWER,num_intervals\n')

model = init_model()

for fk in FILES:
    wav = f'{AUDIO_DIR}/{fk}.wav'
    gt_path = f'{ANSWERS_DIR}/{fk}a.TextGrid'
    out_path = f'{OUTPUT_DIR}/{fk}.TextGrid'
    
    if not os.path.exists(wav):
        print(f'SKIP {fk}: wav not found')
        continue
    
    print(f'\n=== {fk} ===')
    print(f'  Running auto_vad (min_pause=0 for ranking, then real min_pause=0.2 for annotation)...', flush=True)
    model.annote(input_path=wav, verbose=False, skip_existing=False, params=None)
    
    # Move output to expected path
    src = f'{OUTPUT_DIR}/{fk}.TextGrid'
    if not os.path.exists(src):
        # Find it in output dir
        for fn in os.listdir(OUTPUT_DIR):
            if fn.startswith(fk) and fn.endswith('.TextGrid'):
                src = f'{OUTPUT_DIR}/{fn}'
                break
    
    if os.path.exists(src) and os.path.exists(gt_path):
        gt = textgrid.TextGrid.fromFile(gt_path)
        out = textgrid.TextGrid.fromFile(src)
        gt_tier = gt[0]
        out_tier = out[0]
        
        gt_ivs = [i for i in gt_tier.intervals if i.mark.strip()]
        out_ivs = [i for i in out_tier.intervals if i.mark.strip()]
        
        gt_time = sum(i.maxTime - i.minTime for i in gt_ivs)
        out_time = sum(i.maxTime - i.minTime for i in out_ivs)
        
        overlap = 0
        for g in gt_ivs:
            for o in out_ivs:
                s, e = max(g.minTime, o.minTime), min(g.maxTime, o.maxTime)
                if e > s: overlap += e - s
        
        hit = overlap / gt_time if gt_time > 0 else 0
        eff = overlap / out_time if out_time > 0 else 0
        
        import difflib
        gt_text = ''.join(i.mark.strip() for i in gt_tier.intervals)
        out_text = ''.join(i.mark.strip() for i in out_tier.intervals)
        sWER = 1 - difflib.SequenceMatcher(None, gt_text, out_text).ratio()
        
        n_intv = len(out_ivs)
        print(f'  hit={hit:.4f} eff={eff:.4f} sWER={sWER:.4f} #intval={n_intv}')
        
        with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
            f.write(f'{fk},{hit:.6f},{eff:.6f},{sWER:.6f},{n_intv}\n')
        
        # Cleanup
        os.remove(src)
    else:
        print(f'  ERROR: output={os.path.exists(src)} gt={os.path.exists(gt_path)}')
    
    gc.collect()

print('\n=== DONE ===')
# Summary
with open(RESULTS_CSV) as f:
    lines = f.readlines()[1:]
    print(f'\nResults saved to {RESULTS_CSV}')
    print(f'Files processed: {len(lines)}')

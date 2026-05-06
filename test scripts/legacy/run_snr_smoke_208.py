#!/usr/bin/env python3
"""Smoke test: auto_vad (SNR ranking) on all 208 audio files."""
import sys, os, gc, textgrid, difflib
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')

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

from praasper import init_model

AUDIO_DIR = '/mnt/d/audio_data'
ANSWERS_DIR = '/mnt/d/audio_data/answers'
OUTPUT_DIR = '/mnt/d/output/audio_data'
RESULTS_CSV = '/mnt/d/hermes_playground/Praasper/results_snr_smoke_208.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all wav files
wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
print(f'Found {len(wav_files)} wav files')

# Write CSV header
with open(RESULTS_CSV, 'w', encoding='utf-8') as f:
    f.write('file_key,hit_rate,eff_rate,sWER,num_intervals,snr_dB\n')

model = init_model()

for i, wav_name in enumerate(wav_files):
    fk = os.path.splitext(wav_name)[0]
    wav = f'{AUDIO_DIR}/{wav_name}'
    gt_path = f'{ANSWERS_DIR}/{fk}a.TextGrid'
    out_path = f'{OUTPUT_DIR}/{fk}.TextGrid'

    print(f'\n[{i+1}/{len(wav_files)}] === {fk} ===', flush=True)
    print(f'  Running auto_vad (SNR ranking)...', flush=True)

    # Remove old output if exists
    if os.path.exists(out_path):
        os.remove(out_path)

    model.annote(input_path=wav, verbose=False, skip_existing=False, params=None)

    if not os.path.exists(out_path):
        print(f'  ERROR: no output TextGrid')
        continue

    if os.path.exists(gt_path):
        gt = textgrid.TextGrid.fromFile(gt_path)
        out = textgrid.TextGrid.fromFile(out_path)
        gt_tier = gt[0]
        out_tier = out[0]

        gt_ivs = [i for i in gt_tier.intervals if i.mark.strip()]
        out_ivs = [i for i in out_tier.intervals if i.mark.strip()]

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

        gt_text = ''.join(iv.mark.strip() for iv in gt_tier.intervals)
        out_text = ''.join(iv.mark.strip() for iv in out_tier.intervals)
        sWER = 1 - difflib.SequenceMatcher(None, gt_text, out_text).ratio()

        n_intv = len(out_ivs)
        print(f'  hit={hit:.4f} eff={eff:.4f} sWER={sWER:.4f} #intval={n_intv}', flush=True)

        # Get SNR from model.params (we need to compute it separately)
        # For now, log without SNR
        with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
            f.write(f'{fk},{hit:.6f},{eff:.6f},{sWER:.6f},{n_intv},\n')

        os.remove(out_path)
    else:
        print(f'  ERROR: no ground truth')

    gc.collect()

print('\n=== DONE ===')

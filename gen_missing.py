#!/usr/bin/env python3
import os, sys, glob
sys.path.insert(0, '/mnt/e/praasper')

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

AUDIO_DIR = '/mnt/e/Corpus/ma/audio'
OUTPUT_DIR = '/mnt/e/Corpus/ma/output/audio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

needed = [
    '01-1','01-2','01-3','01-4',
    '02-1','02-2','02-3','02-4',
    '03-1'
]

model = init_model()

for fk in needed:
    wav = f'{AUDIO_DIR}/{fk}.wav'
    out_path = f'{OUTPUT_DIR}/{fk}.TextGrid'
    if os.path.exists(out_path):
        print(f'{fk}: already exists')
        continue
    if not os.path.exists(wav):
        print(f'{fk}: WAV missing')
        continue
    print(f'{fk}: running...', flush=True)
    try:
        model.annote(input_path=wav, verbose=False, skip_existing=False, params=None)
        print(f'{fk}: done')
    except Exception as e:
        print(f'{fk}: ERROR {e}')
